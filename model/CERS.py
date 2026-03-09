from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models, ops
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from model.swin_transformer_unet_skip_expand_decoder_sys import PatchExpand, BasicLayer

class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, text_dim: int, attn_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.text_dim = text_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.q_proj = nn.Conv2d(in_channels, attn_dim, kernel_size=1)
        self.k_proj = nn.Linear(text_dim, attn_dim)
        self.v_proj = nn.Linear(text_dim, attn_dim)
        self.mha = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=False)
        self.out_proj = nn.Conv2d(attn_dim, in_channels, kernel_size=1)
        self.norm = nn.LayerNorm(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))  # residual scaling

    def forward(self, feat: torch.Tensor, text_feats: torch.Tensor, text_mask: Optional[torch.Tensor] = None):
        B, C, H, W = feat.shape
        q = self.q_proj(feat).reshape(B, self.attn_dim, -1).permute(2, 0, 1)  # (T, B, E)
        k = self.k_proj(text_feats).permute(1, 0, 2)
        v = self.v_proj(text_feats).permute(1, 0, 2)
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = ~(text_mask.bool())  # True at positions to be masked
        attn_out, _ = self.mha(q, k, v, key_padding_mask=key_padding_mask)
        attn_out = attn_out.permute(1, 2, 0).reshape(B, self.attn_dim, H, W)
        out = self.out_proj(attn_out)
        out = self.gamma * out + feat
        out_ln = out.permute(0, 2, 3, 1).reshape(B, -1, C)
        out_ln = self.norm(out_ln)
        out = out_ln.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class ConvNeXtTinyEncoder(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        convnext = models.convnext_tiny(weights=weights)
        features = convnext.features
        self.stage1 = nn.Sequential(features[0], features[1])
        self.stage2 = nn.Sequential(features[2], features[3])
        self.stage3 = nn.Sequential(features[4], features[5])
        self.stage4 = nn.Sequential(features[6], features[7])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.stage1(x)  # Out: (B, H/4,  W/4,  96)
        x2 = self.stage2(x1)  # Out: (B, H/8,  W/8,  192)
        x3 = self.stage3(x2)  # Out: (B, H/16, W/16, 384)
        x4 = self.stage4(x3)  # Out: (B, H/32, W/32, 768)
        features = [x1, x2, x3, x4]
        return features


class SwinV2TinyEncoder(nn.Module):
    """
    Wraps torchvision's Swin Transformer V2 Tiny and returns intermediate stage features.

    Returns a list of features [feat1, feat2, feat3, feat4] where:
      feat1: stride 4
      feat2: stride 8
      feat3: stride 16
      feat4: stride 32

    Note:
      - Internal computation is (B, H, W, C).
      - Output is converted to (B, C, H, W).
      - Channel dims are [C1, C2, C3, C4] = [96, 192, 384, 768]
    """
    def __init__(self, weights=None):
        super().__init__()
        swin = models.swin_v2_t(weights=weights)
        features = swin.features
        self.stage1 = nn.Sequential(features[0], features[1])
        self.stage2 = nn.Sequential(features[2], features[3])
        self.stage3 = nn.Sequential(features[4], features[5])
        self.stage4 = nn.Sequential(features[6], features[7])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, 3, H, W)
        x1 = self.stage1(x)  # Out: (B, H/4,  W/4,  96)
        x2 = self.stage2(x1)  # Out: (B, H/8,  W/8,  192)
        x3 = self.stage3(x2)  # Out: (B, H/16, W/16, 384)
        x4 = self.stage4(x3)  # Out: (B, H/32, W/32, 768)

        features = [x1, x2, x3, x4]
        return [f.permute(0, 3, 1, 2).contiguous() for f in features]


class ResNet50Encoder(nn.Module):
    """
    Wraps torchvision's ResNet-50 and returns intermediate stage features.

    Returns a list of features [feat1, feat2, feat3, feat4] where:
      feat1: layer1
      feat2: layer2
      feat3: layer3
      feat4: layer4

    Note: channel dims are [C0, C1, C2, C3, C4] = [64, 256, 512, 1024, 2048]
    """

    def __init__(self, weights: bool = None):
        super().__init__()
        resnet = models.resnet50(weights=weights)
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # outputs C=64
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # outputs C=256
        self.layer2 = resnet.layer2  # outputs C=512
        self.layer3 = resnet.layer3  # outputs C=1024
        self.layer4 = resnet.layer4  # outputs C=2048

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, 3, H, W)
        x = self.conv1(x)
        x0 = self.maxpool(x)
        f1 = self.layer1(x0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return [f1, f2, f3, f4]


class TextEncoder(nn.Module):
    def __init__(self, bert_model_name: str, device='cuda'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name, trust_remote_code=True)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, trust_remote_code=True)
        self.device = device

    def forward(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            max_length=128,
                                                            padding='max_length',
                                                            return_tensors='pt').to(self.device)
        embeddings = self.bert(input_ids=tokenizer_output.input_ids,
                               attention_mask=tokenizer_output.attention_mask,
                               output_hidden_states=True,
                               return_dict=True)
        return embeddings['last_hidden_state'], tokenizer_output.attention_mask


class PrototypeContextModule(nn.Module):
    def __init__(self, in_channels=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.context_attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_context = nn.LayerNorm(in_channels)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, curr_feat_spatial, rag_feats_flat):
        B, C, H, W = curr_feat_spatial.shape
        K = rag_feats_flat.shape[1]
        curr_global = F.adaptive_avg_pool2d(curr_feat_spatial, (1, 1)).view(B, 1, C)
        rag_spatial = rag_feats_flat.view(B * K, C, H, W)
        rag_global = F.adaptive_avg_pool2d(rag_spatial, (1, 1)).view(B, K, C)  # [B, K, C]
        context_vec, _ = self.context_attn(
            query=self.norm_context(curr_global),
            key=self.norm_context(rag_global),
            value=self.norm_context(rag_global)
        )
        global_context = curr_global + context_vec
        scale = self.excitation(global_context.squeeze(1)).view(B, C, 1, 1)
        out = curr_feat_spatial * scale
        out = self.out_proj(out)

        return out + curr_feat_spatial


class SpatialRAGFusionModule(nn.Module):
    def __init__(self, in_channels=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.norm_q = nn.LayerNorm(in_channels)
        self.norm_kv = nn.LayerNorm(in_channels)
        self.gate = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.Sigmoid()
        )

        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, curr_feat_spatial, rag_feats_flat):
        B, C, H, W = curr_feat_spatial.shape
        K = rag_feats_flat.shape[1]
        query_flat = curr_feat_spatial.flatten(2).permute(2, 0, 1)
        query_norm = self.norm_q(query_flat)  # [HW, B, C]
        rag_reshaped = rag_feats_flat.view(B, K, C, -1)
        rag_pixels = rag_reshaped.permute(0, 1, 3, 2).reshape(B, K * H * W, C)
        kv_input = rag_pixels.permute(1, 0, 2)
        kv_norm = self.norm_kv(kv_input)
        attn_out, _ = self.cross_attn(
            query=query_norm,
            key=kv_norm,
            value=kv_norm
        )
        attn_out = attn_out.permute(1, 0, 2)
        query_input = query_flat.permute(1, 0, 2)  # [B, HW, C]
        gate_map = self.gate(torch.cat([query_input, attn_out], dim=-1))
        fused_features = query_input + gate_map * attn_out
        fused_features = fused_features.permute(0, 2, 1).reshape(B, C, H, W)

        return self.out_proj(fused_features)


class CoordAttGatedFusion(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, W)
        mip = max(8, out_channels // reduction)
        self.conv1 = nn.Conv2d(out_channels*2, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.GELU()
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, main_feat, rag_feat):
        feat_sum = torch.cat([main_feat, rag_feat], dim=1)
        B, C, H, W = feat_sum.shape
        x_h = self.pool_h(feat_sum)  # [B, C, H, 1]
        x_w = self.pool_w(feat_sum).permute(0, 1, 3, 2)  # [B, C, W, 1]
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [B, C, 1, W]
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        gate = a_h * a_w
        out_rag = gate * rag_feat
        out = main_feat + out_rag

        return self.out_conv(out), out_rag


class ResUNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(ResUNetDecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        combined_channels = in_channels + skip_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(combined_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(combined_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.upsample(x)
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels=768, skip_channels=0, out_channels=384, reduction=16):
        super().__init__()
        self.up_main = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_rag = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_main = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1),
            nn.GELU(),
        )
        self.conv_rag = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
        )

        self.CAGF = CoordAttGatedFusion(
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=out_channels,
            reduction=reduction)
        self.out_conv_main_1 = nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels)
        self.out_ln_main = nn.LayerNorm(out_channels)
        self.out_conv_main_2 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1)
        self.out_act_main = nn.GELU()

        self.out_conv_rag_1 = nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels)
        self.out_ln_rag = nn.LayerNorm(out_channels)
        self.out_conv_rag_2 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1)
        self.out_act_rag = nn.GELU()

        self.out_conv_shared_1 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels)
        self.out_ln_shared = nn.LayerNorm(out_channels)
        self.out_conv_shared_2 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1)
        self.out_act_shared = nn.GELU()

        self.out_conv_main_3 = nn.Conv2d(out_channels*4, out_channels, kernel_size=1)
        self.out_conv_rag_3 = nn.Conv2d(out_channels*4, out_channels, kernel_size=1)


    def forward(self, x: Tensor, skip: Tensor, rag_input: Optional[torch.Tensor] = None) -> tuple[Tensor, Tensor]:
        if skip is not None:
            x_skip = torch.cat([x, skip], dim=1)
        else:
            x_skip = x

        if rag_input is not None:
            x_skip = self.up_main(x_skip)
            x_skip = self.conv_main(x_skip)
            rag_input = self.up_rag(rag_input)
            rag_input = self.conv_rag(rag_input)
            main, rag = self.CAGF(x_skip, rag_input)

        else:
            x_skip = self.up_main(x_skip)
            x_skip = self.conv_main(x_skip)
            x = self.up_rag(x)
            x = self.conv_rag(x)
            main, rag = self.CAGF(x_skip, x)

        out_main_1 = self.out_conv_main_1(main)
        out_main_1 = out_main_1.permute(0, 2, 3, 1)
        out_main_1 = self.out_ln_main(out_main_1)
        out_main_1 = out_main_1.permute(0, 3, 1, 2)
        out_main_1 = self.out_conv_main_2(out_main_1)
        out_main_1 = self.out_act_main(out_main_1)

        out_rag_1 = self.out_conv_rag_1(rag)
        out_rag_1 = out_rag_1.permute(0, 2, 3, 1)
        out_rag_1 = self.out_ln_rag(out_rag_1)
        out_rag_1 = out_rag_1.permute(0, 3, 1, 2)
        out_rag_1 = self.out_conv_rag_2(out_rag_1)
        out_rag_1 = self.out_act_rag(out_rag_1)

        out_main_shared = self.out_conv_shared_1(main)
        out_main_shared = out_main_shared.permute(0, 2, 3, 1)
        out_main_shared = self.out_ln_shared(out_main_shared)
        out_main_shared = out_main_shared.permute(0, 3, 1, 2)
        out_main_shared = self.out_conv_shared_2(out_main_shared)
        out_main_shared = self.out_act_shared(out_main_shared)

        out_rag_shared = self.out_conv_shared_1(rag)
        out_rag_shared = out_rag_shared.permute(0, 2, 3, 1)
        out_rag_shared = self.out_ln_shared(out_rag_shared)
        out_rag_shared = out_rag_shared.permute(0, 3, 1, 2)
        out_rag_shared = self.out_conv_shared_2(out_rag_shared)
        out_rag_shared = self.out_act_shared(out_rag_shared)

        out_main = self.out_conv_main_3(out_main_1 + out_main_shared) + main
        out_rag = self.out_conv_rag_3(out_rag_1 + out_rag_shared) + rag

        return out_main, out_rag


def xavier_init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            nn.init.xavier_uniform_(m.in_proj_weight)
        if m.out_proj is not None:
            nn.init.xavier_uniform_(m.out_proj.weight)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)


class CERSIncomplete(nn.Module):
    def __init__(self,
                 bert_model_name: str,
                 backbone_type: str = 'resnet50',
                 pretrained_backbone: bool = None,
                 attn_dim: int = 256,
                 num_heads: int = 8,
                 decoder_channels: Optional[List[int]] = None,
                 device='cuda'):
        super().__init__()

        # text encoder
        self.text_encoder = TextEncoder(bert_model_name=bert_model_name, device=device)
        self.bert_hidden_size = self.text_encoder.bert_hidden_size

        # Image Encoder Selection
        self.decoder_channels = decoder_channels
        if backbone_type == 'resnet50':
            self.encoder = ResNet50Encoder(weights=pretrained_backbone)
            # ResNet returns [256, 512, 1024, 2048] corresponding to stride 4, 8, 16, 32
            stage_chs = [256, 512, 1024, 2048]
            if self.decoder_channels is None:
                self.last_size = 14
                self.decoder_channels = [1024, 512, 256, 128, 64]
        elif backbone_type == 'swin_v2_t' or backbone_type == 'convnext_t':
            self.encoder = SwinV2TinyEncoder(weights=pretrained_backbone)
            # Swin-T returns [96, 192, 384, 768]
            stage_chs = [96, 192, 384, 768]
            if self.decoder_channels is None:
                self.last_size = 7
                self.decoder_channels = [384, 192, 96, 48, 24]
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # cross-attention blocks for each encoder output
        self.cross_attns = nn.ModuleList()
        for c in stage_chs:
            self.cross_attns.append(CrossAttentionBlock(in_channels=c,
                                                        text_dim=self.bert_hidden_size,
                                                        attn_dim=attn_dim,
                                                        num_heads=num_heads))

        self.decoder_left4 = DecoderBlock(in_channels=stage_chs[3], skip_channels=0, out_channels=self.decoder_channels[0])
        self.decoder_left3 = DecoderBlock(in_channels=self.decoder_channels[0], skip_channels=stage_chs[2], out_channels=self.decoder_channels[1])
        self.decoder_left2 = DecoderBlock(in_channels=self.decoder_channels[1], skip_channels=stage_chs[1], out_channels=self.decoder_channels[2])
        self.decoder_left1 = DecoderBlock(in_channels=self.decoder_channels[2], skip_channels=stage_chs[0],out_channels=self.decoder_channels[3])

        # final upsample to original resolution
        self.final_up_left = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv_left = nn.Sequential(
            nn.Conv2d(self.decoder_channels[2], self.decoder_channels[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.decoder_channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.decoder_channels[4], 1, kernel_size=1)
        )

        self.decoder_right4 = ResUNetDecoderBlock(in_channels=stage_chs[3], skip_channels=0, out_channels=self.decoder_channels[0])
        self.decoder_right3 = ResUNetDecoderBlock(in_channels=self.decoder_channels[0], skip_channels=stage_chs[2], out_channels=self.decoder_channels[1])
        self.decoder_right2 = ResUNetDecoderBlock(in_channels=self.decoder_channels[1], skip_channels=stage_chs[1], out_channels=self.decoder_channels[2])
        self.decoder_right1 = ResUNetDecoderBlock(in_channels=self.decoder_channels[2], skip_channels=stage_chs[0], out_channels=self.decoder_channels[3])

        self.final_up_right = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv_right = nn.Sequential(
            nn.Conv2d(self.decoder_channels[3], self.decoder_channels[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.decoder_channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.decoder_channels[4], 1, kernel_size=1)
        )

        self.cross_attns.apply(xavier_init_weights)

        self.decoder_left4.apply(xavier_init_weights)
        self.decoder_left3.apply(xavier_init_weights)
        self.decoder_left2.apply(xavier_init_weights)
        self.decoder_left1.apply(xavier_init_weights)
        self.final_conv_left.apply(xavier_init_weights)

        self.decoder_right4.apply(xavier_init_weights)
        self.decoder_right3.apply(xavier_init_weights)
        self.decoder_right2.apply(xavier_init_weights)
        self.decoder_right1.apply(xavier_init_weights)
        self.final_conv_right.apply(xavier_init_weights)
        print("Xavier initialization applied to Decoder, CrossAttn, and Head.")

    def forward(self, image: Tensor, text, return_encoder_feats: bool = False, return_all: bool = False) \
            -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
        feats = self.encoder(image)
        text_feats, text_mask = self.text_encoder(text)

        attn_feats = []
        for feat, attn in zip(feats, self.cross_attns):
            attn_feats.append(attn(feat, text_feats, text_mask))

        if return_encoder_feats:
            return feats[-1].view(feats[-1].size(0), -1)

        d4l, d4xl = self.decoder_left4(attn_feats[3], None, None)  # uses f4
        d3l, d3xl = self.decoder_left3(d4l, attn_feats[2], d4xl)  # uses f3
        d2l, d2xl = self.decoder_left2(d3l, attn_feats[1], d3xl)  # uses f2
        d1l, d1xl = self.decoder_left1(d2l, attn_feats[0], d2xl)  # uses f1

        outl = self.final_up_left(torch.cat([d1l, d1xl], dim=1))
        outl = self.final_conv_left(outl)

        d4r = self.decoder_right4(attn_feats[3], None)  # uses f4
        d3r = self.decoder_right3(d4r, attn_feats[2])  # uses f3
        d2r = self.decoder_right2(d3r, attn_feats[1])  # uses f2
        d1r = self.decoder_right1(d2r, attn_feats[0])  # uses f1

        outr = self.final_up_right(d1r)
        outr = self.final_conv_right(outr)

        if return_all:
            return outl, outr, feats[-1].view(feats[-1].size(0), -1)

        return outl, outr


class CERS(CERSIncomplete):
    def __init__(self,
                 bert_model_name: str,
                 backbone_type: str = 'resnet50',
                 pretrained_backbone: bool = None,
                 attn_dim: int = 256,
                 num_heads: int = 8,
                 decoder_channels: Optional[List[int]] = None,
                 device='cuda'):
        super().__init__(bert_model_name=bert_model_name,
                         backbone_type=backbone_type,
                         pretrained_backbone=pretrained_backbone,
                         attn_dim=attn_dim,
                         num_heads=num_heads,
                         decoder_channels=decoder_channels,
                         device=device)

        self.fusion_module = SpatialRAGFusionModule(self.decoder_channels[0] * 2, 8, 0.1)
        self.fusion_module.apply(xavier_init_weights)
        print("Xavier initialization applied to SpatialRAGFusionModule (RAG Fusion).")


    def forward(self, image: Tensor, text, return_encoder_feats: bool = False,
                feats_rag: Optional[Tensor] = None) -> Tuple[Tensor, Tensor] | Tensor:
        feats = self.encoder(image)
        text_feats, text_mask = self.text_encoder(text)
        attn_feats = []
        for feat, attn in zip(feats, self.cross_attns):
            attn_feats.append(attn(feat, text_feats, text_mask))
        feats_last = feats[-1].view(feats[-1].size(0), -1)
        if return_encoder_feats:
            return feats_last
        feats_combine = self.fusion_module(feats[-1], feats_rag)
        d4l, d4xl = self.decoder_left4(attn_feats[3], None, feats_combine)  # uses f4
        d3l, d3xl = self.decoder_left3(d4l, attn_feats[2], d4xl)  # uses f3
        d2l, d2xl = self.decoder_left2(d3l, attn_feats[1], d3xl)  # uses f2
        d1l, d1xl = self.decoder_left1(d2l, attn_feats[0], d2xl)  # uses f1
        outl = self.final_up_left(torch.cat([d1l, d1xl], dim=1))
        outl = self.final_conv_left(outl)

        d4r = self.decoder_right4(attn_feats[3], None)  # uses f4
        d3r = self.decoder_right3(d4r, attn_feats[2])  # uses f3
        d2r = self.decoder_right2(d3r, attn_feats[1])  # uses f2
        d1r = self.decoder_right1(d2r, attn_feats[0])  # uses f1

        outr = self.final_up_right(d1r)
        outr = self.final_conv_right(outr)


        return outl, outr
