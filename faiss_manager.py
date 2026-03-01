import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import faiss
from torch import nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm.auto import tqdm
from data.load_data import SemiSegDataset
from model import CERSIncomplete

class CoTInferenceModel(nn.Module):
    def __init__(self, bert_path, embed_dim=512):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_path, trust_remote_code=True)
        self.text_projection = nn.Linear(self.bert.config.hidden_size, embed_dim, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        embeds = self.text_projection(cls_token)
        return F.normalize(embeds, p=2, dim=1)


class FaissIndexManager:
    def __init__(self, device: torch.device, bert_model_name_or_path: str, batch_size: int = 16):
        self.device = device
        self.batch_size = batch_size
        self.image_index = None  # faiss.Index for images
        self.text_index = None  # faiss.Index for text
        self.image_embeddings = None  # np.ndarray for images
        self.text_embeddings = None  # np.ndarray for text
        self.image_mappings = []  # List[Dict] for images
        self.text_mappings = []  # List[Dict] for text
        print(f"Loading text model '{bert_model_name_or_path}' to {self.device}...")
        self.text_tokenizer = None
        self.text_model = None
        self._load_bert_model(
            bert_model_name_or_path,
            f"your_path_to_cot_model"
        )

        print(f"FaissIndexManager initialized on device: {self.device} with batch_size: {self.batch_size}")

    def _load_bert_model(
            self,
            original_bert_path,
            trained_adapter_path
    ):
        model = CoTInferenceModel(original_bert_path, embed_dim=512).to(self.device)
        print(f"Loading weights from {trained_adapter_path}...")
        checkpoint = torch.load(trained_adapter_path, map_location=self.device)

        model.bert.load_state_dict(checkpoint['cot_bert'])
        model.text_projection.load_state_dict(checkpoint['text_projection'])
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(original_bert_path)
            self.text_model = model
            self.text_model.to(self.device)
            self.text_model.eval()
        except Exception as e:
            print(f"Error loading text model '{trained_adapter_path}': {e}")
            raise

    @staticmethod
    def load_model(ckpt_path: str, device: torch.device, bert_model_name: str = None,
                   pretrained_backbone=None) -> CERSIncomplete:
        map_location = device
        try:
            ckpt = torch.load(ckpt_path, map_location=map_location)
        except Exception as e:
            print(f"Error loading checkpoint from {ckpt_path}: {e}")
            raise

        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        model = CERSIncomplete(bert_model_name=bert_model_name, pretrained_backbone=pretrained_backbone, device=str(device))
        model.to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def _compute_image_embeddings_for_dataset(self, dataset: SemiSegDataset, image_model: torch.nn.Module) -> (np.ndarray, List[str]):
        image_model.eval()
        embeddings = []
        names = []

        n = len(dataset)
        if n == 0:
            return np.zeros((0, 0)), []

        print(f"Computing IMAGE embeddings for {n} items...")
        for i in tqdm(range(0, n, self.batch_size), desc="Computing IMAGE embeddings"):
            batch_start = i
            batch_end = min(i + self.batch_size, n)

            batch_imgs = []
            batch_texts = []
            batch_names = []

            for b_idx in range(batch_start, batch_end):
                try:
                    img, _, meta = dataset[b_idx]
                    batch_imgs.append(img)
                    batch_texts.extend([meta["text"]])
                    batch_names.append(meta["image"])
                except Exception as e:
                    print(f"Warning: Skipping item {b_idx} due to error: {e}")

            if not batch_imgs:
                continue

            imgs = torch.stack(batch_imgs, dim=0).to(self.device)
            texts = batch_texts
            with torch.no_grad():
                feats = image_model(imgs, texts, return_encoder_feats=True)
                arr = feats.cpu().numpy().astype('float32')
                embeddings.append(arr)
                names.extend(batch_names)

        if len(embeddings) == 0:
            return np.zeros((0, 0)), []

        final_embeddings = np.concatenate(embeddings, axis=0)
        print(f"Finished computing IMAGE embeddings. Shape: {final_embeddings.shape}")
        return final_embeddings, names

    def _compute_text_embeddings_for_dataset(self, dataset: SemiSegDataset) -> (np.ndarray, List[str]):
        if self.text_model is None:
            print("Error: Text model (BERT) is not loaded. Skipping text 'cot' embedding computation.")
            return np.zeros((0, 0)), []

        self.text_model.eval()
        embeddings = []
        names = []

        n = len(dataset)
        if n == 0:
            return np.zeros((0, 0)), []

        print(f"Computing TEXT ('cot') embeddings for {n} items using {self.text_model.bert.config._name_or_path}...")
        for i in tqdm(range(0, n, self.batch_size), desc="Computing TEXT ('cot') embeddings"):
            batch_start = i
            batch_end = min(i + self.batch_size, n)

            batch_texts = []
            batch_cots = []
            batch_names = []

            for b_idx in range(batch_start, batch_end):
                try:
                    _, _, meta = dataset[b_idx]
                    if 'cot' in meta and meta['cot']:
                        batch_texts.append(meta['text'])
                        batch_cots.append(meta['cot'])
                        batch_names.append(meta['image'])
                except Exception as e:
                    print(f"Warning: Skipping item {b_idx} for 'cot' due to error: {e}")

            if not batch_cots:
                continue

            inputs = self.text_tokenizer(
                batch_cots,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=256
            ).to(self.device)

            with torch.no_grad():
                feats = self.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
                arr = feats.cpu().numpy().astype('float32')
                embeddings.append(arr)
                names.extend(batch_names)

        if len(embeddings) == 0:
            return np.zeros((0, 0)), []

        final_embeddings = np.concatenate(embeddings, axis=0)
        print(f"Finished computing TEXT ('cot') embeddings. Shape: {final_embeddings.shape}")
        return final_embeddings, names

    def build_index_from_datasets(self, datasets: List[Tuple[SemiSegDataset, str]], image_model: torch.nn.Module):
        print("Building new index (for images and text)...")
        self.image_index, self.text_index = None, None
        self.image_embeddings, self.text_embeddings = None, None
        self.image_mappings, self.text_mappings = [], []

        all_image_embeddings = []
        all_text_embeddings = []

        for ds, mode in datasets:
            print(f"Processing IMAGE embeddings for mode='{mode}' ({len(ds)} items)")
            img_emb, img_paths = self._compute_image_embeddings_for_dataset(ds, image_model)
            if img_emb.size > 0 and img_emb.shape[0] > 0:
                img_mode_map = [{"path": p, "mode": mode, "type": "image"} for p in img_paths]
                self.image_mappings.extend(img_mode_map)
                all_image_embeddings.append(img_emb)
            else:
                print(f"No image embeddings for mode={mode}, skipping")

            print(f"Processing TEXT ('cot') embeddings for mode='{mode}' ({len(ds)} items)")
            txt_emb, txt_paths = self._compute_text_embeddings_for_dataset(ds)
            if txt_emb.size > 0 and txt_emb.shape[0] > 0:
                txt_mode_map = [{"path": p, "mode": mode, "type": "text_cot"} for p in txt_paths]
                self.text_mappings.extend(txt_mode_map)
                all_text_embeddings.append(txt_emb)
            else:
                print(f"No text ('cot') embeddings for mode={mode}, skipping")

        if all_image_embeddings:
            self.image_embeddings = np.concatenate(all_image_embeddings, axis=0).astype('float32')
            if self.image_embeddings.ndim != 2 or self.image_embeddings.shape[0] == 0:
                print("Error: Invalid shape for image embeddings. Image index not built.")
                self.image_embeddings = None
                self.image_mappings = []
            else:
                d_img = self.image_embeddings.shape[1]
                print(f"Building FAISS IMAGE IndexFlatL2 (n={self.image_embeddings.shape[0]}, dim={d_img})")
                self.image_index = faiss.IndexFlatL2(d_img)
                self.image_index.add(self.image_embeddings)
                print("Image index build complete.")
        else:
            print("No image embeddings found. Image index not built.")

        if all_text_embeddings:
            self.text_embeddings = np.concatenate(all_text_embeddings, axis=0).astype('float32')
            if self.text_embeddings.ndim != 2 or self.text_embeddings.shape[0] == 0:
                print("Error: Invalid shape for text embeddings. Text index not built.")
                self.text_embeddings = None
                self.text_mappings = []
            else:
                d_txt = self.text_embeddings.shape[1]
                print(f"Building FAISS TEXT IndexFlatL2 (n={self.text_embeddings.shape[0]}, dim={d_txt})")
                self.text_index = faiss.IndexFlatL2(d_txt)
                self.text_index.add(self.text_embeddings)
                print("Text index build complete.")
        else:
            print("No text embeddings found. Text index not built.")

    def update_index_ema(self, datasets: List[Tuple[SemiSegDataset, str]], ema_image_model: torch.nn.Module,
                         beta: float = 0.999):
        if self.image_embeddings is None or self.image_index is None:
            print("Error: Image index must be built first before updating.")
            raise ValueError("Image index not built. Call 'build_index_from_datasets' first.")

        print(f"Updating index with EMA (beta={beta}) for IMAGE embeddings only...")

        new_all_image_embeddings = []
        new_all_image_mappings_check = []  # 用于验证

        for ds, mode in datasets:
            print(f"Re-computing IMAGE embeddings for mode='{mode}' using EMA model...")
            new_img_emb, new_img_paths = self._compute_image_embeddings_for_dataset(ds, ema_image_model)

            if new_img_emb.size > 0 and new_img_emb.shape[0] > 0:
                new_all_image_embeddings.append(new_img_emb)
                new_all_image_mappings_check.extend([{"path": p, "mode": mode, "type": "image"} for p in new_img_paths])
            else:
                print(f"Warning: No image embeddings found for mode={mode} during EMA update.")

        if not new_all_image_embeddings:
            print("Error: EMA update failed to produce any new image embeddings.")
            return

        new_image_embeddings_array = np.concatenate(new_all_image_embeddings, axis=0)

        current_image_paths = [m["path"] for m in self.image_mappings]
        new_image_paths_check = [m["path"] for m in new_all_image_mappings_check]

        if len(current_image_paths) != len(new_image_paths_check):
            print(
                f"Error: Image count mismatch during EMA. Old: {len(current_image_paths)}, New: {len(new_image_paths_check)}")
            raise ValueError("Image count mismatch during EMA update.")

        if current_image_paths != new_image_paths_check:
            print("Error: Image path mismatch during EMA update. Dataset order or content has changed.")
            raise ValueError("Image path mismatch during EMA update. Cannot apply EMA.")

        if new_image_embeddings_array.shape != self.image_embeddings.shape:
            print(
                f"Error: Shape mismatch. New embeddings: {new_image_embeddings_array.shape}, Old embeddings: {self.image_embeddings.shape}")
            raise ValueError("Shape mismatch during EMA update.")

        print("Applying EMA to stored IMAGE embeddings...")
        self.image_embeddings = beta * self.image_embeddings + (1 - beta) * new_image_embeddings_array

        norms = np.linalg.norm(self.image_embeddings, axis=1, keepdims=True)
        self.image_embeddings = self.image_embeddings / (norms + 1e-8)

        print(f"Rebuilding FAISS IMAGE index (n={self.image_embeddings.shape[0]}, dim={self.image_index.d})")
        self.image_index.reset()
        self.image_index.add(self.image_embeddings.astype('float32'))
        print("EMA update complete.")

    def _search_index(self, index: faiss.Index, mappings: List[Dict], query_embeddings: np.ndarray, k: int) -> Tuple[
        np.ndarray, np.ndarray, List[List[Dict[str, Any]]]]:
        if index is None:
            print("Error: Index is not built. Cannot search.")
            raise ValueError("Index not built.")

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        query_embeddings = query_embeddings.astype('float32')

        if query_embeddings.shape[1] != index.d:
            print(
                f"Error: Query embedding dimension ({query_embeddings.shape[1]}) does not match index dimension ({index.d}).")
            raise ValueError("Query dimension mismatch")

        D, I = index.search(query_embeddings, k)

        meta_results = []
        for i_batch in I:
            batch_meta = []
            for i in i_batch:
                if 0 <= i < len(mappings):
                    batch_meta.append(mappings[i])
                else:
                    batch_meta.append({"error": "invalid index", "index": i})
            meta_results.append(batch_meta)

        return D, I, meta_results

    def search_image_by_embedding(self, query_embeddings: np.ndarray, k: int = 5) -> Tuple[
        np.ndarray, np.ndarray, List[List[Dict[str, Any]]]]:
        return self._search_index(
            self.image_index,
            self.image_mappings,
            query_embeddings,
            k
        )

    def search_by_text(self, text_queries: List[str], k: int = 5) -> Tuple[
        np.ndarray, np.ndarray, List[List[Dict[str, Any]]]]:
        if self.text_index is None:
            raise ValueError("Text index not built. Cannot search by text.")
        if self.text_model is None:
            raise NotImplementedError("Text model (BERT) is not loaded. Cannot perform text search.")
        if not text_queries:
            print("Warning: Empty text query list.")
            return np.array([]), np.array([]), []

        self.text_model.eval()

        inputs = self.text_tokenizer(
            text_queries,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            feats_tensor = self.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

        query_embeddings_np = feats_tensor.cpu().numpy().astype('float32')

        return self._search_index(
            self.text_index,
            self.text_mappings,
            query_embeddings_np,
            k
        )

    def search_hybrid_visual_then_semantic(
            self,
            image_query_embeddings: np.ndarray,
            text_query_strings: List[str],
            k_coarse: int = 50,
            k_fine: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = image_query_embeddings.shape[0]
        D_coarse, I_coarse, _ = self.search_image_by_embedding(image_query_embeddings, k=k_coarse)

        if self.text_model is None:
            raise ValueError("Text model is required for semantic re-ranking.")

        self.text_model.eval()
        inputs = self.text_tokenizer(
            text_query_strings,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            query_text_feats = self.text_model(inputs.input_ids, inputs.attention_mask)

        query_text_np = query_text_feats.cpu().numpy().astype('float32')  # [B, Dim]
        final_indices_list = []
        final_embeddings_list = []

        for b in range(batch_size):
            candidate_indices = I_coarse[b]  # [50]

            valid_mask = candidate_indices != -1
            valid_indices = candidate_indices[valid_mask]

            if len(valid_indices) == 0:
                final_indices_list.append(np.zeros(k_fine, dtype=int))
                final_embeddings_list.append(np.zeros((k_fine, self.image_embeddings.shape[1]), dtype='float32'))
                continue

            candidate_text_feats = self.text_embeddings[valid_indices]
            curr_q_text = query_text_np[b:b + 1]
            sim_scores = np.matmul(curr_q_text, candidate_text_feats.T).squeeze(0)
            top_k_rel_indices = np.argsort(sim_scores)[-k_fine:][::-1]
            selected_global_indices = valid_indices[top_k_rel_indices]

            if len(selected_global_indices) < k_fine:
                pad_len = k_fine - len(selected_global_indices)
                pad_indices = np.full(pad_len, selected_global_indices[0] if len(selected_global_indices) > 0 else 0)
                selected_global_indices = np.concatenate([selected_global_indices, pad_indices])

            final_indices_list.append(selected_global_indices)
            final_embeddings_list.append(self.image_embeddings[selected_global_indices])

        final_indices = np.array(final_indices_list)  # [B, k_fine]
        final_embeddings = np.array(final_embeddings_list)  # [B, k_fine, Dim_img]

        return final_embeddings, final_indices

    def save_index(self, image_index_path: str, text_index_path: str):
        if self.image_index:
            img_meta_path = image_index_path + ".meta.pkl"
            img_emb_path = image_index_path + ".embeddings.npy"

            print(f"Saving IMAGE FAISS index to {image_index_path}")
            faiss.write_index(self.image_index, image_index_path)
            print(f"Saving IMAGE meta mapping ({len(self.image_mappings)} items) to {img_meta_path}")
            with open(img_meta_path, "wb") as f:
                pickle.dump(self.image_mappings, f)
            print(f"Saving IMAGE embeddings ({self.image_embeddings.shape}) to {img_emb_path}")
            np.save(img_emb_path, self.image_embeddings)
            print("Image index save complete.")
        else:
            print("Image index not built, skipping save.")

        if self.text_index:
            txt_meta_path = text_index_path + ".meta.pkl"
            txt_emb_path = text_index_path + ".embeddings.npy"

            print(f"Saving TEXT FAISS index to {text_index_path}")
            faiss.write_index(self.text_index, text_index_path)
            print(f"Saving TEXT meta mapping ({len(self.text_mappings)} items) to {txt_meta_path}")
            with open(txt_meta_path, "wb") as f:
                pickle.dump(self.text_mappings, f)
            print(f"Saving TEXT embeddings ({self.text_embeddings.shape}) to {txt_emb_path}")
            np.save(txt_emb_path, self.text_embeddings)
            print("Text index save complete.")
        else:
            print("Text index not built, skipping save.")

    def load_index(self, image_index_path: str, text_index_path: str):
        try:
            img_meta_path = image_index_path + ".meta.pkl"
            img_emb_path = image_index_path + ".embeddings.npy"

            print(f"Loading IMAGE FAISS index from {image_index_path}")
            self.image_index = faiss.read_index(image_index_path)
            print(f"Loading IMAGE meta mapping from {img_meta_path}")
            with open(img_meta_path, "rb") as f:
                self.image_mappings = pickle.load(f)
            print(f"Loading IMAGE embeddings from {img_emb_path}")
            self.image_embeddings = np.load(img_emb_path)

            if self.image_index.ntotal != len(self.image_mappings) or self.image_index.ntotal != \
                    self.image_embeddings.shape[0]:
                print(f"Warning: IMAGE data mismatch loaded!")
                print(
                    f"  Index items: {self.image_index.ntotal}, Mappings: {len(self.image_mappings)}, Embeddings: {self.image_embeddings.shape[0]}")
            print(f"Image index load complete. ({self.image_index.ntotal} items)")

        except Exception as e:
            print(f"Error loading IMAGE index from {image_index_path}: {e}")
            self.image_index = None
            self.image_mappings = []
            self.image_embeddings = None

        try:
            txt_meta_path = text_index_path + ".meta.pkl"
            txt_emb_path = text_index_path + ".embeddings.npy"

            print(f"Loading TEXT FAISS index from {text_index_path}")
            self.text_index = faiss.read_index(text_index_path)
            print(f"Loading TEXT meta mapping from {txt_meta_path}")
            with open(txt_meta_path, "rb") as f:
                self.text_mappings = pickle.load(f)
            print(f"Loading TEXT embeddings from {txt_emb_path}")
            self.text_embeddings = np.load(txt_emb_path)

            if self.text_index.ntotal != len(self.text_mappings) or self.text_index.ntotal != \
                    self.text_embeddings.shape[0]:
                print(f"Warning: TEXT data mismatch loaded!")
                print(
                    f"  Index items: {self.text_index.ntotal}, Mappings: {len(self.text_mappings)}, Embeddings: {self.text_embeddings.shape[0]}")
            print(f"Text index load complete. ({self.text_index.ntotal} items)")

        except Exception as e:
            print(f"Error loading TEXT index from {text_index_path}: {e}")
            self.text_index = None
            self.text_mappings = []
            self.text_embeddings = None


    def get_all_image_embeddings_tensor(self):
        if self.image_embeddings is None:
            return None
        return torch.from_numpy(self.image_embeddings).to(self.device)