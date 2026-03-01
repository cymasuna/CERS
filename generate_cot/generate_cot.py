import io
import os
import json
import csv
import base64
import time
import random
import sys
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
from tqdm import tqdm

# -------- CONFIG ----------
FRAME_DIR = "your_data_root/frames/"
MASK_DIR = "your_data_root/masks/"
MAPPING_CSV = "your_data_root/text.csv"   # CSV with two columns: image, text
JSON_OUT_DIR = "your_data_root/reports/"
CSV_OUTPUT = "your_data_root/reports/cot_output_gpt5_mask.csv"
COMBINED_JSON = os.path.join(JSON_OUT_DIR, "combined_messages_mask.json")
DEFAULT_TEXT = ""
MODALITY = "MRI"
ANATOMY = "Brain"
CLASSES = "brain tumor"
MODEL_NAME = "your_llm"
DASHSCOPE_API_KEY = "your_api_key"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
USE_MASK = True
# --------------------------

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="your_url",
)

def _is_rate_limit_error(exc: Exception) -> bool:
    """Heuristic: check exception text for rate-limit indicators."""
    txt = str(exc).lower()
    if "limit_requests" in txt or "rate limit" in txt or "429" in txt:
        return True
    return False

def stream_with_retries(make_stream_callable, max_retries: int = 6, base_delay: float = 1.0, max_delay: float = 60.0):
    """
    Robust streamer with exponential backoff and jitter.
    - make_stream_callable: a zero-arg callable that returns the stream/generator from client.chat.completions.create(...)
      e.g. lambda: client.chat.completions.create(model=..., messages=..., stream=True)
    - Yields the same chunks as the original stream.
    - On rate-limit / transient error, it will wait and retry the whole call (restarts the stream).
    NOTE: this restarts the stream from scratch on retry (can't resume mid-stream).
    """
    attempt = 0
    while True:
        try:
            stream = make_stream_callable()
            # iterate through generator and yield chunks
            for chunk in stream:
                yield chunk
            # if stream completes normally, break
            return
        except Exception as e:
            attempt += 1
            is_rate = _is_rate_limit_error(e)
            if not is_rate or attempt > max_retries:
                # raise original exception if not rate-limit or out of retries
                raise
            # exponential backoff with jitter
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            jitter = random.uniform(0, delay * 0.25)
            wait = delay + jitter
            print(f"[WARN] Rate limit or transient error detected (attempt {attempt}/{max_retries}). "
                  f"Sleeping {wait:.1f}s then retrying...", file=sys.stderr)
            time.sleep(wait)
            # loop will retry

def read_csv_mapping(csv_path: str) -> Dict[str, str]:
    """
    Read CSV and return mapping: filename -> text.
    Accepts headers case-insensitively ('image' and 'text').
    """
    mapping: Dict[str, str] = {}
    if not os.path.isfile(csv_path):
        print(f"[WARN] CSV mapping file not found: {csv_path}. Using default empty texts.")
        return mapping

    with open(csv_path, "r", encoding="utf-8-sig", newline='') as f:
        reader = csv.DictReader(f)
        # normalize header names
        headers = {h.lower(): h for h in reader.fieldnames or []}
        img_key = headers.get("image")
        text_key = headers.get("text")

        for row in reader:
            img_name = row.get(img_key, "").strip()
            txt = row.get(text_key, "").strip()
            if img_name:
                mapping[img_name] = txt
    return mapping

def create_overlay_image(image_path, mask_path, alpha=0.5, color=(0, 0, 255)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if img.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    colored_mask = np.full_like(img, color, dtype=np.uint8)
    blended = cv2.addWeighted(img, 1.0 - alpha, colored_mask, alpha, 0)
    overlay_result = img.copy()
    cv2.copyTo(blended, mask, overlay_result)

    return Image.fromarray(cv2.cvtColor(overlay_result, cv2.COLOR_BGR2RGB))

def encode_pil_image(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def build_user_content(image_path: str, mask_path: str, report_text: str, modality: str, anatomy: str, classes: str):
    """Return content list for a single user message (merged instruction + inputs)."""
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    image_abs = os.path.abspath(image_path)

    if USE_MASK:
        overlay_img = create_overlay_image(image_path, mask_path, alpha=0.2, color=(0, 0, 255))
        img_b64 = encode_pil_image(overlay_img)

        text_block = (
            "You are an expert radiologist assisting a segmentation model. "
            "Analyze the provided medical image crop.\n"
            f"Clinical report text: \"{report_text}\"\n"
            f"This is a {modality} image of {anatomy} with {classes} present. \n\n"
            "Provide a structured analysis (strictly under 128 tokens):\n"
            "Step 1: Analyze the **Texture**. Is it homogeneous or heterogeneous? Is it noisy?\n"
            "Step 2: Analyze the **Boundary**. Is the edge with the background sharp or blurry? Is there low contrast?\n"        
            "Step 3: Analyze the **Context**. Are there similar-looking organs nearby that might cause confusion?\n"       
            "Step 4: Based on the above, summarize the **Visual Characteristics** in one sentence.\n\n"
            "Important:\n"
            '- Use standard medical visual descriptors (e.g., "ground-glass opacity", "hypointense", "spiculated").\n'
            '- Do NOT mention spatial coordinates (e.g., "top left") as they are irrelevant for retrieval.\n'
            '- Focus on what makes segmentation difficult.\n\n'
            "Example Output:\n"
            "Texture: Heterogeneous with calcification.\n"
            "Boundary: Indistinct margins blending with liver parenchyma.\n"
            "Context: Close proximity to the portal vein.\n"
            "Summary: Heterogeneous calcified lesion with blurry boundaries near vessels."
        )
    else:
        img_b64 = encode_image(image_abs)
        text_block = (
            "You are an expert radiologist assisting a segmentation model. "
            "Analyze the provided CT image with a semi-transparent RED OVERLAY indicating the ground-truth lesion.\n"
            f"Clinical report text: \"{report_text}\"\n"
            f"This is a {modality} image of {anatomy} with {classes} present. \n\n"
            "Provide a structured analysis (strictly under 128 tokens):\n"
            "Step 1: Analyze the **Texture**. Is it homogeneous or heterogeneous? Is it noisy?\n"
            "Step 2: Analyze the **Boundary**. Is the edge with the background sharp or blurry? Is there low contrast?\n"
            "Step 3: Analyze the **Context**. Are there similar-looking organs nearby that might cause confusion?\n"
            "Step 4: Based on the above, summarize the **Visual Characteristics** in one sentence.\n\n"
            "Important:\n"
            '- Use standard medical visual descriptors (e.g., "ground-glass opacity", "hypointense", "spiculated").\n'
            '- Do NOT mention spatial coordinates (e.g., "top left") as they are irrelevant for retrieval.\n'
            '- Focus on what makes segmentation difficult.\n\n'
            "Example Output:\n"
            "Texture: Heterogeneous with calcification.\n"
            "Boundary: Indistinct margins blending with liver parenchyma.\n"
            "Context: Close proximity to the portal vein.\n"
            "Summary: Heterogeneous calcified lesion with blurry boundaries near vessels."
        )
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        {"type": "text", "text": text_block},
    ]
    return content

def build_messages_for_all(frame_dir: str, mask_dir: str, mapping: Dict[str, str], modality: str, anatomy: str, classes: str):
    """Scan frame_dir and mask_dir, create list of entries for combined JSON."""
    entries = []
    if not os.path.isdir(frame_dir):
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    frame_files = sorted(os.listdir(frame_dir))

    for fname in tqdm(frame_files):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in IMG_EXTS:
            continue
        frame_path = os.path.join(frame_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.isfile(mask_path):
            print(f"[WARN] Mask not found for {fname}, skipping.")
            continue

        # find report text from CSV mapping; try exact filename match first, then basename match
        report_text = mapping.get(fname, mapping.get(base, DEFAULT_TEXT))

        content = build_user_content(frame_path, mask_path, report_text, modality, anatomy, classes)
        messages = [{"role": "user", "content": content}]
        entries.append({"case": base, "messages": messages})

    return entries

def save_combined_json(entries: List[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fw:
        json.dump(entries, fw, ensure_ascii=False, indent=2)
    print(f"[SAVED] combined messages -> {out_path}")

def load_combined_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fr:
        return json.load(fr)

def stream_and_print_answer(messages):
    """
    Send a streaming request for the given messages and print ONLY assistant textual content.
    Returns the aggregated assistant text for potential logging.
    """
    assistant_text = ""
    completion = lambda: client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        # temperature=0.0,  # you can set determinism if desired
    )

    for chunk in stream_with_retries(completion, max_retries=10, base_delay=1.0):
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = choices[0].delta
        content_piece = None
        if isinstance(delta, dict):
            content_piece = delta.get("content") or delta.get("text")
        else:
            content_piece = getattr(delta, "content", None) or getattr(delta, "text", None)

        if content_piece:
            print(content_piece, end="", flush=True)
            assistant_text += content_piece

    print("\n" + "-" * 40)
    return assistant_text

def append_result_to_csv(case: str, cot_text: str):
    file_exists = os.path.isfile(CSV_OUTPUT)
    # Open in append mode and write header only if file didn't exist
    with open(CSV_OUTPUT, "a", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image", "cot"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"image": case, "cot": cot_text})

def run_from_combined_file(combined_json_path: str):
    entries = load_combined_json(combined_json_path)
    if not entries:
        print("[INFO] No entries found in combined json.")
        return

    results = []
    for idx, entry in enumerate(entries, 1):
        case = entry.get("case", f"case_{idx}")
        messages = entry.get("messages")
        if not messages:
            print(f"[WARN] No messages for {case}, skipping.")
            continue

        print(f"[RUNNING] ({idx}/{len(entries)}) case: {case}")
        try:
            assistant_text = stream_and_print_answer(messages)
        except Exception as e:
            print(f"[ERROR] streaming failed for {case}: {e}")
            # optionally capture stack for debugging
            assistant_text = f"[ERROR] {e}"

        # Append to results list for CSV
        results.append({"case": case, "cot": assistant_text})

        append_result_to_csv(case, assistant_text)
        print(f"[SAVED] case={case} -> {CSV_OUTPUT}")


def main():
    # Step 1: read CSV mapping
    mapping = read_csv_mapping(MAPPING_CSV)
    print(f"[INFO] Loaded {len(mapping)} entries from CSV mapping.")

    # Step 2: build combined messages
    entries = build_messages_for_all(FRAME_DIR, MASK_DIR, mapping, MODALITY, ANATOMY, CLASSES)
    if not entries:
        print("[INFO] No valid image+mask pairs found. Exiting.")
        return
    save_combined_json(entries, COMBINED_JSON)

    # Step 3: run model using combined json
    run_from_combined_file(COMBINED_JSON)


if __name__ == "__main__":
    main()