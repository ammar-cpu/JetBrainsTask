import os
import zipfile
import urllib.request

import numpy as np


TEXT8_URLS = [
    "http://mattmahoney.net/dc/text8.zip",
    "https://data.deepai.org/text8.zip",
    "https://huggingface.co/datasets/afmck/text8/resolve/main/text8.zip",
]


def download_text8(data_dir: str = "data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    txt_path = os.path.join(data_dir, "text8")

    if os.path.exists(txt_path):
        print(f"[data] Found existing {txt_path}")
        return txt_path

    zip_path = os.path.join(data_dir, "text8.zip")

    for url in TEXT8_URLS:
        try:
            print(f"[data] Trying {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, \
                 open(zip_path, "wb") as f:
                f.write(resp.read())

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
            os.remove(zip_path)
            print(f"[data] Saved to {txt_path}")
            return txt_path

        except Exception as e:
            print(f"[data] Failed: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            continue

    raise RuntimeError(
        "Could not download text8 from any mirror.\n"
        "Download manually from https://mattmahoney.net/dc/textdata.html\n"
        "Then run: python3 main.py --corpus path/to/text8"
    )


def load_tokens(path: str, max_tokens: int | None = None) -> list[str]:
    with open(path, encoding="utf-8") as f:
        tokens = f.read().split()
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    print(f"[data] Loaded {len(tokens):,} tokens from {path}")
    return tokens


def generate_skipgram_pairs(
    corpus: list[int],
    window_size: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    n = len(corpus)

    for i in range(n):
        center = corpus[i]
        # dynamic window: closer words get sampled more often
        w = int(rng.integers(1, window_size + 1))
        for j in range(max(0, i - w), min(n, i + w + 1)):
            if j != i:
                pairs.append((center, corpus[j]))

    return pairs