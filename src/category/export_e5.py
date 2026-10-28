"""Export multilingual-e5-small to ONNX (int8) so the service can embed text without torch.

Usage: python -m src.category.export_e5

Writes models/category/e5/{model.onnx, tokenizer.json, ...}. The ONNX graph returns the
token embeddings; mean pooling and L2 normalisation (what sentence-transformers does for
e5) are done in numpy by E5Onnx. A check at the end compares the int8 embeddings with the
original model on a few hundred emails (a handful of fixed texts if the data is not there).
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .common import CAT_MODELS

MODEL = "intfloat/multilingual-e5-small"
OUT = CAT_MODELS / "e5"
MAX_LEN = 256


class E5Onnx:
    """Minimal e5 encoder on onnxruntime + tokenizers (both light, no torch)."""

    def __init__(self, path: Path = OUT, max_len: int = MAX_LEN):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("<pad>"), pad_token="<pad>")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(path / "model.onnx"), opts, providers=["CPUExecutionProvider"])

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch([f"query: {t}" for t in texts[i:i + batch_size]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            h = self.sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32)


def main() -> None:
    import pandas as pd
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(OUT)
    dummy = tok(["query: hello world"], return_tensors="pt")

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "fp32.onnx"
        torch.onnx.export(
            Wrap(model), (dummy["input_ids"], dummy["attention_mask"]), fp32,
            input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"},
                          "last_hidden_state": {0: "b", 1: "t"}},
            opset_version=17, dynamo=False,
        )
        quantize_dynamic(fp32, OUT / "model.onnx", weight_type=QuantType.QInt8)
    for f in OUT.glob("*"):
        if f.name not in {"model.onnx", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}:
            f.unlink() if f.is_file() else shutil.rmtree(f)

    sample = CAT_MODELS.parents[1] / "data" / "category" / "sample.parquet"
    if sample.exists():
        texts = [str(t)[:2000] for t in pd.read_parquet(sample).sample(300, random_state=0).input]
    else:  # fresh clone without data: check on a few fixed texts
        texts = ["Dear PayPal member, please verify your account", "Hi Anna, can we move the call to Friday?",
                 "Скидка 40% на все смартфоны", "We are hiring remote developers", "Hot singles near you"]
    ref = SentenceTransformer(MODEL, device="cpu")
    ref.max_seq_length = MAX_LEN
    a = ref.encode([f"query: {t}" for t in texts], normalize_embeddings=True)
    b = E5Onnx().encode(texts)
    cos = (a * b).sum(1)
    size = (OUT / "model.onnx").stat().st_size / 2**20
    print(f"onnx int8 {size:.0f} MB, cosine to original: mean {cos.mean():.4f}, min {cos.min():.4f}")
    (CAT_MODELS.parents[1] / "reports" / "e5_onnx_check.json").write_text(json.dumps(
        {"onnx_int8_mb": round(size, 1), "n_texts": len(texts), "cosine_mean": round(float(cos.mean()), 4),
         "cosine_min": round(float(cos.min()), 4)}, indent=2))


if __name__ == "__main__":
    main()
