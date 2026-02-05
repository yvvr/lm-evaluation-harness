#!/usr/bin/env python
"""Evaluate Bits-Per-Character (BPC) for texts in a JSON/TXT file under a Hugging Face causal LM.

This mirrors `scripts/python/eval_perplexity_hf.py` but reports BPC:

  BPC = (total_nll_nats / total_characters) / ln(2)

Notes:
- Hugging Face CausalLM loss is an average negative log-likelihood in *nats* over predicted tokens.
  We convert to a summed token NLL (nats) and normalize by character count.
- This is a common way to report BPC for token-based LMs, even though the LM operates on tokens.

Input formats:
- .json: top-level object with a `transcripts` array; each element should have a `corrected_data` field
- .txt: one non-empty sentence per line
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Bits-Per-Character (BPC) under a Hugging Face causal LM")

    parser.add_argument(
        "--model",
        required=True,
        help="Model name or path (e.g., gpt2, google/gemma-3-27b-pt, /path/to/model)",
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help=(
            "Path to input to evaluate. Supports: "
            "(1) .json: object with a 'transcripts' array; each element has 'corrected_data'; "
            "(2) .txt: one sentence per line."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of records/lines to score (0 = all)",
    )

    parser.add_argument(
        "--report_every",
        type=int,
        default=50,
        help="Progress report frequency (default: 50)",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on (default: auto)",
    )

    parser.add_argument(
        "--device_map",
        default=None,
        help=(
            "Optional Transformers device_map for large models (e.g., 'auto'). "
            "When set, the model may be sharded/offloaded and --device is ignored for model placement."
        ),
    )

    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Computation dtype (default: auto). fp16/bf16 require GPU support.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=0,
        help=(
            "Max tokens per forward pass. If input is longer, use sliding-window scoring. "
            "0 uses the model context cap (recommended)."
        ),
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help=(
            "Sliding-window stride in tokens (step size). 0 picks a default (half of max_length). "
            "Smaller = more accurate but slower."
        ),
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom model/tokenizer code from the Hub.",
    )

    return parser.parse_args(argv)


@dataclass
class BPCResult:
    bits_per_char: float
    mean_nll_per_char_nats: float
    num_predicted_tokens: int


def _read_json_texts(
    *,
    json_file: str,
    json_array_key: str,
    json_field: str,
    limit: int,
) -> tuple[list[str], int]:
    with open(json_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if json_array_key not in payload:
            raise KeyError(f"JSON object has no key '{json_array_key}'")
        records = payload[json_array_key]
    else:
        raise TypeError("JSON must be an array or an object containing an array")

    if not isinstance(records, list):
        raise TypeError(f"JSON '{json_array_key}' must be a list")

    total_records = len(records)
    if limit and limit > 0:
        records = records[:limit]

    texts: list[str] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise TypeError(f"Record at index {idx} is not an object")
        val = rec.get(json_field)
        if val is None:
            continue
        if not isinstance(val, str):
            val = str(val)
        val = val.strip()
        if not val:
            continue
        texts.append(val)

    return texts, total_records


def _read_txt_texts(*, txt_file: str, limit: int) -> tuple[list[str], int]:
    with open(txt_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    total_lines = len(lines)
    if limit and limit > 0:
        lines = lines[:limit]

    texts: list[str] = []
    for line in lines:
        val = line.strip()
        if not val:
            continue
        texts.append(val)

    return texts, total_lines


def _count_words(text: str) -> int:
    return len(text.split())


def _count_chars(text: str) -> int:
    # Count Unicode codepoints (Python str length), excluding newlines (already stripped/split).
    return len(text)


def _nll_sum_for_input_ids(*, model: Any, input_ids) -> tuple[float, int]:
    """Return (nll_sum_nats, num_predicted_tokens) for a single encoded sequence."""
    import torch

    seq_len = input_ids.size(1)
    num_pred_tokens = int(seq_len - 1)
    if num_pred_tokens <= 0:
        return 0.0, 0

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, use_cache=False)
        loss = outputs.loss

    nll_sum = float(loss) * num_pred_tokens
    return nll_sum, num_pred_tokens


def _nll_sum_for_input_ids_sliding_window(
    *,
    model: Any,
    input_ids,
    max_length: int,
    stride: int,
) -> tuple[float, int]:
    """Return (nll_sum_nats, num_predicted_tokens) using sliding-window scoring."""
    import torch

    seq_len = int(input_ids.size(1))
    if seq_len <= 1:
        return 0.0, 0

    if max_length <= 0:
        raise ValueError("max_length must be > 0 for sliding-window scoring")
    if stride <= 0:
        raise ValueError("stride must be > 0 for sliding-window scoring")
    if stride > max_length:
        stride = max_length

    nll_sum = 0.0
    total_pred_tokens = 0

    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)

        new_tokens = end - prev_end
        if new_tokens <= 0:
            break

        input_window = input_ids[:, begin:end]
        target_ids = input_window.clone()
        if new_tokens < int(target_ids.size(1)):
            target_ids[:, :-new_tokens] = -100

        num_pred_tokens = int((target_ids[:, 1:] != -100).sum().item())
        if num_pred_tokens <= 0:
            prev_end = end
            if end >= seq_len:
                break
            continue

        with torch.no_grad():
            outputs = model(input_window, labels=target_ids, use_cache=False)
            loss = outputs.loss

        nll_sum += float(loss) * num_pred_tokens
        total_pred_tokens += num_pred_tokens

        prev_end = end
        if end >= seq_len:
            break

    return nll_sum, total_pred_tokens


def _pick_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def _pick_dtype(dtype: str, device: str):
    if dtype == "auto":
        try:
            import torch

            if device == "cpu":
                return torch.float32
            if device == "mps":
                return torch.float16
            if device == "cuda":
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
        except Exception:
            return None

    try:
        import torch

        mapping = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return mapping[dtype]
    except Exception:
        return None


def _get_context_cap(*, model: Any, tokenizer: Any) -> int:
    cap = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(cap, int) and cap > 0:
        return cap

    tok_cap = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_cap, int) and 0 < tok_cap < 1_000_000:
        return tok_cap

    return 4096


def _resolve_window_params(*, model: Any, tokenizer: Any, max_length: int, stride: int) -> tuple[int, int]:
    context_cap = _get_context_cap(model=model, tokenizer=tokenizer)
    resolved_max_length = context_cap if (max_length is None or max_length <= 0) else min(int(max_length), context_cap)
    resolved_stride = int(stride) if (stride is not None and stride > 0) else max(1, resolved_max_length // 2)
    if resolved_stride > resolved_max_length:
        resolved_stride = resolved_max_length
    return resolved_max_length, resolved_stride


def compute_bpc_json_hf(
    *,
    model_name_or_path: str,
    json_file: str,
    limit: int = 0,
    report_every: int = 50,
    device: str = "auto",
    device_map: Optional[str] = None,
    dtype: str = "auto",
    max_length: int = 0,
    stride: int = 0,
    trust_remote_code: bool = False,
) -> tuple[BPCResult, dict[str, int]]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies. Install at least: transformers, torch. "
            "(Optionally: accelerate, sentencepiece for some models.)"
        ) from e

    resolved_device = _pick_device(device)
    resolved_dtype = _pick_dtype(dtype, resolved_device)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        token=hf_token,
        use_fast=True,
    )

    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        token=hf_token,
        torch_dtype=resolved_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

    model.eval()

    if device_map is None:
        if resolved_device == "cuda":
            model.to("cuda")
        elif resolved_device == "mps":
            model.to("mps")
        else:
            model.to("cpu")

    resolved_max_length, resolved_stride = _resolve_window_params(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )

    json_array_key = "transcripts"
    json_field = "corrected_data"

    texts, total_records = _read_json_texts(
        json_file=json_file,
        json_array_key=json_array_key,
        json_field=json_field,
        limit=limit,
    )
    if not texts:
        raise ValueError(f"No non-empty '{json_field}' texts found")

    nll_sum = 0.0
    total_pred_tokens = 0
    total_input_tokens = 0
    total_words = 0
    total_chars = 0

    for i, text in enumerate(texts, start=1):
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"]
        total_input_tokens += int(input_ids.numel())
        total_words += _count_words(text)
        total_chars += _count_chars(text)
        input_ids = input_ids.to(model.device)

        if int(input_ids.size(1)) > resolved_max_length:
            nll_i, tok_i = _nll_sum_for_input_ids_sliding_window(
                model=model,
                input_ids=input_ids,
                max_length=resolved_max_length,
                stride=resolved_stride,
            )
        else:
            nll_i, tok_i = _nll_sum_for_input_ids(model=model, input_ids=input_ids)

        nll_sum += nll_i
        total_pred_tokens += tok_i

        if report_every and report_every > 0 and (i % report_every == 0 or i == len(texts)):
            denom_chars = max(total_chars, 1)
            bpc_so_far = (nll_sum / denom_chars) / math.log(2.0)
            print(
                f"progress: {i}/{len(texts)} texts, tokens={total_pred_tokens}, chars={total_chars}, bpc={bpc_so_far:.6f}",
                file=sys.stderr,
            )

    if total_chars <= 0:
        raise ValueError("No characters found across scored texts")

    mean_nll_per_char_nats = nll_sum / total_chars
    bits_per_char = mean_nll_per_char_nats / math.log(2.0)

    meta = {
        "total_records": int(total_records),
        "scored_texts": int(len(texts)),
        "num_input_tokens": int(total_input_tokens),
        "num_words": int(total_words),
        "num_chars": int(total_chars),
    }

    return (
        BPCResult(
            bits_per_char=bits_per_char,
            mean_nll_per_char_nats=mean_nll_per_char_nats,
            num_predicted_tokens=total_pred_tokens,
        ),
        meta,
    )


def compute_bpc_txt_hf(
    *,
    model_name_or_path: str,
    txt_file: str,
    limit: int = 0,
    report_every: int = 50,
    device: str = "auto",
    device_map: Optional[str] = None,
    dtype: str = "auto",
    max_length: int = 0,
    stride: int = 0,
    trust_remote_code: bool = False,
) -> tuple[BPCResult, dict[str, int]]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies. Install at least: transformers, torch. "
            "(Optionally: accelerate, sentencepiece for some models.)"
        ) from e

    resolved_device = _pick_device(device)
    resolved_dtype = _pick_dtype(dtype, resolved_device)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        token=hf_token,
        use_fast=True,
    )

    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        token=hf_token,
        torch_dtype=resolved_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

    model.eval()

    if device_map is None:
        if resolved_device == "cuda":
            model.to("cuda")
        elif resolved_device == "mps":
            model.to("mps")
        else:
            model.to("cpu")

    resolved_max_length, resolved_stride = _resolve_window_params(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )

    texts, total_lines = _read_txt_texts(txt_file=txt_file, limit=limit)
    if not texts:
        raise ValueError("No non-empty lines found in txt file")

    nll_sum = 0.0
    total_pred_tokens = 0
    total_input_tokens = 0
    total_words = 0
    total_chars = 0

    for i, text in enumerate(texts, start=1):
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"]
        total_input_tokens += int(input_ids.numel())
        total_words += _count_words(text)
        total_chars += _count_chars(text)
        input_ids = input_ids.to(model.device)

        if int(input_ids.size(1)) > resolved_max_length:
            nll_i, tok_i = _nll_sum_for_input_ids_sliding_window(
                model=model,
                input_ids=input_ids,
                max_length=resolved_max_length,
                stride=resolved_stride,
            )
        else:
            nll_i, tok_i = _nll_sum_for_input_ids(model=model, input_ids=input_ids)

        nll_sum += nll_i
        total_pred_tokens += tok_i

        if report_every and report_every > 0 and (i % report_every == 0 or i == len(texts)):
            denom_chars = max(total_chars, 1)
            bpc_so_far = (nll_sum / denom_chars) / math.log(2.0)
            print(
                f"progress: {i}/{len(texts)} lines, tokens={total_pred_tokens}, chars={total_chars}, bpc={bpc_so_far:.6f}",
                file=sys.stderr,
            )

    if total_chars <= 0:
        raise ValueError("No characters found across scored lines")

    mean_nll_per_char_nats = nll_sum / total_chars
    bits_per_char = mean_nll_per_char_nats / math.log(2.0)

    meta = {
        "total_records": int(total_lines),
        "scored_texts": int(len(texts)),
        "num_input_tokens": int(total_input_tokens),
        "num_words": int(total_words),
        "num_chars": int(total_chars),
    }

    return (
        BPCResult(
            bits_per_char=bits_per_char,
            mean_nll_per_char_nats=mean_nll_per_char_nats,
            num_predicted_tokens=total_pred_tokens,
        ),
        meta,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        _, ext = os.path.splitext(args.input_file)
        ext = ext.lower()
        if ext == ".txt":
            res, meta = compute_bpc_txt_hf(
                model_name_or_path=args.model,
                txt_file=args.input_file,
                limit=args.limit,
                report_every=args.report_every,
                device=args.device,
                device_map=args.device_map,
                dtype=args.dtype,
                max_length=args.max_length,
                stride=args.stride,
                trust_remote_code=args.trust_remote_code,
            )
        else:
            res, meta = compute_bpc_json_hf(
                model_name_or_path=args.model,
                json_file=args.input_file,
                limit=args.limit,
                report_every=args.report_every,
                device=args.device,
                device_map=args.device_map,
                dtype=args.dtype,
                max_length=args.max_length,
                stride=args.stride,
                trust_remote_code=args.trust_remote_code,
            )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"model={args.model}")
    print(f"input_file={args.input_file}")
    if meta:
        print(f"total_records={meta.get('total_records', 0)}")
        print(f"scored_texts={meta.get('scored_texts', 0)}")
        print(f"num_input_tokens={meta.get('num_input_tokens', 0)}")
        print(f"num_predicted_tokens={res.num_predicted_tokens}")
        print(f"num_words={meta.get('num_words', 0)}")
        print(f"num_chars={meta.get('num_chars', 0)}")
    print(f"mean_nll_per_char_nats={res.mean_nll_per_char_nats:.8f}")
    print(f"bits_per_char={res.bits_per_char:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
