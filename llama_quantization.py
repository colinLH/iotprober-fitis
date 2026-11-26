"""Shared bitsandbytes settings for the local Llama base (not its LoRA weights)."""

from transformers import BitsAndBytesConfig


def resolve_load_bits(gpu, load_in_4bit=None, load_in_8bit=False):
    if load_in_4bit and load_in_8bit:
        raise ValueError("4-bit and 8-bit loading are mutually exclusive")
    if load_in_8bit:
        return 8
    if load_in_4bit or (load_in_4bit is None and gpu >= 0):
        return 4
    return None


def bnb_config(bits, compute_dtype):
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    raise ValueError(f"Expected 4 or 8 bits, got {bits!r}")
