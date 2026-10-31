"""Skip hashing the tokenizer closure for Soup 0.77.0's in-memory SFT row map."""

import os
from functools import wraps


SFT_ROWS_FINGERPRINT = "soup-c6-sft-format-rows-v1"


def _is_soup_sft_formatter(function):
    return (getattr(function, "__module__", None) == "soup_cli.trainer.sft"
            and getattr(function, "__qualname__", None)
            == "_map_text_sft_rows.<locals>.checked_format_row")


def patch_sft_dataset_map():
    """Give only Soup's in-memory text SFT formatting map a fixed cache fingerprint."""
    if os.environ.get("SOUP_X_FPRINT") != "1":
        return

    try:
        from datasets import Dataset
    except Exception:
        return

    if getattr(Dataset.map, "_soup_x_fprint_patched", False):
        return

    original = Dataset.map

    @wraps(original)
    def map_with_sft_fingerprint(self, function=None, *args, **kwargs):
        if _is_soup_sft_formatter(function):
            kwargs.setdefault("new_fingerprint", SFT_ROWS_FINGERPRINT)
            kwargs.setdefault("load_from_cache_file", False)
        return original(self, function, *args, **kwargs)

    map_with_sft_fingerprint._soup_x_fprint_patched = True
    map_with_sft_fingerprint._soup_x_fprint_original = original
    Dataset.map = map_with_sft_fingerprint
