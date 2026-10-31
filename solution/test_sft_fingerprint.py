"""Soup's in-memory SFT row map keeps the same values with a fixed fingerprint."""

import pytest


def test_fixed_fingerprint_preserves_sft_rows(monkeypatch):
    datasets = pytest.importorskip("datasets")
    from solution.sft_fingerprint import SFT_ROWS_FINGERPRINT, patch_sft_dataset_map

    class ToyTokenizer:
        def encode(self, text):
            return [ord(char) for char in text]

    tokenizer = ToyTokenizer()

    def make_formatter():
        def checked_format_row(row, index):
            return {"input_ids": tokenizer.encode(row["text"]), "row_number": index + 1}

        checked_format_row.__module__ = "soup_cli.trainer.sft"
        checked_format_row.__qualname__ = "_map_text_sft_rows.<locals>.checked_format_row"
        return checked_format_row

    rows = {"text": ["alpha", "Soup", "T4"]}
    expected = datasets.Dataset.from_dict(rows).map(
        make_formatter(), with_indices=True, remove_columns=["text"])

    monkeypatch.setenv("SOUP_X_FPRINT", "1")
    original_map = datasets.Dataset.map
    patch_sft_dataset_map()
    try:
        actual = datasets.Dataset.from_dict(rows).map(
            make_formatter(), with_indices=True, remove_columns=["text"])
    finally:
        datasets.Dataset.map = original_map

    assert list(actual) == list(expected)
    assert actual._fingerprint == SFT_ROWS_FINGERPRINT
