"""Dataset loader utilities."""

from .d4_synthetic import D4RetrievalDataset, D4SyntheticDataset, generate_d4_synthetic_dataset
from .d4_text import DEFAULT_BEIR_SUBSETS, load_d4_from_local_bundles
from .processed import (
    PROCESSED_SCHEMA_VERSION,
    dataset_dir_sha256,
    load_processed_d4_bundle,
    load_processed_text_dataset,
)

__all__ = [
    "D4RetrievalDataset",
    "D4SyntheticDataset",
    "DEFAULT_BEIR_SUBSETS",
    "PROCESSED_SCHEMA_VERSION",
    "dataset_dir_sha256",
    "generate_d4_synthetic_dataset",
    "load_d4_from_local_bundles",
    "load_processed_d4_bundle",
    "load_processed_text_dataset",
]
