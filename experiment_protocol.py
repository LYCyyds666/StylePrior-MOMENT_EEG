"""Shared helpers for subject-disjoint StylePrior-MOMENT experiments."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from torch.utils.data import Subset


def _as_int_list(values: Iterable) -> List[int]:
    result = []
    for value in values:
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
            result.extend(int(item) for item in value)
        else:
            result.append(int(value))
    return result


def subject_ids_from_dataset(dataset) -> List[int]:
    """Return sorted unique subject IDs without iterating a shuffled loader."""
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        if hasattr(base_dataset, "subject_ids"):
            all_subject_ids = np.asarray(base_dataset.subject_ids)
            selected = all_subject_ids[np.asarray(dataset.indices, dtype=np.int64)]
            return sorted({int(value) for value in selected.tolist()})

    if hasattr(dataset, "subject_ids"):
        return sorted({int(value) for value in _as_int_list(dataset.subject_ids)})

    subject_ids = []
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        if len(sample) < 3:
            raise ValueError(
                "The dataset does not expose subject_ids and samples do not "
                "contain a subject ID in position 2."
            )
        value = sample[2]
        if torch.is_tensor(value):
            value = int(value.detach().cpu().reshape(-1)[0].item())
        subject_ids.append(int(value))
    return sorted(set(subject_ids))


def subject_ids_from_loader(loader) -> List[int]:
    return subject_ids_from_dataset(loader.dataset)


def initialize_unseen_subject_embeddings(
    model,
    reference_subject_ids,
    target_subject_ids,
    *,
    verify: bool = True,
):
    """Initialize unseen rows with the current mean of source-training rows.

    The model uses one-based subject IDs. This function is intentionally called
    before every validation pass because the source-training embeddings change
    during each epoch.
    """
    reference_subject_ids = sorted({int(value) for value in reference_subject_ids})
    target_subject_ids = sorted({int(value) for value in target_subject_ids})
    unseen_subject_ids = sorted(
        set(target_subject_ids) - set(reference_subject_ids)
    )

    if not reference_subject_ids:
        raise ValueError("reference_subject_ids must not be empty")
    if min(reference_subject_ids + target_subject_ids) < 1:
        raise ValueError("Subject IDs must be one-based positive integers")
    if not unseen_subject_ids:
        return {
            "unseen_subject_ids": [],
            "initialized_tables": 0,
        }

    initialized_tables = 0
    reference_indices = torch.tensor(
        [subject_id - 1 for subject_id in reference_subject_ids],
        dtype=torch.long,
    )
    unseen_indices = torch.tensor(
        [subject_id - 1 for subject_id in unseen_subject_ids],
        dtype=torch.long,
    )

    for block in model.model.encoder.block:
        shared_knowledge = getattr(block, "shared_knowledge", None)
        subject_embedding = getattr(
            shared_knowledge,
            "subject_embedding",
            None,
        )
        if subject_embedding is None:
            continue

        number_of_rows = subject_embedding.num_embeddings
        if max(reference_subject_ids + unseen_subject_ids) > number_of_rows:
            raise IndexError(
                "Subject ID exceeds the subject embedding table: "
                f"maximum ID={max(reference_subject_ids + unseen_subject_ids)}, "
                f"rows={number_of_rows}"
            )

        reference_indices_device = reference_indices.to(
            subject_embedding.weight.device
        )
        unseen_indices_device = unseen_indices.to(subject_embedding.weight.device)

        with torch.no_grad():
            source_mean = subject_embedding.weight.index_select(
                0,
                reference_indices_device,
            ).mean(dim=0)
            subject_embedding.weight.index_copy_(
                0,
                unseen_indices_device,
                source_mean.unsqueeze(0).expand(len(unseen_subject_ids), -1),
            )

            if verify:
                initialized_rows = subject_embedding.weight.index_select(
                    0,
                    unseen_indices_device,
                )
                expected_rows = source_mean.unsqueeze(0).expand_as(initialized_rows)
                if not torch.allclose(
                    initialized_rows,
                    expected_rows,
                    rtol=0.0,
                    atol=1e-7,
                ):
                    raise RuntimeError(
                        "Failed to initialize unseen subject embeddings exactly"
                    )

        initialized_tables += 1

    if initialized_tables == 0:
        raise RuntimeError("No subject embedding tables were found in the model")

    return {
        "unseen_subject_ids": unseen_subject_ids,
        "initialized_tables": initialized_tables,
    }
