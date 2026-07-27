"""Streaming aggregation of SA-MoE routing statistics."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch


def _new_layer_counts(num_experts):
    return {
        "num_experts": int(num_experts),
        "token_count": 0,
        "assignment_count": 0,
        "entropy_sum": 0.0,
        "probability_sum": np.zeros(num_experts, dtype=np.float64),
        "top1_count": np.zeros(num_experts, dtype=np.int64),
        "topk_count": np.zeros(num_experts, dtype=np.int64),
        "subject_topk_count": defaultdict(
            lambda: np.zeros(num_experts, dtype=np.int64)
        ),
        "class_topk_count": defaultdict(
            lambda: np.zeros(num_experts, dtype=np.int64)
        ),
    }


class RoutingAccumulator:
    """Accumulate small sufficient statistics without storing token tensors."""

    def __init__(self):
        self.layers = {}

    @staticmethod
    def _group_counts(indices, token_groups, num_experts):
        result = {}
        top_k = indices.shape[1]
        for group_value in torch.unique(token_groups).tolist():
            mask = token_groups == int(group_value)
            selected = indices[mask].reshape(-1)
            counts = torch.bincount(
                selected,
                minlength=num_experts,
            )
            result[int(group_value)] = counts.detach().cpu().numpy().astype(
                np.int64,
                copy=False,
            )
            expected = int(mask.sum().item()) * top_k
            if int(counts.sum().item()) != expected:
                raise RuntimeError("Routing assignment accounting mismatch")
        return result

    def update(self, model, subject_ids=None, labels=None):
        for block_index, block in enumerate(model.model.encoder.block):
            module = getattr(block, "shared_knowledge", None)
            router_probs = getattr(module, "_last_router_probs", None)
            top_k_indices = getattr(module, "_last_indices", None)
            if router_probs is None or top_k_indices is None:
                continue

            probabilities = router_probs.detach().float().reshape(
                -1,
                router_probs.shape[-1],
            )
            indices = top_k_indices.detach().long().reshape(
                -1,
                top_k_indices.shape[-1],
            )
            num_tokens, num_experts = probabilities.shape
            top_k = indices.shape[1]

            layer_id = int(getattr(module, "layer_id", block_index))
            if layer_id not in self.layers:
                self.layers[layer_id] = _new_layer_counts(num_experts)
            counts = self.layers[layer_id]

            counts["token_count"] += int(num_tokens)
            counts["assignment_count"] += int(num_tokens * top_k)
            counts["entropy_sum"] += float(
                (
                    -probabilities
                    * torch.log(probabilities.clamp_min(1e-12))
                ).sum(dim=1).sum().item()
            )
            counts["probability_sum"] += (
                probabilities.sum(dim=0).cpu().numpy()
            )
            counts["top1_count"] += (
                torch.bincount(indices[:, 0], minlength=num_experts)
                .cpu()
                .numpy()
            )
            counts["topk_count"] += (
                torch.bincount(indices.reshape(-1), minlength=num_experts)
                .cpu()
                .numpy()
            )

            if subject_ids is not None:
                batch_subject_ids = subject_ids.detach().long().reshape(-1)
                if num_tokens % len(batch_subject_ids) != 0:
                    raise RuntimeError(
                        "Router token count is not divisible by batch size"
                    )
                tokens_per_sample = num_tokens // len(batch_subject_ids)
                token_subject_ids = batch_subject_ids.repeat_interleave(
                    tokens_per_sample
                ).to(indices.device)
                grouped = self._group_counts(
                    indices,
                    token_subject_ids,
                    num_experts,
                )
                for subject_id, values in grouped.items():
                    counts["subject_topk_count"][subject_id] += values

            if labels is not None:
                batch_labels = labels.detach().long().reshape(-1)
                if num_tokens % len(batch_labels) != 0:
                    raise RuntimeError(
                        "Router token count is not divisible by label batch size"
                    )
                tokens_per_sample = num_tokens // len(batch_labels)
                token_labels = batch_labels.repeat_interleave(
                    tokens_per_sample
                ).to(indices.device)
                grouped = self._group_counts(
                    indices,
                    token_labels,
                    num_experts,
                )
                for class_id, values in grouped.items():
                    counts["class_topk_count"][class_id] += values

    def finalize(self):
        raw_layers = {}
        for layer_id, counts in sorted(self.layers.items()):
            raw_layers[str(layer_id)] = {
                "num_experts": counts["num_experts"],
                "token_count": counts["token_count"],
                "assignment_count": counts["assignment_count"],
                "entropy_sum": counts["entropy_sum"],
                "probability_sum": counts["probability_sum"].tolist(),
                "top1_count": counts["top1_count"].tolist(),
                "topk_count": counts["topk_count"].tolist(),
                "subject_topk_count": {
                    str(key): value.tolist()
                    for key, value in sorted(
                        counts["subject_topk_count"].items()
                    )
                },
                "class_topk_count": {
                    str(key): value.tolist()
                    for key, value in sorted(counts["class_topk_count"].items())
                },
            }
        return finalize_routing_counts(raw_layers)


def _safe_fraction(values, denominator):
    values = np.asarray(values, dtype=np.float64)
    if denominator <= 0:
        return np.zeros_like(values).tolist()
    return (values / float(denominator)).tolist()


def _normalize_group_counts(group_counts):
    normalized = {}
    for key, values in sorted(group_counts.items(), key=lambda item: int(item[0])):
        values_array = np.asarray(values, dtype=np.float64)
        normalized[str(key)] = _safe_fraction(values_array, values_array.sum())
    return normalized


def finalize_routing_counts(raw_layers):
    layers = {}
    for layer_id, counts in sorted(raw_layers.items(), key=lambda item: int(item[0])):
        num_experts = int(counts["num_experts"])
        token_count = int(counts["token_count"])
        assignment_count = int(counts["assignment_count"])
        entropy = (
            float(counts["entropy_sum"]) / token_count
            if token_count
            else 0.0
        )
        layers[str(layer_id)] = {
            "num_experts": num_experts,
            "token_count": token_count,
            "assignment_count": assignment_count,
            "mean_router_probability": _safe_fraction(
                counts["probability_sum"],
                token_count,
            ),
            "top1_fraction": _safe_fraction(
                counts["top1_count"],
                token_count,
            ),
            "topk_assignment_fraction": _safe_fraction(
                counts["topk_count"],
                assignment_count,
            ),
            "router_entropy": entropy,
            "normalized_router_entropy": (
                entropy / math.log(num_experts)
                if num_experts > 1
                else 0.0
            ),
            "subject_topk_fraction": _normalize_group_counts(
                counts["subject_topk_count"]
            ),
            "class_topk_fraction": _normalize_group_counts(
                counts["class_topk_count"]
            ),
        }

    return {
        "definition": {
            "mean_router_probability": (
                "Mean pre-top-k softmax probability over tokens"
            ),
            "top1_fraction": "Fraction of tokens routed to each top-1 expert",
            "topk_assignment_fraction": (
                "Fraction of all top-k assignments received by each expert"
            ),
            "router_entropy": (
                "Mean token-wise entropy of pre-top-k router probabilities"
            ),
            "normalized_router_entropy": (
                "Router entropy divided by log(number of experts)"
            ),
        },
        "layers": layers,
        "_counts": raw_layers,
    }


def merge_routing_analyses(analyses):
    merged = {}
    for analysis in analyses:
        if not analysis:
            continue
        for layer_id, counts in analysis.get("_counts", {}).items():
            if layer_id not in merged:
                merged[layer_id] = {
                    "num_experts": int(counts["num_experts"]),
                    "token_count": 0,
                    "assignment_count": 0,
                    "entropy_sum": 0.0,
                    "probability_sum": np.zeros(
                        counts["num_experts"],
                        dtype=np.float64,
                    ),
                    "top1_count": np.zeros(
                        counts["num_experts"],
                        dtype=np.int64,
                    ),
                    "topk_count": np.zeros(
                        counts["num_experts"],
                        dtype=np.int64,
                    ),
                    "subject_topk_count": defaultdict(
                        lambda: np.zeros(
                            counts["num_experts"],
                            dtype=np.int64,
                        )
                    ),
                    "class_topk_count": defaultdict(
                        lambda: np.zeros(
                            counts["num_experts"],
                            dtype=np.int64,
                        )
                    ),
                }
            target = merged[layer_id]
            target["token_count"] += int(counts["token_count"])
            target["assignment_count"] += int(counts["assignment_count"])
            target["entropy_sum"] += float(counts["entropy_sum"])
            target["probability_sum"] += np.asarray(
                counts["probability_sum"],
                dtype=np.float64,
            )
            target["top1_count"] += np.asarray(
                counts["top1_count"],
                dtype=np.int64,
            )
            target["topk_count"] += np.asarray(
                counts["topk_count"],
                dtype=np.int64,
            )
            for key, values in counts["subject_topk_count"].items():
                target["subject_topk_count"][str(key)] += np.asarray(
                    values,
                    dtype=np.int64,
                )
            for key, values in counts["class_topk_count"].items():
                target["class_topk_count"][str(key)] += np.asarray(
                    values,
                    dtype=np.int64,
                )

    serializable = {}
    for layer_id, counts in merged.items():
        serializable[layer_id] = {
            "num_experts": counts["num_experts"],
            "token_count": counts["token_count"],
            "assignment_count": counts["assignment_count"],
            "entropy_sum": counts["entropy_sum"],
            "probability_sum": counts["probability_sum"].tolist(),
            "top1_count": counts["top1_count"].tolist(),
            "topk_count": counts["topk_count"].tolist(),
            "subject_topk_count": {
                str(key): value.tolist()
                for key, value in counts["subject_topk_count"].items()
            },
            "class_topk_count": {
                str(key): value.tolist()
                for key, value in counts["class_topk_count"].items()
            },
        }
    return finalize_routing_counts(serializable) if serializable else None
