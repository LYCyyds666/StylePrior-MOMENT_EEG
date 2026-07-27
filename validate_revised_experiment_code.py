"""Fast CPU-only validation for the corrected experiment helpers."""

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from experiment_protocol import (
    initialize_unseen_subject_embeddings,
    subject_ids_from_loader,
)
from routing_analysis import RoutingAccumulator, merge_routing_analyses


class DummyDataset(Dataset):
    def __init__(self):
        self.subject_ids = [1, 1, 2, 2, 3, 3]

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, index):
        return (
            torch.zeros(2, 8),
            torch.tensor(index % 2),
            torch.tensor(self.subject_ids[index]),
        )


class DummySharedKnowledge(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.subject_embedding = torch.nn.Embedding(3, 4)
        self._last_router_probs = None
        self._last_indices = None


class DummyBlock(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.shared_knowledge = DummySharedKnowledge(layer_id)


class DummyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.ModuleList(
            [DummyBlock(0), DummyBlock(1)]
        )


class DummyInnerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DummyEncoder()


class DummyPipeline:
    def __init__(self):
        self.model = DummyInnerModel()


def test_subject_id_extraction():
    loader = DataLoader(
        Subset(DummyDataset(), [0, 1, 4, 5]),
        batch_size=2,
        shuffle=True,
    )
    assert subject_ids_from_loader(loader) == [1, 3]


def test_embedding_initialization():
    model = DummyPipeline()
    initial_weights = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
            [9.0, 9.0, 9.0, 9.0],
        ]
    )
    for block in model.model.encoder.block:
        with torch.no_grad():
            block.shared_knowledge.subject_embedding.weight.copy_(
                initial_weights
            )

    report = initialize_unseen_subject_embeddings(
        model,
        [1, 2],
        [3],
    )
    assert report == {
        "unseen_subject_ids": [3],
        "initialized_tables": 2,
    }
    expected_mean = torch.tensor([2.0, 3.0, 4.0, 5.0])
    for block in model.model.encoder.block:
        actual = block.shared_knowledge.subject_embedding.weight[2]
        assert torch.allclose(actual, expected_mean)


def test_routing_accumulator():
    model = DummyPipeline()
    for block in model.model.encoder.block:
        block.shared_knowledge._last_router_probs = torch.tensor(
            [
                [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]],
                [[0.2, 0.2, 0.6], [0.8, 0.1, 0.1]],
            ]
        )
        block.shared_knowledge._last_indices = torch.tensor(
            [
                [[0, 1], [1, 2]],
                [[2, 0], [0, 1]],
            ]
        )

    accumulator = RoutingAccumulator()
    accumulator.update(
        model,
        subject_ids=torch.tensor([1, 3]),
        labels=torch.tensor([0, 1]),
    )
    summary = accumulator.finalize()

    assert set(summary["layers"]) == {"0", "1"}
    for layer in summary["layers"].values():
        assert layer["token_count"] == 4
        assert abs(sum(layer["topk_assignment_fraction"]) - 1.0) < 1e-12
        assert set(layer["subject_topk_fraction"]) == {"1", "3"}
        assert set(layer["class_topk_fraction"]) == {"0", "1"}

    merged = merge_routing_analyses([summary, summary])
    assert merged["layers"]["0"]["token_count"] == 8


if __name__ == "__main__":
    test_subject_id_extraction()
    test_embedding_initialization()
    test_routing_accumulator()
    print("PASS: corrected embedding and routing helper tests")
