import json

import numpy as np
import pytest
import torch

from benchmark import data
from benchmark.metrics import aggregate, bootstrap, per_image
from benchmark.model import LaneDGSemantic, segmentation_loss
from benchmark.prepare import select_disjoint


def test_tusimple_missing_points_do_not_bridge(monkeypatch):
    monkeypatch.setattr(data, "source_annotation", lambda _: {"lanes": [[10, 10, -2, 10, 10]], "h_samples": [0, 10, 20, 30, 40]})
    mask, valid = data.rasterize({"dataset": "tusimple"}, 100, 100, size=100, line_width=1)
    assert mask[5, 10] == 1 and mask[35, 10] == 1
    assert mask[20, 10] == 0
    assert valid.all()


def test_exact_non_square_coordinate_transform(monkeypatch):
    monkeypatch.setattr(data, "paths", lambda _: [(np.array([[100, 0], [100, 100]]), False)])
    mask, _ = data.rasterize({}, width=200, height=100, size=100, line_width=1)
    assert mask[50, 50] == 1
    assert mask[:, 49].sum() == 0


def test_cubic_bezier_uses_controls_not_polyline():
    curve = data.bezier_path([[0, 0], [0, 10], [10, 10], [10, 0]], "LCCC")
    assert curve.shape == (65, 2)
    assert np.allclose(curve[32], [5, 7.5])
    assert np.allclose(curve[-1], [10, 0])
    assert np.allclose(curve, data.bezier_path([[0, 0], [0, 10], [10, 10], [10, 0]], "CCCL"))
    with pytest.raises(ValueError):
        data.bezier_path([[0, 0], [1, 2]], "LC")


def test_bdd_category_and_ignore_mapping(monkeypatch):
    objects = [
        {"category": "lane/single white", "attributes": {"direction": "parallel"}, "poly2d": [[10, 0, "L"], [10, 90, "L"]]},
        {"category": "lane/road curb", "attributes": {}, "poly2d": [[20, 0, "L"], [20, 90, "L"]]},
        {"category": "lane/single yellow", "attributes": {"direction": "vertical"}, "poly2d": [[30, 0, "L"], [30, 90, "L"]]},
        {"category": "lane/single white", "attributes": {"occluded": True}, "poly2d": [[40, 0, "L"], [40, 90, "L"]]},
    ]
    monkeypatch.setattr(data, "source_annotation", lambda _: {"frames": [{"objects": objects}]})
    mask, valid = data.rasterize({"dataset": "bdd100k"}, 100, 100, 100, 1)
    assert mask[40, 10] == 1 and valid[40, 10] == 1
    assert mask[40, 20:41].sum() == 0
    assert valid[40, 20] == valid[40, 30] == valid[40, 40] == 0


def test_pixel_metrics_ignore_pixels_and_handle_empty_union():
    record = per_image(np.array([.9, .9, .1, .9]), np.array([1, 0, 1, 0]), np.array([1, 1, 1, 0]), .5)
    assert (record["tp"], record["fp"], record["fn"]) == (1, 1, 1)
    assert record["iou"] == pytest.approx(1/3)
    result = aggregate([record])
    assert result["pixel_f1"] == .5
    assert result["mask_iou_success_50"] == 0
    assert per_image(np.zeros(2), np.zeros(2), np.ones(2), .5)["iou"] == 1


def test_pixel_micro_metrics_not_mean_image_f1():
    rows = [{"tp": 1, "fp": 0, "fn": 0, "iou": 1}, {"tp": 0, "fp": 99, "fn": 0, "iou": 0}]
    assert aggregate(rows)["pixel_precision"] == .01
    assert aggregate(rows)["pixel_f1"] == pytest.approx(2/101)


def test_bootstrap_resamples_sequence_clusters_reproducibly():
    rows = [{"tp": 10, "fp": 0, "fn": 0, "iou": 1., "sequence_id": "a"},
            {"tp": 0, "fp": 10, "fn": 0, "iou": 0., "sequence_id": "b"}]
    result = bootstrap(rows, repetitions=200)
    assert result == bootstrap(rows, repetitions=200)
    assert result["sequence_groups"] == 2
    assert result["intervals_95"]["mean_image_iou"] == [0., 1.]


def test_selection_preserves_original_splits_and_disjoint_sequences():
    training = [{"sequence_id": str(i), "image_id": f"{i}.jpg", "original_partition": "train"} for i in range(10)]
    validation = [{"sequence_id": str(i), "image_id": f"{i}b.jpg", "original_partition": "val"} for i in range(3, 8)]
    testing = [{"sequence_id": str(i), "image_id": f"{i}.jpg", "original_partition": "test"} for i in range(10, 15)]
    pools = {"train": training, "validation": validation, "evaluation": testing}
    rows = select_disjoint(pools, 42, {"train": 4, "validation": 2, "evaluation": 2})
    assert len({r["sequence_id"] for r in rows}) == len(rows)
    assert all(r["original_partition"] == "val" for r in rows if r["role"] == "validation")
    assert rows == select_disjoint(pools, 42, {"train": 4, "validation": 2, "evaluation": 2})


def test_model_has_no_padding_parameters_and_all_paths_train():
    torch.manual_seed(42)
    model = LaneDGSemantic()
    image = torch.randn(2, 3, 64, 64)
    truth = torch.zeros(2, 1, 64, 64)
    truth[:, :, :, 30:33] = 1
    valid = torch.ones_like(truth)
    logits = model(image)
    assert logits.shape == truth.shape
    segmentation_loss(logits, truth, valid).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not any("calibration" in n for n, _ in model.named_parameters())


def test_ignored_pixels_do_not_affect_loss():
    logits = torch.zeros(1, 1, 4, 4)
    truth = torch.zeros_like(logits)
    truth[:, :, 1:3, 1:3] = 1
    valid = torch.ones_like(logits)
    valid[:, :, 0] = 0
    expected = segmentation_loss(logits, truth, valid)
    logits[:, :, 0] = 100
    assert segmentation_loss(logits, truth, valid) == expected
