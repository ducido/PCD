"""Checks that sdn.selection reproduces the implementation used for the paper results.

The `_reference_*` functions are verbatim copies of the original experiment code.
Run with: python -m pytest tests/
"""

import pytest
import torch

from sdn.selection import (
    knn_grounding_scores,
    sdn_select,
    select_by_grounding,
    select_by_smoothness,
    smoothness_scores,
)


def _reference_knn_scores(actions, contrast_actions, knn_k, eps=1e-8):
    N = actions.shape[0]
    C = contrast_actions.shape[0]
    A = actions.reshape(N, -1).float()
    B = contrast_actions.reshape(C, -1).float()
    dist_AB = torch.cdist(A, B, p=2) ** 2
    knn_dist_AB, _ = torch.topk(dist_AB, k=knn_k, largest=False, dim=1)
    R_B = knn_dist_AB.sum(dim=1)
    dist_AA = torch.cdist(A, A, p=2) ** 2
    inf_mask = torch.eye(N, device=A.device) * 1e9
    dist_AA = dist_AA + inf_mask
    knn_dist_AA, _ = torch.topk(dist_AA, k=knn_k, largest=False, dim=1)
    R_A = knn_dist_AA.sum(dim=1)
    return torch.log(R_B + eps) - torch.log(R_A + eps)


def _reference_smoothest_delta_action(actions, lambda_energy=0.05, ignore_last_dim=True):
    act = actions[..., :-1] if ignore_last_dim else actions
    std = act.std(dim=(0, 1), keepdim=True)
    act = act / (std + 1e-6)
    acc = act[:, 2:] - 2 * act[:, 1:-1] + act[:, :-2]
    acc_sq = (acc ** 2).sum(dim=-1)
    smoothness = acc_sq.mean(dim=1)
    energy = (act ** 2).sum(dim=-1).mean(dim=1)
    scores = smoothness + lambda_energy * energy
    return scores, int(torch.argmin(scores))


def _reference_grounding_and_smooth(actions, contrast_actions, knn_k, top_k, exec_horizon):
    scores = _reference_knn_scores(actions[:, :exec_horizon], contrast_actions[:, :exec_horizon], knn_k)
    _, top_indices = torch.topk(scores, k=top_k, largest=True)
    candidates = actions[top_indices]
    _, best_idx = _reference_smoothest_delta_action(candidates)
    return candidates[best_idx:best_idx + 1][:, :exec_horizon]


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("knn_k", [3, 6, 10])
def test_grounding_matches_reference(seed, knn_k):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(12, 4, 7, generator=g)
    neg = torch.randn(12, 4, 7, generator=g) + 0.5
    torch.testing.assert_close(knn_grounding_scores(pos, neg, knn_k), _reference_knn_scores(pos, neg, knn_k))
    assert select_by_grounding(pos, neg, knn_k) == int(torch.argmax(_reference_knn_scores(pos, neg, knn_k)))


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("horizon", [4, 5, 10])
def test_smoothness_matches_reference(seed, horizon):
    g = torch.Generator().manual_seed(seed)
    actions = torch.randn(12, horizon, 7, generator=g)
    ref_scores, ref_idx = _reference_smoothest_delta_action(actions)
    torch.testing.assert_close(smoothness_scores(actions), ref_scores)
    assert select_by_smoothness(actions) == ref_idx


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("knn_k,top_m,long_horizon", [(6, 3, 4), (10, 5, 5), (6, 5, 10), (3, 3, 10)])
def test_sdn_select_matches_reference(seed, knn_k, top_m, long_horizon):
    exec_horizon = 4
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(12, long_horizon, 7, generator=g)
    neg = torch.randn(12, long_horizon, 7, generator=g) + 0.3
    idx = sdn_select(pos, neg, knn_k, top_m, grounding_horizon=exec_horizon)
    expected = _reference_grounding_and_smooth(pos, neg, knn_k, top_m, exec_horizon)
    torch.testing.assert_close(pos[idx:idx + 1, :exec_horizon], expected)


def test_invalid_hyperparameters_raise():
    pos, neg = torch.randn(12, 4, 7), torch.randn(12, 4, 7)
    with pytest.raises(ValueError):
        knn_grounding_scores(pos, neg, knn_k=12)
    with pytest.raises(ValueError):
        sdn_select(pos, neg, knn_k=6, top_m=13)
    with pytest.raises(ValueError):
        smoothness_scores(torch.randn(12, 2, 7))
