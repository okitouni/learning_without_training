# %%
from scipy.optimize import linear_sum_assignment
import torch
from lwot.models import GEMLinear, GEMBase
from copy import deepcopy
import numpy as np


def print_(*args):
    for arg in args:
        print(arg)


def weight_matching(m1, m2, inplace=True):
    # TODO so far this only works for nn.Sequential's
    permutations = [... for l in m1.children() if isinstance(l, torch.nn.Linear)]
    module_idx = [idx for idx in range(len(m1)) if isinstance(m1[idx], torch.nn.Linear)]
    m3 = m2 if inplace else deepcopy(m2)
    converged = False
    while not converged:
        converged = True
        for i, l in enumerate(module_idx):
            w1 = m1[l].weight.data.clone()
            w2 = m2[l].weight.data.clone()

            prev_permutation = ... if i == 0 else permutations[i - 1]
            total = -w1 @ w2.T[prev_permutation]

            # this is the contribution W1_{l+1}.T @ P_{l+1} @ W2_{l+1}
            if i < len(module_idx) - 1:  # no contribution for last layer
                l_next = module_idx[i + 1]
                next_permutation = permutations[i + 1]
                w1_next = m1[l_next].weight.data.clone()
                w2_next = m2[l_next].weight.data.clone()
                w2_next = w2_next[next_permutation]
                total -= w1_next.T @ w2_next

            _, col_ind = linear_sum_assignment(total.detach().cpu().numpy())
            if (col_ind != permutations[i]).any():
                converged = False
            permutations[i] = col_ind

    for i, l in enumerate(module_idx):
        m3[l].weight.data = m2[l].weight.data[permutations[i]]
        if m3[l].bias is not None:
            m3[l].bias.data = m2[l].bias[permutations[i]]
            
    for i, l in enumerate(module_idx):
        if i == 0: continue
        m3[l].weight.data = m2[l].weight.data.T[permutations[i-1]].T

    return m3


def weight_matching_with_scores(m1, m2, inplace=True):
    m3 = m1 if inplace else deepcopy(m1)
    prev_permutation = None
    for l1, l2, l3 in zip(m1.children(), m2.children(), m3.children()):
        if not isinstance(l1, GEMBase):
            continue
        if prev_permutation is not None:
            l3.weight_scores.data = l3.weight_scores.data.T[
                prev_permutation
            ].T  # permute score columns
            l3.weight.data = l3.weight.data.T[
                prev_permutation
            ].T  # permute actual weight columns
        w1 = l3.masked_weight.data.detach().cpu().numpy()
        w2 = l2.masked_weight.data.detach().cpu().numpy()
        _, col_ind = linear_sum_assignment(-w1 @ w2.T)
        l3.weight.data = l3.weight.data[col_ind]  # permute weight rows
        l3.weight_scores.data = l3.weight_scores.data[col_ind]  # permute score rows
        if l3.bias is not None:
            l3.bias.data = l3.bias.data[col_ind]
            l3.bias_scores.data = l3.bias_scores.data[col_ind]
        prev_permutation = col_ind
    return m3


def test_weight_matching():
    print("test weight matching")
    w1_A = torch.tensor([[1, 2], [3, 4]]).float()
    w1_B = torch.tensor([[3, 4], [1, 2]]).float()

    w2_A = torch.tensor([[5, 6], [7, 8]]).float()
    w2_B = torch.tensor([[6, 5], [8, 7]]).float()
    # w2_B = torch.tensor([[8, 7], [6, 5]]).float()

    w3_A = torch.tensor([[9, 10]]).float()
    w3_B = torch.tensor([[9, 10]]).float()
    # w3_B = torch.tensor([[10, 9]]).float()

    bias = True
    m1 = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=bias), torch.nn.ReLU(), torch.nn.Linear(2, 2, bias=bias), torch.nn.Linear(2, 1)
    )
    m2 = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=bias), torch.nn.ReLU(),torch.nn.Linear(2, 2, bias=bias), torch.nn.Linear(2, 1)
    )
    m1[0].weight.data = w1_A
    m1[2].weight.data = w2_A
    m1[3].weight.data = w3_A
    m2[0].weight.data = w1_B
    m2[2].weight.data = w2_B
    m2[3].weight.data = w3_B

    x = torch.rand(1, 2)
    def _print_all(m1, m2):
        print_("A")
        print_(*[l.weight.data.tolist() for l in m1 if isinstance(l, torch.nn.Linear)])
        print_(m1(x))
        print_("B")
        print_(*[l.weight.data.tolist() for l in m2 if isinstance(l, torch.nn.Linear)])
        print_(m2(x))

    print_("pre-matching")
    _print_all(m1, m2)
    m3 = weight_matching(m1, m2, inplace=False)
    print_("post-matching")
    _print_all(m1, m3)
    print(m3 == m2)


def test_weight_matching_with_masks():
    print("test weight matching with scores")
    w1_A = torch.tensor([[1, 2], [3, 4]]).float()
    m1_A = torch.tensor([[1, 0], [0, 1]]).float()
    w1_B = w1_A.flip(0).clone()
    m1_B = m1_A.flip(0).clone()

    w2_A = torch.tensor([[5, 6], [7, 8]]).float()
    m2_A = torch.tensor([[1, 0], [0, 1]]).float()
    w2_B = w2_A.flip(0, 1).clone()
    m2_B = m2_A.flip(0, 1).clone()

    w3_A = torch.tensor([[9, 10]]).float()
    m3_A = torch.tensor([[1, 0]]).float()
    w3_B = w3_A.flip(1).clone()
    m3_B = m3_A.flip(1).clone()

    m1 = torch.nn.Sequential(
        GEMLinear(2, 2, bias=False),
        GEMLinear(2, 2, bias=False),
        GEMLinear(2, 1, bias=False),
    )
    m2 = torch.nn.Sequential(
        GEMLinear(2, 2, bias=False),
        GEMLinear(2, 2, bias=False),
        GEMLinear(2, 1, bias=False),
    )

    m1[0].weight.data, m1[0].weight_scores.data = w1_A, m1_A
    m1[1].weight.data, m1[1].weight_scores.data = w2_A, m2_A
    m1[2].weight.data, m1[2].weight_scores.data = w3_A, m3_A
    m2[0].weight.data, m2[0].weight_scores.data = w1_B, m1_B
    m2[1].weight.data, m2[1].weight_scores.data = w2_B, m2_B
    m2[2].weight.data, m2[2].weight_scores.data = w3_B, m3_B

    def _print_all():
        print_(
            m1[0].masked_weight.data, m1[1].masked_weight.data, m1[2].masked_weight.data
        )
        print_(
            "prod",
            m1[0].masked_weight.data.T
            @ m1[1].masked_weight.data.T
            @ m1[2].masked_weight.data.T,
        )
        print_("B")
        print_(
            m2[0].masked_weight.data, m2[1].masked_weight.data, m2[2].masked_weight.data
        )
        print_(
            "prod",
            m2[0].masked_weight.data.T
            @ m2[1].masked_weight.data.T
            @ m2[2].masked_weight.data.T,
        )

    print_("pre-matching")
    _print_all()
    weight_matching_with_scores(m2, m1, inplace=True)
    print_("post-matching")
    _print_all()


def activation_matching(a1, a2):
    # permute a2 to match a1 as close as possible in terms of matrix frobenius norm
    # a1 and a2 are of shape (n_training_points, features)
    a1 = a1.detach().cpu().numpy().T
    a2 = a2.detach().cpu().numpy().T
    _, col_ind = linear_sum_assignment(-a1 @ a2.T)
    return col_ind


def activation_match_model_2_to_1(data, model1, model2):
    # side effects on model2 (changing the weights to match model1)
    # W'_l = P_l @ W_l @ P_{l-1}.T
    # data.shape = (n_training_points, features)
    # model1 and model2 are of type torch.nn.Sequential
    prev_permutation = None
    data1_i = data
    data2_i = data
    model3 = deepcopy(model2)
    for i, l3 in enumerate(model3):
        data1_i = model1[i](data1_i)
        data2_i = model2[i](data2_i)
        if isinstance(l3, GEMBase):
            permutation = activation_matching(data1_i, data2_i)
            if prev_permutation is not None:
                l3.weight.data = l3.weight.data.T[prev_permutation].T
            l3.weight.data = l3.weight.data[permutation]
            if l3.bias is not None:
                l3.bias.data = l3.bias.data[permutation]
            prev_permutation = permutation

    return model3


def test_activation_matching():
    torch.manual_seed(10)
    print("test activation matching with scores")
    w1_A = torch.tensor([[1, 2], [3, 4]]).float()
    w1_B = w1_A.flip(0).clone()

    w2_A = torch.tensor([[5, 6], [7, 8]]).float()
    w2_B = w2_A.flip(0, 1).clone()

    w3_A = torch.tensor([[9, 10]]).float()
    w3_B = w3_A.flip(1).clone()

    m1 = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(2, 1, bias=False),
    )
    m2 = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(2, 1, bias=False),
    )

    m1[0].weight.data = w1_A
    m1[2].weight.data = w2_A
    m1[4].weight.data = w3_A
    m2[0].weight.data = w1_B
    m2[2].weight.data = w2_B
    m2[4].weight.data = w3_B

    # generate some data
    data = torch.randn(100, 2)

    def _print_all():
        with torch.no_grad():
            print_(m1[0].weight.data, m1[2].weight.data, m1[4].weight.data)
            print_(
                "result", m1(data)[:10, 0],
            )
            print_("B")
            print_(m2[0].weight.data, m2[2].weight.data, m2[4].weight.data)
            print_(
                "prod", m2(data)[:10, 0],
            )

    print_("pre-matching")
    _print_all()
    activation_match_model_2_to_1(data, m1, m2)
    print_("post-matching")
    _print_all()


if __name__ == "__main__":
    test_weight_matching()
    # test_weight_matching_with_masks()
    # test_activation_matching()


# %%

