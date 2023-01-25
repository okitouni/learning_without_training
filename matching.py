# %%

from scipy.optimize import linear_sum_assignment
import torch
from lwot.models import GEMLinear
from copy import deepcopy


def print_(*args):
    for arg in args:
        print(arg)


def dot_product_matching(m1, m2, inplace=True):
    m3 = m1 if inplace else deepcopy(m1)
    prev_permutation = None
    for l1, l2, l3 in zip(m1.children(), m2.children(), m3.children()):
        if not isinstance(l1, torch.nn.Linear):
            continue
        w1 = l1.weight.data.clone()
        w2 = l2.weight.data
        if prev_permutation is not None:
            w1 = w1.T[prev_permutation].T
        w1_ = w1.detach().cpu().numpy()
        w2_ = w2.detach().cpu().numpy()
        _, col_ind = linear_sum_assignment(-w1_ @ w2_.T)
        prev_permutation = col_ind
        l3.weight.data = w1[col_ind]
        if l3.bias is not None:
          l3.bias.data = l1.bias[col_ind]
    return m3


def dot_product_matching_with_scores(m1, m2, inplace=True):
    m3 = m1 if inplace else deepcopy(m1)
    prev_permutation = None
    for l1, l2, l3 in zip(m1.children(), m2.children(), m3.children()):
        if not isinstance(l1, GEMLinear):
            continue
        w1 = l1.masked_weight.data.clone()
        w2 = l2.masked_weight.data.clone()
        if prev_permutation is not None:
            w1 = w1.T[prev_permutation].T
            l3.weight_scores.data = l3.weight_scores.data.T[prev_permutation].T
            l3.weight.data = l3.weight.data.T[prev_permutation].T
        w1 = w1.detach().cpu().numpy()
        w2 = w2.detach().cpu().numpy()
        _, col_ind = linear_sum_assignment(-w1 @ w2.T)
        l3.weight.data = l3.weight.data[col_ind]
        l3.weight_scores.data = l3.weight_scores.data[col_ind]
        if l3.bias is not None:
          l3.bias.data = l1.bias[col_ind]
          l3.bias_scores.data = l1.bias_scores[col_ind]
        prev_permutation = col_ind
    return m3


def test_weight_matching():
    print("test weight matching")
    w1_A = torch.tensor([[1, 2], [3, 4]]).float()
    w1_B = torch.tensor([[3, 4], [0, 2]]).float()

    w2_A = torch.tensor([[5, 6], [7, 8]]).float()
    w2_B = torch.tensor([[6, 5], [8, 7]]).float()
    # w2_B = torch.tensor([[8, 7], [6, 5]]).float()

    w3_A = torch.tensor([[9, 10]]).float()
    w3_B = torch.tensor([[9, 10]]).float()
    # w3_B = torch.tensor([[10, 9]]).float()

    m1 = torch.nn.Sequential(
        torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    )
    m2 = torch.nn.Sequential(
        torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    )
    m1[0].weight.data = w1_A
    m1[1].weight.data = w2_A
    m1[2].weight.data = w3_A
    m2[0].weight.data = w1_B
    m2[1].weight.data = w2_B
    m2[2].weight.data = w3_B

    def _print_all():
        print_("A")
        print_(m1[0].weight.data, m1[1].weight.data, m1[2].weight.data)
        print_("prod", m1[0].weight.data.T @ m1[1].weight.data.T @ m1[2].weight.data.T)
        print_("B")
        print_(m2[0].weight.data, m2[1].weight.data, m2[2].weight.data)
        print_("prod", m2[0].weight.data.T @ m2[1].weight.data.T @ m2[2].weight.data.T)

    print_("pre-matching")
    _print_all()
    dot_product_matching(m2, m1, inplace=True)
    print_("post-matching")
    _print_all()


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
    dot_product_matching_with_scores(m2, m1, inplace=True)
    print_("post-matching")
    _print_all()


def activation_matching(a1, a2):
    # permute a2 to match a1 as close as possible in terms of matrix frobenius norm
    # a1 and a2 are of shape (n_training_points, features)
    a1 = a1.detach().cpu().numpy().T
    a2 = a2.detach().cpu().numpy().T
    _, col_ind = linear_sum_assignment(-a1 @ a2.T)
    return col_ind


def match_model_2_to_1(data, model1, model2):
    # side effects on model2 (changing the weights to match model1)
    # W'_l = P_l @ W_l @ P_{l-1}.T
    # data.shape = (n_training_points, features)
    # model1 and model2 are of type torch.nn.Sequential
    prev_permutation = torch.arange(data.shape[1])
    for i, l2 in enumerate(model2):
        data1_i = model1[:i](data)
        data2_i = model2[:i](data)
        permutation = activation_matching(data1_i, data2_i)
        l2_inverse_prev_permuted = (l2.weight.data.T[:, prev_permutation]).T
        l2.weight.data = l2_inverse_prev_permuted[:, permutation]
        l2.bias.data = l2.bias.data[permutation]
        prev_permutation = permutation

    return model2


if __name__ == "__main__":
    test_weight_matching()
    test_weight_matching_with_masks()
    # a1 = torch.tensor([[1, 2], [4, 5], [7, 8]])
    # a2 = torch.tensor([[2, 1], [4.9, 5], [7, 8]])
    # print(activation_matching(a1, a2))


# %%

