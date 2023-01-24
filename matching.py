# %%

from scipy.optimize import linear_sum_assignment
import torch
from copy import deepcopy


def dot_product_matching(m1, m2, inplace=True):
    m3 = m1 if inplace else deepcopy(m1)
    prev_permutation = None
    for l1, l2, l3 in zip(m1.children(), m2.children(), m3.children()):
        if not isinstance(l1, torch.nn.Linear):
            continue
        w1 = l1.weight.data.clone()
        w2 = l2.weight.data
        if prev_permutation is not None:
            w1 = w1.T[prev_permutation, :].T
        w1_ = w1.detach().cpu().numpy()
        w2_ = w2.detach().cpu().numpy()
        _, col_ind = linear_sum_assignment(-w1_ @ w2_.T)
        prev_permutation = col_ind
        l3.weight.data = w1[col_ind, :]
        l3.bias.data = l1.bias[col_ind]
    return m3


def test():
    w1_A = torch.tensor([[1, 2], [3, 4]]).float()
    w1_B = torch.tensor([[3, 4], [0, 2]]).float()

    # greedy dot product matching recovers this ^

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
    print("pre-matching")
    print("A")
    print(m1[0].weight.data, m1[1].weight.data, m1[2].weight.data)
    print("prod", m1[0].weight.data.T @ m1[1].weight.data.T @ m1[2].weight.data.T)
    print("B")
    print(m2[0].weight.data, m2[1].weight.data, m2[2].weight.data)
    print("prod", m2[0].weight.data.T @ m2[1].weight.data.T @ m2[2].weight.data.T)
    dot_product_matching(m2, m1, inplace=False)
    print("post-matching")
    print("A")
    print(m1[0].weight.data, m1[1].weight.data, m1[2].weight.data)
    print("prod", m1[0].weight.data.T @ m1[1].weight.data.T @ m1[2].weight.data.T)
    print("B")
    print(m2[0].weight.data, m2[1].weight.data, m2[2].weight.data)
    print("prod", m2[0].weight.data.T @ m2[1].weight.data.T @ m2[2].weight.data.T)


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
    test()
    # a1 = torch.tensor([[1, 2], [4, 5], [7, 8]])
    # a2 = torch.tensor([[2, 1], [4.9, 5], [7, 8]])
    # print(activation_matching(a1, a2))


# %%

