# -*- coding: utf-8 -*-
# @Time    : 2020-02-27 16:17
# @Author  : WenYi
# @Contact : 1244058349@qq.com
# @Description :  some utils tool for model

import torch.nn as nn
import torch
import os
import numpy as np


def linear_layer(input, output, dropout=0):
    """
    linear layer for deep neural network model
    :param input: int, input layer unit
    :param output: int, output layer unit
    :param dropout: float, dropout ratio default 0
    :return: tensor
    """
    return nn.Sequential(
        nn.Linear(input, output),
        nn.LeakyReLU(),
        nn.Dropout(dropout)
    )


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_rating():
    print('reading rating file ...')

    # reading rating file
    rating_file = './MKR-data/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np)

    return n_user, n_item, train_data, eval_data, test_data


def load_kg():
    print('reading KG file ...')

    # reading kg file
    kg_file = './MKR-data/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg = np.load(kg_file + '.npy')
    else:
        kg = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg)

    n_entity = len(set(kg[:, 0]) | set(kg[:, 2]))
    n_relation = len(set(kg[:, 1]))

    return n_entity, n_relation, kg


def multi_loss(pred, target, types, loss_function):
    if types == "rec":
        loss = loss_function(pred, target)
        return loss
    else:

        loss = torch.sigmoid(torch.sum(pred * target))
        return loss


if __name__ == "__main__":
    n_user, n_item, train_data, eval_data, test_data = load_rating()
    print(n_user, n_item)
    print(train_data.shape, eval_data.shape, test_data.shape)
