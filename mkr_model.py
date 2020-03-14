# -*- coding: utf-8 -*-
# @Time    : 2020-02-27 16:06
# @Author  : WenYi
# @Contact : wenyi@cvte.com
# @Description :  《Multi-task Learning for KG enhanced Recommendation》

import torch.nn as nn
import torch
from utils import linear_layer


class MultiKR(nn.Module):
    """
    model base on 《Multi-task Learning for KG enhanced Recommendation》
    """
    def __init__(self, user_num, item_num, entity_num, relation_num, n_layer, embed_dim, hidden_layers, dropouts, output_rec):
        """

        :param user_num:
        :param item_num:
        :param entity_num:
        :param relation_num:
        :param n_layer:
        :param embed_dim:
        :param hidden_layers:
        :param dropouts:
        """
        super(MultiKR, self).__init__()

        # user embedding
        self.user_embed = nn.Embedding(user_num, embed_dim)

        # item embedding
        self.item_embed = nn.Embedding(item_num, embed_dim)

        # entity embedding
        self.entity_embed = nn.Embedding(entity_num, embed_dim)

        # relation embedding
        self.relation_embed = nn.Embedding(relation_num, embed_dim)

        # low mlp layer number
        self.n_layer = n_layer

        # compress vector
        self.compress_weight_vv = torch.rand((embed_dim, 1), requires_grad=True)
        self.compress_weight_ev = torch.rand((embed_dim, 1), requires_grad=True)
        self.compress_weight_ve = torch.rand((embed_dim, 1), requires_grad=True)
        self.compress_weight_ee = torch.rand((embed_dim, 1), requires_grad=True)
        self.compress_bias_v = torch.rand(1, requires_grad=True)
        self.compress_bias_e = torch.rand(1, requires_grad=True)

        # mlp for low layer
        self.user_low_mlp_layer = linear_layer(embed_dim, embed_dim, dropout=0.5)
        self.relation_low_mlp_layer = linear_layer(embed_dim, embed_dim, dropout=0.5)

        # mlp for kg sub model
        self.kg_layers = nn.Sequential()
        layers = [2*embed_dim] + hidden_layers
        for i in range(len(layers)-1):
            self.kg_layers.add_module(
                'kg_hidden_layer_{}'.format(i + 1),
                linear_layer(layers[i], layers[i+1], dropouts[i]))
        self.kg_layers.add_module('kg_last_layer', linear_layer(layers[-1], embed_dim))

        # mlp for recommend sub model
        self.rec_layers = nn.Sequential()
        layers = [2*embed_dim] + hidden_layers
        for i in range(len(layers)-1):
            self.rec_layers.add_module(
                'rec_hidden_layer_{}'.format(i + 1),
                linear_layer(layers[i], layers[i+1], dropouts[i]))
        self.rec_layers.add_module('rec_last_layer', linear_layer(layers[-1], output_rec))

    def __init_weight(self):
        # embedding weight init
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)

    def cross_compress_unit(self, item_embed, head_embed):
        """
        Cross&compress Unit for item and entity
        :param item_embed: batch * embed_dim
        :param head_embed: batch * embed_dim
        :return:
        """
        item_embed_reshape = item_embed.unsqueeze(-1) # batch_size * embed_dim * 1
        head_embed_reshape = head_embed.unsqueeze(-1) # batch_size * embed_dim * 1
        c = item_embed_reshape * head_embed_reshape.permute((0, 2, 1)) # batch_size * embed_dim * embed_dim
        c_t = head_embed_reshape * item_embed_reshape.permute((0, 2, 1)) # batch_size * embed_dim * embed_dim
        item_embed_c = torch.matmul(c,self.compress_weight_vv).squeeze() + \
                       torch.matmul(c_t, self.compress_weight_ev).squeeze() + self.compress_bias_v  # batch_size * embed_dim
        head_embed_c = torch.matmul(c, self.compress_weight_ve).squeeze() + \
                       torch.matmul(c_t, self.compress_weight_ee).squeeze() + self.compress_bias_e # batch_size * embed_dim
        return item_embed_c, head_embed_c

    def forward(self, data, train_type):

        if train_type == 'rec':
            # rec module
            user_embed = self.user_embed(data[0].long())   # batch * embed_dim
            item_embed = self.item_embed(data[1].long())   # batch * embed_dim
            head_embed = self.entity_embed(data[1].long()) # batch * embed_dim
            rec_target = data[2].float()
            for i in range(self.n_layer):
                user_embed = self.user_low_mlp_layer(user_embed)
                item_embed, head_embed = self.cross_compress_unit(item_embed, head_embed)
            high_layer = torch.cat((user_embed, item_embed), dim=1)
            rec_out = self.rec_layers(high_layer)
            return rec_out.squeeze(), rec_target
        else:
            # kg module
            head_embed = self.entity_embed(data[0].long())
            item_embed = self.item_embed(data[0].long())
            relation_embed = self.relation_embed(data[1].long())
            tail_embed = self.entity_embed(data[2].long())

            for i in range(self.n_layer):
                item_embed, head_embed = self.cross_compress_unit(item_embed, head_embed)
                relation_embed = self.relation_low_mlp_layer(relation_embed)
            high_layer = torch.cat((head_embed, relation_embed), dim=1)
            tail_out = self.kg_layers(high_layer)

            return tail_out, tail_embed


if __name__ == "__main__":
    import numpy as np
    user_num = 20
    item_num = 25
    entity_num = 50
    relation_num = 10
    n_layer = 2
    embed_dim = 32
    hidden_layers = [64, 64]
    dropouts = [0.5, 0.5]
    output_rec = 1
    mkr = MultiKR(user_num, item_num, entity_num, relation_num, n_layer, embed_dim, hidden_layers, dropouts, output_rec)
    d = np.array([[3, 5, 1], [4, 2, 0], [1, 2, 1], [4, 2, 0], [3, 6, 1], [4, 8, 0]])
    data = torch.from_numpy(d)
    type = 'kg'
    out_pred, out_true = mkr(data, type)
    print(out_pred.size(), out_true.size())
