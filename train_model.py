# -*- coding: utf-8 -*-
# @Time    : 2020-02-28 11:50
# @Author  : WenYi
# @Contact : 1244058349@qq.com
# @Description :  script description

from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
import time
import numpy as np
from utils import multi_loss


class TrainSet(Dataset):
    def __init__(self, data):
        self.user = data[0]
        self.item = data[1]
        self.target = data[2]

    def __getitem__(self, index):
        user = self.user[index]
        item = self.item[index]
        target = self.target[index]
        data = (user, item, target)
        return data

    def __len__(self):
        return len(self.target)


def train_epoch(model, train_loader_rec, train_loader_kg, optimizer, epoch, loss_function):
    model.train()
    epoch_loss = 0.0
    starts = time.time()
    if epoch % 5 != 4:
        for idxs, d in enumerate(train_loader_rec):
            optimizer.zero_grad()
            rec_pred, rec_ture = model(d, train_type='rec')
            rec_loss = multi_loss(rec_pred, rec_ture, 'rec', loss_function)
            epoch_loss += rec_loss.item()
            rec_loss.backward()
            optimizer.step()
        print("Epoch %d finished consume time is %.3f and loss is %.3f" % (epoch + 1, time.time() - starts,
                                                                           epoch_loss / len(train_loader_rec) + 1))
    else:
        for idxs, d in enumerate(train_loader_kg):
            optimizer.zero_grad()
            kg_pred, kg_true = model(d, train_type='kg')
            kg_loss = multi_loss(kg_pred, kg_true, 'kg', loss_function)
            epoch_loss += kg_loss.item()
            kg_loss.backward()
            optimizer.step()
        print("Epoch %d finished consume time is %.3f and loss is %.3f" %(epoch+1, time.time() - starts,
                                                                          epoch_loss/len(train_loader_kg)+1))


def valid_epoch(model, eval_loader_rec, epoch):
    model.eval()
    auc = []
    acc = []
    for idxs, d in enumerate(eval_loader_rec):
        rec_pred, rec_true = model(d, train_type='rec')
        auc.append(roc_auc_score(rec_true.detach().numpy(), rec_pred.detach().numpy()))
        rec_pred_con = []
        for i in rec_pred.detach().numpy():
            if i < 0.5:
                rec_pred_con.append(0)
            else:
                rec_pred_con.append(1)
        acc.append(accuracy_score(rec_true.detach().numpy(), np.array(rec_pred_con)))
    print("Epoch %d eval finished auc is %.3f and acc is %.3f" % (epoch+1, np.mean(auc), np.mean(acc)))
    # return np.mean(auc)


def test_epoch(model, test_loader_rec):
    model.eval()
    auc = []
    acc = []
    for idxs, d in enumerate(test_loader_rec):
        rec_pred, rec_true = model(d, train_type='rec')
        rec_pred_con = []
        for i in rec_true.detach().numpy():
            if i < 0.5:
                rec_pred_con.append(0)
            else:
                rec_pred_con.append(1)
        auc.append(roc_auc_score(rec_true.detach().numpy(), rec_pred.detach().numpy()))
        acc.append(accuracy_score(rec_true.detach().numpy(), np.array(rec_pred_con)))
    print("Test finished auc is %.3f and acc is %.3f" % (np.mean(auc), np.mean(acc)))


def train_model(model, train_loader_rec, train_loader_kg, eval_loader_rec, optimizer, loss_function, epochs):
    for epoch in range(epochs):
        train_epoch(model, train_loader_rec, train_loader_kg, optimizer, epoch, loss_function)
        valid_epoch(model, eval_loader_rec, epoch)

