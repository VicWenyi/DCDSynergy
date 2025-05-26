import random
import sys

import dgl

from module.DCDSynergy import DCDSynergy

sys.path.insert(0, sys.path[0]+"/../../")
from utils import metrics, evaluate
import pickle
import numpy as np
import torch


P = {
    "SEED": 5,
    "EPOCHES": 100,
    "BATCH_SIZE": 64,
    "TEST_BATCH": 256,
    "dropout": 0.1,
    "lr": 0.0003,
    "lr_gamma": 0.95,
}
dgl.random.seed(P["SEED"])
random.seed(P["SEED"])
torch.manual_seed(P["SEED"])
torch.cuda.manual_seed(P["SEED"])
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DCDSynergy(P)
state_dict = torch.load(r'./model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()
model.to(device)

def clssification(y_true, y_pred, threshold):

    y_true = np.array(list(map(lambda x : 1 if x >= threshold[1] else x, y_true)))
    y_true = np.array(list(map(lambda x : 0 if x < threshold[0] else x, y_true)))

    y_pred = np.array(list(map(lambda x : 1 if x >= threshold[1] else x, y_pred)))
    y_pred = np.array(list(map(lambda x : 0 if x < threshold[0] else x, y_pred)))

    _yt = []
    _yp = []
    for a, b in zip(y_true, y_pred):
        if (a==0 or a==1) and (b==0 or b==1):
            _yt.append(a)
            _yp.append(b)
    y_true = np.array(_yt)
    y_pred = np.array(_yp)

    acc = metrics.acc(y_true, y_pred)
    kappa = metrics.kappa(y_true, y_pred)
    bacc = metrics.bacc(y_true, y_pred)
    roc_auc = metrics.roc_auc(y_true, y_pred)
    prec = metrics.prec(y_true, y_pred)

    print(f"| roc_auc: {roc_auc} |")
    print(f"| acc: {acc} |")
    print(f"| bacc: {bacc} |")
    print(f"| prec: {prec} |")
    print(f"| kappa: {kappa} |")


if __name__ == '__main__':

    with open(f"../../data/ONEIL_test_addproteintoDruggraphyWithProteinInterconnection.pkl", 'rb') as fp:
        _e = pickle.load(fp)

    (trues, preds), _ = evaluate(model, _e, device)
    threshold = [-5, 5]

    clssification(trues, preds, threshold)
