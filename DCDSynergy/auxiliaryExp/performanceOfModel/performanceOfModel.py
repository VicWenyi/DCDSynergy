import random
import sys

import dgl
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.stats import gaussian_kde

from module.DCDSynergy import DCDSynergy

sys.path.insert(0, sys.path[0]+"/../../")
from utils import metrics, dataset, DataLoader, collate_merg
import numpy as np
import torch
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def performanceOfModel():
    # 原始代码部分保持不变
    model.eval()
    trues = []
    preds = []
    predab = []
    predba = []

    data = pickle.load(open("../../data/ONEIL_test_addproteintoDruggraphyWithProteinInterconnection.pkl", 'rb'))

    data = dataset([i[0] for i in data], [i[1] for i in data], device)
    dataL = DataLoader(data, batch_size=256, shuffle=False, collate_fn=lambda x: collate_merg(x, device))

    with torch.no_grad():
        with tqdm(total=len(dataL)) as tqq:
            for i, (dAB, dBA, c, y) in enumerate(dataL):
                pred1, _ = model((dAB, c))
                pred2, _ = model((dBA, c))
                trues.append(y)
                predab.append(pred1)
                predba.append(pred2)
                preds.append((pred1 + pred2) / 2)
                tqq.set_description(f"Item:{i}")
                tqq.update(1)

        # 处理最后一个batch的padding
        offset = trues[-2].shape[0] - trues[-1].shape[0]
        trues[-1].resize_(trues[-2].shape)
        preds[-1].resize_(preds[-2].shape)
        predab[-1].resize_(predab[-2].shape)
        predba[-1].resize_(predba[-2].shape)

        # 转换数据为tensor
        trues = torch.stack(trues, 0).view(-1).cpu()
        preds = torch.stack(preds, 0).view(-1).cpu()
        predab = torch.stack(predab, 0).view(-1).cpu()
        predba = torch.stack(predba, 0).view(-1).cpu()

        if offset > 0:
            trues = trues[:-offset]
            preds = preds[:-offset]
            predab = predab[:-offset]
            predba = predba[:-offset]

    # 转换为numpy数组
    trues = np.array(trues)
    preds = np.array(preds)

    # 计算评估指标（保持三位小数）
    mse = metrics.mse(trues, preds)
    mae = metrics.mae(trues, preds)
    r = np.corrcoef(trues, preds)[0, 1]

    plt.figure(figsize=(10, 8), dpi=300)

    # 自定义蓝-白-红渐变
    # colors = [
    #     (0.0, (0, 0, 1)),  # 底部蓝色
    #     (0.5, (1, 1, 1)),  # 中间白色（原先是黄色，改为白色）
    #     (1.0, (1, 0, 0))  # 顶部红色
    # ]
    colors = [
        (0.0, 'mediumblue'),  # 底部蓝色
        (0.5, 'white'),  # 中间白色（原先是黄色，改为白色）
        (1.0, 'red')  # 顶部红色
    ]

    cmap = LinearSegmentedColormap.from_list('custom_red_white_blue', colors, N=256)

    # 调整坐标轴范围到-90至30
    x_range = [-90, 30]
    y_range = [-90, 30]

    # 计算 KDE 密度
    xy = np.vstack([trues, preds])
    kde = gaussian_kde(xy)
    density = kde(xy)

    # 绘制散点图（颜色代表密度） - 关键修改：将LogNorm改为Normalize
    plt.scatter(
        trues, preds,
        c=density,
        cmap=cmap,
        norm=plt.Normalize(vmin=0.002, vmax=0.006),  # 改为线性归一化
        s=10,
        alpha=0.5,
        edgecolors='none'
    )

    # 其他设置保持不变
    plt.plot(x_range, y_range, 'k-', linewidth=1, alpha=0.8)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xticks(np.arange(-90, 40, 20))
    plt.yticks(np.arange(-90, 40, 20))
    plt.xlabel("True Score", fontsize=12, labelpad=10)
    plt.ylabel("Predicted Score", fontsize=12, labelpad=10)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # 设置为2，比默认的1更粗

    # 调整颜色条（5个刻度） - 现在刻度间距将均匀分布
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    cb.set_ticks([0.002, 0.003, 0.004, 0.005, 0.006])
    cb.set_ticklabels(['2×10$^{-3}$', '3×10$^{-3}$', '4×10$^{-3}$', '5×10$^{-3}$', '6×10$^{-3}$'])

    plt.text(28, -88, f"MSE = {mse:.3f}\nMAE = {mae:.3f}\nPerson = {r:.3f}",
             ha='right', va='bottom', fontsize=11,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round'))

    plt.title("Scatterplot of test data", fontsize=14, pad=20)
    plt.savefig('scatterplot_test_data.png', bbox_inches='tight', dpi=300)
    plt.savefig('scatterplot_test_data.pdf', bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    performanceOfModel()