import heapq
import pickle
import random
from collections import Counter

import dgl

import pandas as pd
import torch
from dgllife.utils import CanonicalAtomFeaturizer, smiles_to_bigraph
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm
import numpy as np
from module.DCDSynergy import DCDSynergy
from utils import collate_merg, dataset
from torch.utils.data import DataLoader

P = {
    "SEED": 5,
    "EPOCHES": 100,
    "BATCH_SIZE": 64,
    "TEST_BATCH": 256,
    "dropout": 0.1,
    "lr": 0.0003,
    "lr_gamma": 0.95,
}

node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
def DrugtoGraphy(drug):
    drugGraph = smiles_to_bigraph(drug, node_featurizer=node_featurizer)
    # 处理节点特征
    actual_node_feats = drugGraph.ndata.pop('h')  # v_d.ndata.pop('h')：从图v_d的节点数据中取出节点特征h，并将其从节点数据中移除。size:[节点：20,特征长度：74]
    num_actual_nodes = actual_node_feats.shape[0]  # num_actual_nodes：获取实际节点特征的数量，即图中实际节点的数量。
    # num_virtual_nodes = self.max_drug_nodes - num_actual_nodes # num_virtual_nodes：计算需要添加的虚拟节点的数量，通过self.max_drug_nodes（在__init__方法中指定的最大药物节点数）减去实际节点数得到。
    virtual_node_bit = torch.zeros([num_actual_nodes, 1])  # virtual_node_bit：创建一个形状为[num_actual_nodes, 1]的全零张量，用于标识实际节点。
    actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit),1)  # torch.cat((actual_node_feats, virtual_node_bit), 1)：将实际节点特征和虚拟节点标识张量在维度 1 上拼接，更新实际节点特征。为了区别actual和virtual
    drugGraph.ndata['h'] = actual_node_feats
    return drugGraph

def get_att_values_and_features(model, dataL, device):
    model.eval()
    with torch.no_grad():
        for (dAB, dBA, c) in dataL:
            batchx = (dAB, c)
            _, att_maps = model(batchx)
    return att_maps

def get_top_attention_coords(att_matrix, drug_nodes, percentile=5):
    """
    从注意力矩阵中提取前30%权重值及其坐标，确保横坐标不超过(drug_nodes-1)

    参数:
        att_matrix (np.ndarray): 2D注意力矩阵 [num_rows, num_cols]
        drug_nodes (int): 药物分子中的原子数
        percentile (int): 要提取的百分比，默认为30

    返回:
        top_coords: 包含前30%的(row, col)坐标的列表
        top_values: 对应的权重值列表
        highlight_atoms: 高亮原子索引列表
    """
    # 1. 首先筛选出横坐标有效的行
    valid_rows = np.arange(att_matrix.shape[0]) <= (drug_nodes - 1)
    valid_matrix = att_matrix[valid_rows, :]

    # 2. 计算前30%对应的元素数量
    total_elements = valid_matrix.size
    top_k = int(total_elements * percentile / 100)

    # 确保至少选择一个元素
    top_k = max(top_k, 1)

    # 3. 获取Top K的平铺索引
    flat_indices = np.argpartition(valid_matrix.ravel(), -top_k)[-top_k:]

    # 4. 转换为在valid_matrix中的坐标
    valid_coords = np.unravel_index(flat_indices, valid_matrix.shape)

    # 5. 转换回原始矩阵的坐标
    original_rows = np.where(valid_rows)[0][valid_coords[0]]
    original_cols = valid_coords[1]

    # 6. 提取对应的值
    values = att_matrix[original_rows, original_cols]

    # 7. 按值降序排序
    sorted_indices = np.argsort(values)[::-1]

    # 8. 准备最终结果
    top_coords = list(zip(original_rows[sorted_indices], original_cols[sorted_indices]))
    top_values = values[sorted_indices]
    highlight_atoms = [row for row, _ in top_coords]

    # 验证结果
    assert all(row <= (drug_nodes - 1) for row, _ in top_coords), "存在不合规的横坐标！"

    return top_coords, top_values, highlight_atoms

# def get_top_attention_coords(att_matrix, drug_nodes, top_k=10):
#     """
#     从注意力矩阵中提取Top K权重值及其坐标，确保横坐标不超过(drug_nodes-1)
#
#     参数:
#         att_matrix (np.ndarray): 2D注意力矩阵 [num_rows, num_cols]
#         drug_nodes (int): 药物分子中的原子数
#         top_k (int): 要提取的Top K值，默认为10
#
#     返回:
#         top_coords: 包含(top_k)个(row, col)坐标的列表
#         top_values: 对应的权重值列表
#         highlight_atoms: 高亮原子索引列表
#     """
#     # 1. 首先筛选出横坐标有效的行
#     valid_rows = np.arange(att_matrix.shape[0]) <= (drug_nodes - 1)
#     valid_matrix = att_matrix[valid_rows, :]
#
#     # 2. 获取Top K的平铺索引
#     flat_indices = np.argpartition(valid_matrix.ravel(), -top_k)[-top_k:]
#
#     # 3. 转换为在valid_matrix中的坐标
#     valid_coords = np.unravel_index(flat_indices, valid_matrix.shape)
#
#     # 4. 转换回原始矩阵的坐标
#     original_rows = np.where(valid_rows)[0][valid_coords[0]]
#     original_cols = valid_coords[1]
#
#     # 5. 提取对应的值
#     values = att_matrix[original_rows, original_cols]
#
#     # 6. 按值降序排序
#     sorted_indices = np.argsort(values)[::-1]
#
#     # 7. 准备最终结果
#     top_coords = list(zip(original_rows[sorted_indices], original_cols[sorted_indices]))
#     top_values = values[sorted_indices]
#     highlight_atoms = [row for row, _ in top_coords]
#
#     # 验证结果
#     assert all(row <= (drug_nodes - 1) for row, _ in top_coords), "存在不合规的横坐标！"
#
#     return top_coords, top_values, highlight_atoms

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

with open(f"./interpretation-sample2.pkl", 'rb') as fp:
    ds = pickle.load(fp)

X = [i[0] for i in ds]
Y = [i[1] for i in ds]
test_data = dataset(X, Y, device)
testDL = DataLoader(test_data, batch_size=P["TEST_BATCH"], shuffle=False, collate_fn=lambda x: collate_merg(x, device))

sample_data = []
for (dAB, dBA, c, y) in testDL:
    for i in range(len(testDL)):
        sample_data.append((dAB,dBA,c))

att_maps = get_att_values_and_features(model,sample_data, device)

att_map1 = att_maps[0]
att_map2 = att_maps[2]

attn1 = att_map1[0,:,:]
attn2 = att_map2[0,:,:]

att_values1 = np.array(attn1.cpu())
att_values2 = np.array(attn2.cpu())

df = pd.read_csv("./interpretation-sample2.csv")
drug1_name = df["drug_row"].item()
drug2_name = df["drug_col"].item()

df = pd.read_csv("./drugs.csv")
# 筛选 dname 列匹配 drug_name 的行（忽略大小写）
drug1_smiles = df.loc[df['dname'].str.lower() == drug1_name.lower(), 'smiles'].item()
drug1_smiles_Original = DrugtoGraphy(drug1_smiles)
drug1_nodes = drug1_smiles_Original.number_of_nodes()
drug1_mol = Chem.MolFromSmiles(drug1_smiles)

drug2_smiles = df.loc[df['dname'].str.lower() == drug2_name.lower(), 'smiles'].item()
drug2_smiles_Original = DrugtoGraphy(drug2_smiles)
drug2_nodes = drug2_smiles_Original.number_of_nodes()
drug2_mol = Chem.MolFromSmiles(drug2_smiles)

_,_,highlight_atoms1 = get_top_attention_coords(att_values1, drug1_nodes)
_,_,highlight_atoms2 = get_top_attention_coords(att_values2, drug2_nodes)

highlight_atoms1 = [int(x) for x in highlight_atoms1]
highlight_atoms2 = [int(x) for x in highlight_atoms2]

atom_counts1 = Counter(highlight_atoms1)
atom_counts2 = Counter(highlight_atoms2)

# 生成渐变色函数
def get_gradient_colors(counts_dict, base_color=(1, 0.4, 0)):
    """生成从浅到深的渐变色"""
    max_count = max(counts_dict.values(), default=1)
    colors = []
    for atom, count in counts_dict.items():
        # 计算颜色强度（0.3-1.0范围保证基础颜色可见）
        intensity = 0.3 + 0.7 * (count / max_count)
        gradient_color = tuple(c * intensity for c in base_color)
        colors.append((atom, gradient_color))
    return colors
# highlight_atoms1 = list(set(highlight_atoms1))
# highlight_atoms2 = list(set(highlight_atoms2))



# 生成带渐变色的原子-颜色映射
gradient_colors1 = get_gradient_colors(atom_counts1)
gradient_colors2 = get_gradient_colors(atom_counts2)

# 拆分原子索引和颜色列表
atoms1, colors1 = zip(*gradient_colors1) if gradient_colors1 else ([], [])
atoms2, colors2 = zip(*gradient_colors2) if gradient_colors2 else ([], [])

# 绘制带渐变色的分子图像
img1 = Draw.MolToImage(
    drug1_mol,
    highlightAtoms=atoms1,
    highlightColors=colors1,
    size=(800, 600),
    fitImage=True
)

# 保存多格式图像
img1.save(f"./highlighted_molecule_{drug1_name}2.png")
plt.figure(figsize=(8, 6))
plt.imshow(img1)
plt.axis('off')
plt.savefig(f"./highlighted_molecule_{drug1_name}2.pdf", bbox_inches='tight', pad_inches=0)
plt.close()


img2 = Draw.MolToImage(
    drug2_mol,
    highlightAtoms=atoms2,
    highlightColors=colors2,
    size=(800, 600),
    fitImage=True
)
img2.save(f"./highlighted_molecule_{drug2_name}.png")
plt.figure(figsize=(8, 6))
plt.imshow(img2)
plt.axis('off')
plt.savefig(f"./highlighted_molecule_{drug2_name}.pdf", bbox_inches='tight', pad_inches=0)
plt.close()