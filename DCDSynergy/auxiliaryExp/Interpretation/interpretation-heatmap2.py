import pickle
import random
import re

import seaborn as sns
import dgl
import pandas as pd
import torch
from dgllife.utils import CanonicalAtomFeaturizer, smiles_to_bigraph
from matplotlib import pyplot as plt
from rdkit import Chem
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

import numpy as np


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


def sort_neighbors(current_atom, prev_atom, priorities):
    neighbors = []
    for bond in current_atom.GetBonds():
        neighbor = bond.GetOtherAtom(current_atom)
        if neighbor != prev_atom:  # 排除回溯
            neighbors.append(neighbor)

    # 按优先级排序（示例规则）
    return sorted(neighbors,
                  key=lambda x: (
                      -priorities.index(x.GetSymbol()) if x.GetSymbol() in priorities else 100,
                      x.GetDegree()  # 次要排序：连接度
                  ))

def select_start_atom(mol, priorities):
    # 按优先级选择第一个匹配的原子
    for pattern in priorities:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == pattern:
                return atom.GetIdx()
    # 默认选择第一个原子
    return 0

def dfs_traversal(mol, start_atom=-1, priorities=None):
    visited = set()
    stack = []
    path = []

    # 1. 选择起始原子
    if start_atom == -1:
        start_atom = select_start_atom(mol, priorities)  # 自定义优先级选择

    # 2. 初始化栈
    stack.append((mol.GetAtomWithIdx(start_atom), None))

    # 3. DFS遍历核心
    while stack:
        current_atom, prev_atom = stack.pop()
        if current_atom.GetIdx() not in visited:
            # 记录原子符号
            path.append(current_atom.GetSymbol())
            visited.add(current_atom.GetIdx())

            # 获取相邻原子并按优先级排序
            neighbors = sort_neighbors(current_atom, prev_atom, priorities)
            for neighbor in reversed(neighbors):  # 保证顺序正确
                stack.append((neighbor, current_atom))

    return path

def smiles_to_heatmap_label(smiles):
    # 实现深度优先遍历与符号转换
    label = []
    for atom in dfs_traversal(smiles):  # 自定义遍历算法
        if atom in ['O','F','N']:
            label.append(atom.upper())  # 官能团大写
        elif atom == 'C':
            label.append('c')          # 碳骨架小写
    return ''.join(label)

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

with open(f"./interpretation-sample1.pkl", 'rb') as fp:
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

att_values1 = att_values1[:9, :]
att_values2 = att_values2[:26, :]

df = pd.read_csv("./interpretation-sample1.csv")
drug1_name = df["drug_row"].item()
drug2_name = df["drug_col"].item()
cell_name = df["cell_line_name"].item()

df = pd.read_csv("./drugs.csv")
# 筛选 dname 列匹配 drug_name 的行（忽略大小写）
drug1_smiles = df.loc[df['dname'].str.lower() == drug1_name.lower(), 'smiles'].item()
drug1_mol = Chem.MolFromSmiles(drug1_smiles)

drug2_smiles = df.loc[df['dname'].str.lower() == drug2_name.lower(), 'smiles'].item()
drug2_mol = Chem.MolFromSmiles(drug2_smiles)

df = pd.read_csv("../../data/cells.csv")

cell_line = df[df["name"]==cell_name]
cell_line.set_index('name', inplace=True)
landmark = pd.read_csv('../../data/raw/landmarkGene.txt', sep='\t')
landmark = list(landmark['Symbol'])
exclude = ['PAPD7', 'HDGFRP3', 'AARS', 'TMEM2', 'TMEM5', 'SQRDL', 'H2AFV', 'KIAA0907', 'HIST2H2BE', 'KIAA0355',
           'IKBKAP', 'TSTA3', 'TMEM110', 'WRB', 'FAM69A', 'FAM57A', 'ATP5S', 'NARFL', 'KIF1BP', 'HN1L', 'EPRS',
           'HIST1H2BK']
'''
The data for the following genes does not exist in CCLE_expression_full, so it is deleted.
'PAPD7', 'HDGFRP3', 'AARS', 'TMEM2', 'TMEM5', 'SQRDL', 'H2AFV', 'KIAA0907', 'HIST2H2BE', 'KIAA0355', 'IKBKAP', 'TSTA3', 'TMEM110', 'WRB', 'FAM69A', 'FAM57A', 'ATP5S', 'NARFL', 'KIF1BP', 'HN1L', 'EPRS', 'HIST1H2BK'
'''
mark = list(set(landmark) - set(exclude))
mark.sort()
cell_line = cell_line[mark]

# 假设你的数据格式：
# att_map: 2D数组，行对应基因，列对应原子
# gene_names: 基因列表 [TSPAN6, TNMD, DPM1...]
# atom_labels: 原子标签列表 [C1, O2, N3...]

# 准备数据
gene_names = cell_line.columns[0:].tolist()
atom_labels = "CCNCCCNCOCncNcCcccCccFncOc"

# 增大画布尺寸
plt.figure(figsize=(9 * 0.8 + 1.5, 3.2))

# 计算8个均匀分布的刻度值
ticks = np.linspace(0.0007, 0.0015, 8)

ax = sns.heatmap(
    att_values2.T,
    annot=False,
    cmap="coolwarm",
    linewidths=0,
    cbar_kws={
        "label": "Attention Score",
        "ticks": ticks,
        "format": "%.4f",
        "aspect": 8,
        "pad": 0.05,
        "shrink": 0.8
    },
    vmin=0.0007,
    vmax=0.0015,
    square=False
)

# 设置 x 轴刻度（24个）
ax.set_xticks(np.arange(26) + 0.5)

# 关键修改：设置 x 轴标签为 atom_labels 中的字符
ax.set_xticklabels(list(atom_labels))  # 旋转90度避免重叠

# 调整药物名称标签
ax.set_xlabel(drug2_name,
             fontsize=12,
             labelpad=12,
             position=(0.5, -0.2))

# 调整 genes 标签显示
ax.set_yticks([])
ax.set_ylabel("genes",
             fontsize=12,
             rotation=90,
             va="center",
             ha="right",
             x=-0.1,      # 水平偏移（负值向左）
             labelpad=10)  # 额外间距（可选）

# 颜色条调整
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10, length=0)
cbar.ax.set_ylabel("Attention Score",
                  fontsize=10,
                  rotation=90,
                  labelpad=12,
                  va="center")

# 调整边距
plt.subplots_adjust(
    left=0.2,
    right=0.85,
    bottom=0.25,  # 增加底部空间以适应旋转的标签
    top=0.95
)

plt.savefig(f'{drug2_name}_attention_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{drug2_name}_attention_heatmap.pdf', bbox_inches='tight')
plt.show()