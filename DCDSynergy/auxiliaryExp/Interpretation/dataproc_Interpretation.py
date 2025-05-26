import itertools
import math
import pickle

import numpy as np
import pandas as pd
import torch
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from torch import nn

SEED=5
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

def get_train_test(study):
    # 读取并预处理数据
    comb = pd.read_csv(f'./{study}.csv')
    comb = comb[['drug_row', 'drug_col', 'cell_line_name', 'synergy_loewe']]

    # 异常值处理
    comb = comb[comb['synergy_loewe'] < 400]
    comb = comb.sample(frac=1, random_state=SEED)  # 打乱数据

    # 读取药物SMILES数据
    smiles = pd.read_csv('./data/drugs.csv')
    smiles = smiles[['dname', 'smiles']]
    _d = list(set(list(comb['drug_row']) + list(comb['drug_col'])))
    smiles = smiles[smiles['dname'].isin(_d)]
    smiles.set_index('dname', inplace=True)

    # 读取细胞系数据
    cells = pd.read_csv('./data/cells.csv')
    _c = list(set(comb['cell_line_name']))
    cells = cells[cells['name'].isin(_c)]
    cells.set_index('name', inplace=True)

    # 处理landmark基因
    landmark = pd.read_csv('./data/raw/landmarkGene.txt', sep='\t')
    landmark = list(landmark['Symbol'])
    exclude = ['PAPD7', 'HDGFRP3', 'AARS', 'TMEM2', 'TMEM5', 'SQRDL', 'H2AFV', 'KIAA0907', 'HIST2H2BE', 'KIAA0355',
               'IKBKAP', 'TSTA3', 'TMEM110', 'WRB', 'FAM69A', 'FAM57A', 'ATP5S', 'NARFL', 'KIF1BP', 'HN1L', 'EPRS',
               'HIST1H2BK']
    mark = list(set(landmark) - set(exclude))
    mark.sort()
    cells = cells[mark]

    # 准备保存所有数据
    all_data = []
    max_drug_nodes = 90

    for item in comb.itertuples():
        # 获取药物A和B的SMILES
        smileA = smiles.loc[item.drug_row]['smiles']
        smileB = smiles.loc[item.drug_col]['smiles']
        cellGene = cells.loc[item.cell_line_name].values

        # 构建药物图
        gA = DrugtoGraphy(smileA)
        gB = DrugtoGraphy(smileB)

        old_num_nodes_A = gA.number_of_nodes()
        old_num_nodes_B = gB.number_of_nodes()

        # 处理细胞基因特征
        v_p = cellGene
        v_p = torch.tensor(v_p, dtype=torch.float32)
        v_p = torch.cat((v_p, torch.zeros(19)), dim=0)
        v_p_split = torch.split(v_p, 75, dim=0)

        # 添加虚拟节点到药物A图
        actual_node_feats_A = gA.ndata['h']
        num_actual_nodes_A = actual_node_feats_A.shape[0]
        num_virtual_nodes_A = max_drug_nodes - num_actual_nodes_A - v_p_split.__len__()
        virtual_node_feat_A = torch.cat((torch.zeros(num_virtual_nodes_A, 74),
                                         torch.ones(num_virtual_nodes_A, 1)), 1)
        gA.add_nodes(num_virtual_nodes_A, {"h": virtual_node_feat_A})

        # 添加虚拟节点到药物B图
        actual_node_feats_B = gB.ndata['h']
        num_actual_nodes_B = actual_node_feats_B.shape[0]
        num_virtual_nodes_B = max_drug_nodes - num_actual_nodes_B - v_p_split.__len__()
        virtual_node_feat_B = torch.cat((torch.zeros(num_virtual_nodes_B, 74),
                                         torch.ones(num_virtual_nodes_B, 1)), 1)
        gB.add_nodes(num_virtual_nodes_B, {"h": virtual_node_feat_B})

        # 添加蛋白质节点到图中
        for i, sub_tensor in enumerate(v_p_split):
            tensor_expanded = sub_tensor.unsqueeze(0)

            num_nodes_A = gA.number_of_nodes()
            num_nodes_B = gB.number_of_nodes()

            new_node_id_A = num_nodes_A
            new_node_id_B = num_nodes_B

            if i == 0:
                pAid_first = new_node_id_A
                pBid_first = new_node_id_B

            new_node_feat = tensor_expanded

            src_A, dst_A = gA.edges()
            src_B, dst_B = gB.edges()

            add_src_A = torch.arange(old_num_nodes_A)
            add_src_B = torch.arange(old_num_nodes_B)
            add_dst_A = torch.full((old_num_nodes_A,), new_node_id_A)
            add_dst_B = torch.full((old_num_nodes_B,), new_node_id_B)

            new_src_A = torch.cat([src_A, add_src_A])
            new_dst_A = torch.cat([dst_A, add_dst_A])
            new_src_B = torch.cat([src_B, add_src_B])
            new_dst_B = torch.cat([dst_B, add_dst_B])

            new_gA = dgl.graph((new_src_A, new_dst_A))
            new_gB = dgl.graph((new_src_B, new_dst_B))

            old_feat_A = gA.ndata['h']
            new_feat_A = torch.cat([old_feat_A, new_node_feat], dim=0)
            new_gA.ndata['h'] = new_feat_A
            old_feat_B = gB.ndata['h']
            new_feat_B = torch.cat([old_feat_B, new_node_feat], dim=0)
            new_gB.ndata['h'] = new_feat_B

            gA = new_gA
            gB = new_gB

            # 实现蛋白质的互连
        src_A, dst_A = gA.edges()
        src_B, dst_B = gB.edges()

        add_src_A1 = torch.arange(pAid_first, pAid_first + 11)
        add_src_B1 = torch.arange(pBid_first, pBid_first + 11)
        add_dst_A1 = torch.arange(pAid_first + 1, pAid_first + 12)
        add_dst_B1 = torch.arange(pBid_first + 1, pBid_first + 12)

        add_src_A2 = torch.arange(pAid_first + 1, pAid_first + 12)
        add_src_B2 = torch.arange(pBid_first + 1, pBid_first + 12)
        add_dst_A2 = torch.arange(pAid_first, pAid_first + 11)
        add_dst_B2 = torch.arange(pBid_first, pBid_first + 11)

        add_src_A_all = torch.cat([add_src_A1, add_src_A2])
        add_src_B_all = torch.cat([add_src_B1, add_src_B2])
        add_dst_A_all = torch.cat([add_dst_A1, add_dst_A2])
        add_dst_B_all = torch.cat([add_dst_B1, add_dst_B2])

        new_src_A = torch.cat([src_A, add_src_A_all])
        new_dst_A = torch.cat([dst_A, add_dst_A_all])
        new_src_B = torch.cat([src_B, add_src_B_all])
        new_dst_B = torch.cat([dst_B, add_dst_B_all])

        new_gA = dgl.graph((new_src_A, new_dst_A))
        new_gB = dgl.graph((new_src_B, new_dst_B))

        new_gA.ndata['h'] = gA.ndata['h']
        new_gB.ndata['h'] = gB.ndata['h']

        gA = new_gA
        gB = new_gB

        gA = dgl.add_self_loop(gA)
        gB = dgl.add_self_loop(gB)

        # 保存数据
        all_data.append(((gA, gB, cellGene), item.synergy_loewe))

    # 将所有数据保存到一个文件中
    with open(f'./data/interpretation-sample1.pkl', 'wb') as f:
        pickle.dump(all_data, f)

if __name__ == "__main__":
    get_train_test("auxiliaryExp/Interpretation/interpretation-sample1")