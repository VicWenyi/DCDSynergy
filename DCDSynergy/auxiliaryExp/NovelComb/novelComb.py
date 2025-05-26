import random
import sys

from dataproc import DrugtoGraphy
from module.DCDSynergy import DCDSynergy

sys.path.insert(0, sys.path[0]+"/../../")
from utils import evaluate
import pandas as pd
import torch
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
import dgl
node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')

P = {
    "SEED": 5,
    "EPOCHES": 100,
    "BATCH_SIZE": 64,
    "TEST_BATCH": 256,
    "dropout": 0.1,
    "lr": 0.0003,
    "lr_gamma": 0.95,
}
SEED = 5
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


def novelComb(data, name):
    (label, pred), _ = evaluate(model, data, device)
    pred = pred.sort(descending=True)

    print("drugA\tdrugB\tcell-line\tsynergist")
    for i in list(range(20)):
        _i = pred.indices[i].item()
        print(f"{name[_i][0]}\t{name[_i][1]}\t{name[_i][2]}\t{pred.values[i].item()}")


def get_novelComb_dataset():
    comb = pd.read_csv(f'../../data/ONEIL.csv')
    comb = comb[['drug_row', 'drug_col', 'cell_line_name', 'synergy_loewe']]
    
    # Outlier cleaning
    comb = comb[comb['synergy_loewe']<400]

    comb = comb.sample(frac=1, random_state=SEED)

    smiles = pd.read_csv('../../data/drugs.csv')
    smiles = smiles[['dname', 'smiles']]
    _d = list(set(list(comb['drug_row'])+list(comb['drug_col'])))
    smiles = smiles[smiles['dname'].isin(_d)]
    smiles.set_index('dname', inplace=True)

    cells = pd.read_csv('../../data/cells.csv')
    _c = list(set(comb['cell_line_name']))
    cells = cells[cells['name'].isin(_c)]
    cells.set_index('name', inplace=True)

    mark = []
    landmark = pd.read_csv('../../data/raw/landmarkGene.txt', sep='\t')
    landmark = list(landmark['Symbol'])
    exclude = ['PAPD7', 'HDGFRP3', 'AARS', 'TMEM2', 'TMEM5', 'SQRDL', 'H2AFV', 'KIAA0907', 'HIST2H2BE', 'KIAA0355', 'IKBKAP', 'TSTA3', 'TMEM110', 'WRB', 'FAM69A', 'FAM57A', 'ATP5S', 'NARFL', 'KIF1BP', 'HN1L', 'EPRS', 'HIST1H2BK']
    '''
    The data for the following genes does not exist in CCLE_expression_full, so it is deleted.
    'PAPD7', 'HDGFRP3', 'AARS', 'TMEM2', 'TMEM5', 'SQRDL', 'H2AFV', 'KIAA0907', 'HIST2H2BE', 'KIAA0355', 'IKBKAP', 'TSTA3', 'TMEM110', 'WRB', 'FAM69A', 'FAM57A', 'ATP5S', 'NARFL', 'KIF1BP', 'HN1L', 'EPRS', 'HIST1H2BK'
    '''
    mark = list(set(landmark)-set(exclude))
    mark.sort()
    cells = cells[mark]

    cs = list(set(comb['cell_line_name']))

    olddcs = [(i[1]['drug_row'], i[1]['drug_col']) for i in comb.iterrows()]
    olddcs = list(set(olddcs))

    ds = list(set(list(comb['drug_row']) + list(comb['drug_col'])))
    ds.sort()
    newdcs = []

    for i in list(range(len(ds))):
        for j in list(range(i+1, len(ds))):
            if (ds[i], ds[j]) not in olddcs:
                newdcs.append((ds[i], ds[j]))

    d_graph = {}
    max_drug_nodes = 90
    save_set = []
    name_set = []

    _i = 1
    for _dc in newdcs:
        for _c in cs:
            smileA = smiles.loc[_dc[0]]['smiles']
            smileB = smiles.loc[_dc[1]]['smiles']
            cellGene = cells.loc[_c].values

            gA = DrugtoGraphy(smileA)
            gB = DrugtoGraphy(smileB)

            old_num_nodes_A = gA.number_of_nodes()
            old_num_nodes_B = gB.number_of_nodes()

            v_p = cellGene
            v_p = torch.tensor(v_p, dtype=torch.float32)
            v_p = torch.cat((v_p, torch.zeros(19)), dim=0)
            v_p_split = torch.split(v_p, 75, dim=0)

            actual_node_feats_A = gA.ndata['h']
            num_actual_nodes_A = actual_node_feats_A.shape[0]
            num_virtual_nodes_A = max_drug_nodes - num_actual_nodes_A - v_p_split.__len__()
            virtual_node_feat_A = torch.cat((torch.zeros(num_virtual_nodes_A, 74), torch.ones(num_virtual_nodes_A, 1)),
                                            1)  # virtual_node_feat：创建虚拟节点的特征张量，由一个形状为[num_virtual_nodes, 74]的全零张量和一个形状为[num_virtual_nodes, 1]的全一张量在维度 1 上拼接而成。为了区别actual和virtual
            gA.add_nodes(num_virtual_nodes_A, {"h": virtual_node_feat_A})

            actual_node_feats_B = gB.ndata['h']
            num_actual_nodes_B = actual_node_feats_B.shape[0]
            num_virtual_nodes_B = max_drug_nodes - num_actual_nodes_B - v_p_split.__len__()
            virtual_node_feat_B = torch.cat((torch.zeros(num_virtual_nodes_B, 74), torch.ones(num_virtual_nodes_B, 1)),
                                            1)  # virtual_node_feat：创建虚拟节点的特征张量，由一个形状为[num_virtual_nodes, 74]的全零张量和一个形状为[num_virtual_nodes, 1]的全一张量在维度 1 上拼接而成。为了区别actual和virtual
            gB.add_nodes(num_virtual_nodes_B, {"h": virtual_node_feat_B})

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

            save_set.append(((gA, gB, cellGene), _i))
            name_set.append((_dc[0], _dc[1], _c, _i))
            _i += 1
    
    return save_set, name_set



if __name__ == "__main__":
    save_set, name_set = get_novelComb_dataset()
    novelComb(save_set, name_set)