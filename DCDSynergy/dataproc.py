import pickle
import pandas as pd
import torch
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
import dgl

SEED=5
node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')


def getDrugs():
    with open("./data/raw/drugs_drugcomb.txt", "r") as f:
        text = f.read()

    text = text.replace('null', 'None')
    tlist = eval(text)
    data = pd.DataFrame(tlist)
    df_drug = data[~data['smiles'].isin(['NULL'])]

    # Keep the drug that appears for the first time and delete the others.
    drugs = df_drug.groupby(['dname']).first().reset_index()

    allDrug = list(drugs['dname'])
    dropDrug = []
    for dname in allDrug:
        smiles = drugs[drugs['dname']==dname]['smiles'].values[0]
        g = smiles_to_bigraph(smiles, node_featurizer=node_featurizer)
        if g is None:
            dropDrug.append(dname)
    drugs = drugs[~drugs['dname'].isin(dropDrug)]

    drugs.set_index('id', inplace=True)
    drugs.sort_index(inplace=True)
    drugs.to_csv("./data/drugs.csv")



def getCells():
    with open("./data/raw/cells_drugcomb.txt", "r") as f:
        text = f.read()

    text = text.replace('null', 'None')
    tlist = eval(text)
    data = pd.DataFrame(tlist)
    data.set_index('id', inplace=True)
    data = data[~data['depmap_id'].isin(['NA'])]
    # Special value processing: The original depmap_id value was "ACH-000833; ACH-001189", which was modified to "ACH-000833"
    data.loc[1288, 'depmap_id'] = 'ACH-000833'
    df_cell = data[['name', 'depmap_id']]

    df_cellexpression = pd.read_csv('./data/raw/CCLE_expression_full.csv')
    df_cellexpression.rename(columns={'Unnamed: 0':'cellName'}, inplace=True)

    cellExpression = pd.merge(df_cell, df_cellexpression, how='inner', left_on='depmap_id', right_on='cellName')
    # Drop non-data columns
    cellExpression.drop(['depmap_id', 'cellName'], axis=1, inplace=True)
    # Drop columns with all zeros.
    cellExpression = cellExpression.loc[:, (cellExpression!=0).any(axis=0)]

    cellExpression.set_index('name', inplace=True)
    cellExpression.columns = cellExpression.columns.str.split(" \(").str[0]
    cellExpression.to_csv("./data/cells.csv")


# def getDrugCombs(study):
#     df_synergy = pd.read_csv('./data/raw/summary_v_1_5.csv')
#
#     synergy = df_synergy[['drug_row', 'drug_col', 'cell_line_name', 'study_name',
#         'tissue_name', 'synergy_zip', 'synergy_loewe', 'synergy_hsa', 'synergy_bliss']]
#
#     # Drop samples for non-cancer research
#     synergy = synergy[~synergy['study_name'].isin(['TOURET','GORDON','ELLINGER','MOTT','NCATS_SARS-COV-2DPI','BOBROWSKI','DYALL'])]
#     # Drop single drug samples
#     synergy = synergy[~synergy['drug_col'].isnull()]
#
#     synergy = synergy[synergy['study_name'].isin(study.split('_'))]
#     assert len(synergy)>0, "Study name was entered incorrectly."
#
#     # Drop non-numeric rows in synergy_loewe
#     synergy = synergy[~pd.to_numeric(synergy['synergy_loewe'] ,errors='coerce').isnull()]
#     synergy['synergy_loewe'] = synergy['synergy_loewe'].astype('float64')
#     # Drop rows without cell data
#     _cell = pd.read_csv('./data/cells.csv')
#     cells = list(_cell['name'])
#     del _cell
#     synergy = synergy[synergy['cell_line_name'].isin(cells)]
#     # Drop rows without drug data
#     _drug = pd.read_csv('./data/drugs.csv')
#     drugs = list(_drug['dname'])
#     del _drug
#     synergy = synergy[synergy['drug_row'].isin(drugs)]
#     synergy = synergy[synergy['drug_col'].isin(drugs)]
#
#     mask = list(map(lambda x, y: x>y, synergy['drug_row'].astype(str), synergy['drug_col'].astype(str)))
#     synergy.loc[mask, 'drug_row'], synergy.loc[mask, 'drug_col'] = synergy.loc[mask, 'drug_col'], synergy.loc[mask, 'drug_row']
#
#     merge_data = synergy[['drug_row', 'drug_col', 'cell_line_name', 'study_name', 'tissue_name']]
#     merge_data = merge_data.drop_duplicates(subset = ['drug_row', 'drug_col', 'cell_line_name'])
#     merge_data.set_index(['drug_row', 'drug_col', 'cell_line_name'], inplace=True)
#
#     comb = synergy.groupby(['drug_row', 'drug_col', 'cell_line_name']).agg('mean')
#
#     data = pd.merge(merge_data, comb, left_index=True, right_index=True)
#     data.reset_index(inplace=True)
#
#     data.to_csv(f'./data/{study}.csv', index=False)


def getDrugCombs(study):
    df_synergy = pd.read_csv('./data/raw/drugcombs_scored.csv')

    synergy = df_synergy[['Drug1', 'Drug2', 'Cell line', 'ZIP', 'Bliss', 'Loewe', 'HSA']]

    # Drop samples for non-cancer research
    # synergy = synergy[~synergy['study_name'].isin(['TOURET','GORDON','ELLINGER','MOTT','NCATS_SARS-COV-2DPI','BOBROWSKI','DYALL'])]
    # Drop single drug samples
    # synergy = synergy[~synergy['drug_col'].isnull()]

    # synergy = synergy[synergy['study_name'].isin(study.split('_'))]
    # assert len(synergy)>0, "Study name was entered incorrectly."

    # Drop non-numeric rows in synergy_loewe
    synergy = synergy[~pd.to_numeric(synergy['Loewe'] ,errors='coerce').isnull()]
    synergy['Loewe'] = synergy['Loewe'].astype('float64')
    # Drop rows without cell data
    _cell = pd.read_csv('./data/cells.csv')
    cells = list(_cell['name'])
    del _cell
    synergy = synergy[synergy['Cell line'].isin(cells)]
    # Drop rows without drug data
    _drug = pd.read_csv('./data/drugs.csv')
    drugs = list(_drug['dname'])
    del _drug
    synergy = synergy[synergy['Drug1'].isin(drugs)]
    synergy = synergy[synergy['Drug2'].isin(drugs)]

    mask = list(map(lambda x, y: x>y, synergy['Drug1'].astype(str), synergy['Drug2'].astype(str)))
    synergy.loc[mask, 'c'], synergy.loc[mask, 'Drug2'] = synergy.loc[mask, 'Drug2'], synergy.loc[mask, 'Drug1']

    merge_data = synergy[['Drug1', 'Drug2', 'Cell line']]
    merge_data = merge_data.drop_duplicates(subset = ['Drug1', 'Drug2', 'Cell line'])
    merge_data.set_index(['Drug1', 'Drug2', 'Cell line'], inplace=True)

    comb = synergy.groupby(['Drug1', 'Drug2', 'Cell line']).agg('mean')

    data = pd.merge(merge_data, comb, left_index=True, right_index=True)
    data.reset_index(inplace=True)

    data.to_csv(f'./data/{study}_drugcombs_scored.csv', index=False)

def DrugtoGraphy(drug):
    drugGraph = smiles_to_bigraph(drug, node_featurizer=node_featurizer)
    # 处理节点特征
    actual_node_feats = drugGraph.ndata.pop('h')  # v_d.ndata.pop('h')：从图v_d的节点数据中取出节点特征h，并将其从节点数据中移除。size:[节点：20,特征长度：74]
    num_actual_nodes = actual_node_feats.shape[0]  # num_actual_nodes：获取实际节点特征的数量，即图中实际节点的数量。
    virtual_node_bit = torch.zeros([num_actual_nodes, 1])  # virtual_node_bit：创建一个形状为[num_actual_nodes, 1]的全零张量，用于标识实际节点。
    actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit),1)  # torch.cat((actual_node_feats, virtual_node_bit), 1)：将实际节点特征和虚拟节点标识张量在维度 1 上拼接，更新实际节点特征。为了区别actual和virtual
    drugGraph.ndata['h'] = actual_node_feats
    return drugGraph

def get_train_test(study, ratio):

    comb = pd.read_csv(f'./data/{study}.csv')
    comb = comb[['drug_row', 'drug_col', 'cell_line_name', 'synergy_loewe']]
    comb = comb.rename(columns={'Cell line': 'cell_line_name'})
    # Outlier cleaning
    comb = comb[comb['synergy_loewe'] < 400]

    comb = comb.sample(frac=1, random_state=SEED)

    smiles = pd.read_csv('./data/drugs.csv')
    smiles = smiles[['dname', 'smiles']]
    _d = list(set(list(comb['drug_row']) + list(comb['drug_col'])))
    smiles = smiles[smiles['dname'].isin(_d)]
    smiles.set_index('dname', inplace=True)

    cells = pd.read_csv('./data/cells.csv')
    _c = list(set(comb['cell_line_name']))
    cells = cells[cells['name'].isin(_c)]
    cells.set_index('name', inplace=True)

    mark = []
    landmark = pd.read_csv('./data/raw/landmarkGene.txt', sep='\t')
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
    cells = cells[mark]

    datas = []
    datas.append(comb[0: int(len(comb) * ratio)])
    datas.append(comb[int(len(comb) * ratio):])

    d_graph = {}
    max_drug_nodes = 90

    for l, da in zip(['train', 'test'], datas):
        save_set = []

        for item in da.itertuples():
            smileA = smiles.loc[item.drug_row]['smiles']
            smileB = smiles.loc[item.drug_col]['smiles']
            cellGene = cells.loc[item.cell_line_name].values

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
            virtual_node_feat_A = torch.cat((torch.zeros(num_virtual_nodes_A, 74), torch.ones(num_virtual_nodes_A, 1)),1)  # virtual_node_feat：创建虚拟节点的特征张量，由一个形状为[num_virtual_nodes, 74]的全零张量和一个形状为[num_virtual_nodes, 1]的全一张量在维度 1 上拼接而成。为了区别actual和virtual
            gA.add_nodes(num_virtual_nodes_A, {"h": virtual_node_feat_A})

            actual_node_feats_B = gB.ndata['h']
            num_actual_nodes_B = actual_node_feats_B.shape[0]
            num_virtual_nodes_B = max_drug_nodes - num_actual_nodes_B - v_p_split.__len__()
            virtual_node_feat_B = torch.cat((torch.zeros(num_virtual_nodes_B, 74), torch.ones(num_virtual_nodes_B, 1)),1)  # virtual_node_feat：创建虚拟节点的特征张量，由一个形状为[num_virtual_nodes, 74]的全零张量和一个形状为[num_virtual_nodes, 1]的全一张量在维度 1 上拼接而成。为了区别actual和virtual
            gB.add_nodes(num_virtual_nodes_B, {"h": virtual_node_feat_B})

            for i, sub_tensor in enumerate(v_p_split):
                tensor_expanded = sub_tensor.unsqueeze(0)

                num_nodes_A = gA.number_of_nodes()
                num_nodes_B = gB.number_of_nodes()

                new_node_id_A = num_nodes_A
                new_node_id_B = num_nodes_B

                if i==0:
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

            add_src_A1 = torch.arange(pAid_first,pAid_first+11)
            add_src_B1 = torch.arange(pBid_first,pBid_first+11)
            add_dst_A1 = torch.arange(pAid_first+1,pAid_first+12)
            add_dst_B1 = torch.arange(pBid_first+1,pBid_first+12)

            add_src_A2 = torch.arange(pAid_first+1,pAid_first+12)
            add_src_B2 = torch.arange(pBid_first+1,pBid_first+12)
            add_dst_A2 = torch.arange(pAid_first,pAid_first+11)
            add_dst_B2 = torch.arange(pBid_first,pBid_first+11)

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

            #实现蛋白质的单向连接
            # src_A, dst_A = gA.edges()
            # src_B, dst_B = gB.edges()
            #
            # add_src_A1 = torch.arange(pAid_first,pAid_first+12)
            # add_src_B1 = torch.arange(pBid_first,pBid_first+12)
            # add_dst_A1 = torch.arange(pAid_first+1,pAid_first+13)
            # add_dst_B1 = torch.arange(pBid_first+1,pBid_first+13)
            #
            # new_src_A = torch.cat([src_A, add_src_A1])
            # new_dst_A = torch.cat([dst_A, add_dst_A1])
            # new_src_B = torch.cat([src_B, add_src_B1])
            # new_dst_B = torch.cat([dst_B, add_dst_B1])
            #
            # new_gA = dgl.graph((new_src_A, new_dst_A))
            # new_gB = dgl.graph((new_src_B, new_dst_B))
            #
            # new_gA.ndata['h'] = gA.ndata['h']
            # new_gB.ndata['h'] = gB.ndata['h']
            #
            # gA = new_gA
            # gB = new_gB
            #
            # gA = dgl.add_self_loop(gA)
            # gB = dgl.add_self_loop(gB)


            save_set.append(((gA, gB, cellGene), item.Loewe))

        with open(f'./data/{study}_{l}_drugcombs_scored_addproteintoDruggraphyWithProteinInterconnection.pkl', 'wb') as f:
            pickle.dump(save_set, f)



if __name__ == "__main__":
    # getDrugs()
    # getCells()
    #getDrugCombs("ONEIL")
    get_train_test("ONEIL", 0.9)