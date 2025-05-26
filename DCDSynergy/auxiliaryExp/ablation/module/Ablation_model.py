import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .crossAtt import BiCrossAttention
from .layers import DrugGCN, MLPDecoder, CellResidualEncoder


#No cell residual encoder
class DCDSynergy(nn.Module):
    def __init__(self, params):
        super(DCDSynergy, self).__init__()

        self.dropout = params["dropout"]

        self.drug_extractor = DrugGCN(in_feats=75, dim_embedding=128,
                                      padding=True,
                                      hidden_feats=[128, 128, 128])

        #self.protein_extractor = CellResidualEncoder(embedding_dim=128, num_filters=[128, 128, 128], kernel_size=[3, 6, 9])
        self.embedding = nn.Embedding(26, 128, padding_idx=0)

        self.crossAtt = BiCrossAttention()

        self.mlp_classifier = MLPDecoder(in_dim=512, hidden_dim=256, out_dim=128, binary=1)


    def forward(self, data):
        drug = data[0]
        cell = data[1]

        drug_emb = self.drug_extractor(drug)

        drug_emb_1 = drug_emb[:, 0, :, :]
        drug_emb_2 = drug_emb[:, 1, :, :]

        cell_Origin =  self.embedding(cell.long())

        #cell_emb = self.protein_extractor(cell_Origin)

        f1 = self.crossAtt(drug_emb_1,cell_Origin)
        f2 = self.crossAtt(drug_emb_2, cell_Origin)
        f = torch.cat((f1, f2), dim=1)
        out = self.mlp_classifier(f)

        return out

# no DrugGCN
class DCDSynergy(nn.Module):
    def __init__(self, params):
        super(DCDSynergy, self).__init__()

        self.dropout = params["dropout"]

        # self.drug_extractor = DrugGCN(in_feats=75, dim_embedding=128,
        #                               padding=True,
        #                               hidden_feats=[128, 128, 128])
        self.init_transform = nn.Linear(75, 128, bias=False)

        self.protein_extractor = CellResidualEncoder(embedding_dim=128, num_filters=[128, 128, 128], kernel_size=[3, 6, 9])
        self.embedding = nn.Embedding(26, 128, padding_idx=0)

        self.crossAtt = BiCrossAttention()

        self.mlp_classifier = MLPDecoder(in_dim=512, hidden_dim=256, out_dim=128, binary=1)


    def forward(self, data):
        drug = data[0]
        cell = data[1]
        drug_emb = self.init_transform(drug.ndata['h'])
        # drug_emb = self.drug_extractor(drug)
        batch_size = drug.batch_size
        drug_emb = drug_emb.view(batch_size // 2, 2, -1, 128)
        drug_emb_1 = drug_emb[:, 0, :, :]
        drug_emb_2 = drug_emb[:, 1, :, :]

        cell_Origin =  self.embedding(cell.long())

        cell_emb = self.protein_extractor(cell_Origin)

        f1 = self.crossAtt(drug_emb_1,cell_emb)
        f2 = self.crossAtt(drug_emb_2, cell_emb)
        f = torch.cat((f1, f2), dim=1)
        out = self.mlp_classifier(f)

        return out

# No att
class DCDSynergy(nn.Module):
    def __init__(self, params):
        super(DCDSynergy, self).__init__()

        self.dropout = params["dropout"]

        self.drug_extractor = DrugGCN(in_feats=75, dim_embedding=128,
                                      padding=True,
                                      hidden_feats=[128, 128, 128])
        self.init_transform = nn.Linear(75, 128, bias=False)

        self.protein_extractor = CellResidualEncoder(embedding_dim=128, num_filters=[128, 128, 128], kernel_size=[3, 6, 9])
        self.embedding = nn.Embedding(26, 128, padding_idx=0)

        # self.crossAtt = BiCrossAttention()

        self.mlp_classifier = MLPDecoder(in_dim=959 * 128 + 90 * 128 * 2, hidden_dim=256, out_dim=128, binary=1)


    def forward(self, data):
        drug = data[0]
        cell = data[1]

        drug_emb = self.drug_extractor(drug)

        drug_emb_1 = drug_emb[:, 0, :, :]
        drug_emb_2 = drug_emb[:, 1, :, :]

        cell_Origin =  self.embedding(cell.long())

        cell_emb = self.protein_extractor(cell_Origin)

        # f1 = self.crossAtt(drug_emb_1,cell_emb)
        # f2 = self.crossAtt(drug_emb_2, cell_emb)
        cell_flattened = cell_emb.reshape(64, -1)
        drug_flattened1 = drug_emb_1.reshape(64, -1)
        drug_flattened2 = drug_emb_2.reshape(64, -1)

        f = torch.cat((drug_flattened1, cell_flattened, drug_flattened2), dim=1)

        out = self.mlp_classifier(f)

        return out