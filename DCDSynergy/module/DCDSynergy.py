import torch
import torch.nn as nn
from .crossAtt import BiCrossAttention
from .layers import DrugGCN, MLPDecoder, CellResidualEncoder

class DCDSynergy(nn.Module):
    def __init__(self, params):
        super(DCDSynergy, self).__init__()

        self.dropout = params["dropout"]

        self.drug_extractor = DrugGCN(in_feats=75, dim_embedding=128,
                                      padding=True,
                                      hidden_feats=[128, 128, 128])

        self.cell_extractor = CellResidualEncoder(embedding_dim=128, num_filters=[128, 128, 128], kernel_size=[3, 6, 9])

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

        cell_emb = self.cell_extractor(cell_Origin)

        f1, att_maps1 = self.crossAtt(drug_emb_1, cell_emb)
        f2, att_maps2 = self.crossAtt(drug_emb_2, cell_emb)
        att_maps = att_maps1 + att_maps2
        f = torch.cat((f1, f2), dim=1)
        out = self.mlp_classifier(f)

        return out, att_maps
