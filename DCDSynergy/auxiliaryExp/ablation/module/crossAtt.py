import torch
import torch.nn as nn
import torch.nn.functional as F


class BiCrossAttention(nn.Module):
    def __init__(self, drug_dim=128, protein_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim

        # Drug->Protein attention
        self.Wq1 = nn.Linear(drug_dim, drug_dim)
        self.Wk1 = nn.Linear(protein_dim, drug_dim)
        self.Wv1 = nn.Linear(protein_dim, drug_dim)

        # Protein->Drug attention
        self.Wq2 = nn.Linear(protein_dim, protein_dim)
        self.Wk2 = nn.Linear(drug_dim, protein_dim)
        self.Wv2 = nn.Linear(drug_dim, protein_dim)

        # Fusion parameters
        self.lambda_c = nn.Parameter(torch.tensor(0.5))
        self.lambda_p = nn.Parameter(torch.tensor(0.5))

        # Output layers
        self.output_drug = nn.Linear(drug_dim, drug_dim)
        self.output_protein = nn.Linear(protein_dim, protein_dim)

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(drug_dim + protein_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V)

    def forward(self, Zc, Zp):
        # Drug -> Protein attention
        batch_size = Zc.size(0)

        # Drug to Protein
        Q1 = self.split_heads(self.Wq1(Zc), self.num_heads)  # [batch, heads, n_c, dim/heads]
        K1 = self.split_heads(self.Wk1(Zp), self.num_heads)
        V1 = self.split_heads(self.Wv1(Zp), self.num_heads)

        attn_output1 = self.scaled_dot_product_attention(Q1, K1, V1)
        attn_output1 = self.combine_heads(attn_output1)  # [batch, n_c, drug_dim]
        Fc = self.output_drug(attn_output1)

        # Protein to Drug
        Q2 = self.split_heads(self.Wq2(Zp), self.num_heads)  # [batch, heads, n_p, dim/heads]
        K2 = self.split_heads(self.Wk2(Zc), self.num_heads)
        V2 = self.split_heads(self.Wv2(Zc), self.num_heads)

        attn_output2 = self.scaled_dot_product_attention(Q2, K2, V2)
        attn_output2 = self.combine_heads(attn_output2)  # [batch, n_p, protein_dim]
        Fp = self.output_protein(attn_output2)

        # Feature fusion
        Fc = self.lambda_c * Zc + (1 - self.lambda_c) * Fc
        Fp = self.lambda_p * Zp + (1 - self.lambda_p) * Fp

        # Pooling and prediction
        drug_pool = Fc.max(dim=1)[0]  # Global max pooling
        protein_pool = Fp.max(dim=1)[0]

        combined = torch.cat([drug_pool, protein_pool], dim=1)
        # y_pred = self.mlp(combined)

        return combined

    def split_heads(self, x, num_heads):
        batch_size, seq_len, dim = x.size()
        head_dim = dim // num_heads
        return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)