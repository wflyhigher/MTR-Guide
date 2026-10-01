import torch 
import torch.nn as nn 
import torch.nn.functional as F

device = 0
torch.cuda.set_device(device)

class sgrna_net(nn.Module):
    def __init__(self, type=11, protein_dim=256):
        super().__init__()
        
        self.conv1 = nn.Conv1d(5, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, 5, padding=2)
        self.conv4 = nn.Conv1d(5, 64, 5, padding=2)

        self.gene_unipert_conv = nn.Conv1d(256, 32, 3, padding=1)
        self.gene_genept_conv = nn.Conv1d(1536, 32, 3, padding=1)
        
        self.feat_28d_fc1 = nn.Linear(28, 256)
        self.feat_28d_fc2 = nn.Linear(256, 512)
        
        self.sgrna_to_32 = nn.Sequential(
            nn.Linear(2944, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 32)
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(2944+32+32+512, 2048),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2048, 512), 
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),  
            
        )

        self.protein_fc = nn.Sequential(
            nn.Linear(protein_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.weight_net = nn.Sequential(
            nn.Linear(256 + 256, 128),  # [sgrna_256; protein_256]
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        self.stage2_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, seq, features, embed_genept, embed_unipert, protein, train=True):
        N = seq.size(0)
        
        feat_28d = torch.relu(self.feat_28d_fc1(features))
        if train: feat_28d = F.dropout(feat_28d, p=0.5)
        feat_28d = torch.relu(self.feat_28d_fc2(feat_28d))
        if train: feat_28d = F.dropout(feat_28d, p=0.5)

        seq_sgRNA = F.one_hot(seq, 5).float().view(N, 5, 23)
        out = seq_sgRNA.float()
        be = self.conv4(out)
        si = self.conv1(out)
        out = torch.relu(si)
        if train: out = F.dropout(out, p=0.5)
        out = self.conv2(out)
        out = torch.relu(out)
        if train: out = F.dropout(out, p=0.5)
        out = self.conv3(out)
        out = torch.relu(out)
        if train: out = F.dropout(out, p=0.5)
        
        out = out.view(N, -1)
        be4 = be.view(N, -1)
        sgrna_feat = torch.cat((out, be4), dim=-1)

        unipert_32 = self.gene_unipert_conv(embed_unipert.unsqueeze(-1)).view(N, -1)
        unipert_32 = torch.relu(unipert_32)
        if train: unipert_32 = F.dropout(unipert_32, p=0.5)

        genept_32 = self.gene_genept_conv(embed_genept.unsqueeze(-1)).view(N, -1)
        genept_32 = torch.relu(genept_32)
        if train: genept_32 = F.dropout(genept_32, p=0.5)
        
        sgrna_32 = self.sgrna_to_32(sgrna_feat)
        sgrna_32 = torch.relu(sgrna_32)
        if train: sgrna_32 = F.dropout(sgrna_32, p=0.5)
        interact_g = sgrna_32 * genept_32
        interact_u = sgrna_32 * unipert_32

        combined = torch.cat([sgrna_feat, interact_g, interact_u, feat_28d], dim=-1)

        out_fusion=self.fusion_fc(combined)

        # 5. Protein → 256
        out_protein = self.protein_fc(protein)  # (N, 256)

        combined = torch.cat([out_fusion, out_protein], dim=-1)  # (N, 512)
        weights = self.weight_net(combined)  # (N, 2)
        
        weight_sgrna = weights[:, 0:1]   # (N, 1)
        weight_protein = weights[:, 1:2]  # (N, 1)
        
        fused = weight_sgrna * out_fusion + weight_protein * out_protein  # (N, 256)

        return self.stage2_fc(fused)