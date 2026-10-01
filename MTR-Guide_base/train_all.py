import torch 
import torch.nn as nn 
import torch.nn.functional as F

device = 0
if torch.cuda.is_available():
    torch.cuda.set_device(device)

# ====================0.6718 ====================
class sgrna_net(nn.Module):
    def __init__(self, type=11):
        super().__init__()
        self.conv1 = nn.Conv1d(5, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, 5, padding=2)
        self.conv4 = nn.Conv1d(5, 64, 5, padding=2)

        self.gene_unipert_conv = nn.Conv1d(256, 32, 3, padding=1)
        self.gene_genept_conv = nn.Conv1d(1536, 32, 3, padding=1)
        
        self.sgrna_to_32 = nn.Sequential(
            nn.Linear(2944, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 32)
        )

        self.feat_28d_fc1 = nn.Linear(28, 256)
        self.feat_28d_fc2 = nn.Linear(256, 512)

        self.fusion_fc = nn.Sequential(
            nn.Linear(3520, 2048),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2048, 512), 
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128),  
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, seq, features, embed_genept, embed_unipert, train=True):
        N = seq.size(0)
        
        feat_28d = torch.relu(self.feat_28d_fc1(features))  # 28→256
        if train: feat_28d = F.dropout(feat_28d, p=0.5)
        feat_28d = torch.relu(self.feat_28d_fc2(feat_28d))   # 256→512
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
        return self.fusion_fc(combined)

