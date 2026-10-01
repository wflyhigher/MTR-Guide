import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.data as Data
import os
import pandas as pd
from sklearn.model_selection import  KFold
from sklearn.preprocessing import StandardScaler
import math
import pickle

device = 0
torch.cuda.set_device(device)

basedir = ["A", "T", "C", "G", "N"]
transdir = {base: idx for idx, base in enumerate(basedir)}

FEATURE_COLUMNS = [
    "has_polyT", "has_polyA", "start_with_G", "end_with_G",
    "has_palindrome", "has_self_complement", "has_triplet_repeat", "has_mononucleotide_repeat",
    "GC_total", "GC_1_5", "GC_6_10", "GC_11_15", "GC_16_20",
    "Tm_total", "Tm_1_5", "Tm_6_10", "Tm_11_15", "Tm_16_20",
    "MFE_total", "MFE_1_5", "MFE_6_10", "MFE_11_15", "MFE_16_20",
    "is_high_fidelity", "is_ultra_high_fidelity", "has_rec2_mut", "has_rec3_mut", "has_hnh_mut"
]

CONTINUOUS_FEATURES = [
    "GC_total", "GC_1_5", "GC_6_10", "GC_11_15", "GC_16_20",
    "Tm_total", "Tm_1_5", "Tm_6_10", "Tm_11_15", "Tm_16_20",
    "MFE_total", "MFE_1_5", "MFE_6_10", "MFE_11_15", "MFE_16_20"
]
BINARY_FEATURES = [col for col in FEATURE_COLUMNS if col not in CONTINUOUS_FEATURES]

class data_loader(Dataset):
    def __init__(self, dic, GenePT_embedding_path,unipert_embedding_path):
        super().__init__()
        self.GenePT_embeddings = pickle.load(open(GenePT_embedding_path, 'rb'))
        self.unipert_embeddings = pickle.load(open(unipert_embedding_path, 'rb'))
        
        self.allsgrna = []
        self.alleff = []
        self.allgeneid = []
        self.allgeneprotein = []
        self.allcas9protein = []
        self.all_features = []
        self.dataset_name = os.path.basename(dic)
        
        print(f"Loading dataset from: {dic}")
        
        if not os.path.exists(dic):
            print(f"Warning: Directory not found: {dic}")
            return
        
        files = [f for f in os.listdir(dic) if f.endswith(('.txt', '.csv'))]
        
        if not files:
            print(f"Warning: No data files found in: {dic}")
            return
        
        print(f"Found {len(files)} data files")
        
        temp_features = []
        
        for file_gene in files:
            file_path = os.path.join(dic, file_gene)
            print(f"Processing file: {file_gene}")
            try:
                data = pd.read_csv(file_path, sep="\t")
                print(f"  Rows in file: {len(data)}")
                
                missing_cols = [col for col in FEATURE_COLUMNS if col not in data.columns]
                if missing_cols:
                    print(f"  Warning: Missing feature columns: {missing_cols}")
                    continue
                
                for idx, row in data.iterrows():
                    try:
                        sgrna = row['Sequence'].upper() 
                        eff = float(row['Value'])  
                        gene_protein = row['Protein_ID'] 
                        gene_id = row['Gene_B'] 
                        Cas9_protein_name = row['Protein']
                        features = [row[col] for col in FEATURE_COLUMNS]
                    except Exception as e:
                        #print(f"  Error parsing row {idx}: {e}")
                        continue
                    
                    if (gene_protein in self.unipert_embeddings and 
                        gene_id in self.GenePT_embeddings and
                        len(sgrna) == 23 and 
                        eff <= 1000):
                        line = [transdir[base] for base in sgrna]
                        self.allsgrna.append(line)
                        self.alleff.append(eff)
                        self.allgeneid.append(gene_id)
                        self.allgeneprotein.append(gene_protein)
                        self.allcas9protein.append(Cas9_protein_name)
                        temp_features.append(features)
                        
            except Exception as e:
                print(f"  Error processing file {file_gene}: {e}")
                continue

        if temp_features:
            features_df = pd.DataFrame(temp_features, columns=FEATURE_COLUMNS)
            
            features_df = features_df.replace(-1, np.nan)
            features_df = features_df.replace(-1.0, np.nan)
            
            for col in BINARY_FEATURES:
                features_df[col] = features_df[col].fillna(features_df[col].mode()[0])
            for col in CONTINUOUS_FEATURES:
                features_df[col] = features_df[col].fillna(features_df[col].mean())
            
            scaler = StandardScaler()
            features_df[CONTINUOUS_FEATURES] = scaler.fit_transform(features_df[CONTINUOUS_FEATURES])
            
            self.all_features = features_df.values.astype(np.float32)
            self.feature_scaler = scaler
        else:
            self.all_features = np.array([])
            self.feature_scaler = None

        self.allsgrna = np.array(self.allsgrna).astype(np.float16)
        self.alleff = np.array(self.alleff).astype(np.float16)
        self.allgeneid = np.array(self.allgeneid).astype(str)
        self.allgeneprotein = np.array(self.allgeneprotein).astype(str)
        self.allcas9protein = np.array(self.allcas9protein).astype(str)
        
        print("After filtering, allsgrna", len(self.allsgrna), 
              "alleff", len(self.alleff), 
              "allgeneid", len(self.allgeneid), 
              "allgeneprotein", len(self.allgeneprotein), 
              "allcas9protein", len(self.allcas9protein),
              "all_features", len(self.all_features))

    def __len__(self):
        return len(self.allsgrna)

    def __getitem__(self, index):
        return (torch.LongTensor(self.allsgrna[index]),
                torch.FloatTensor(self.all_features[index]),
                torch.FloatTensor([self.alleff[index]]),
                self.allgeneid[index],
                self.allgeneprotein[index],
                self.allcas9protein[index])


def cross_validation(data, features, labels, geneids, geneproteins, Cas9_protein_names, 
                     n_splits, random_seed, batch_size=256, num_workers=4, worker_init_fn=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_loaders = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        train_data, train_features, train_labels = data[train_idx], features[train_idx], labels[train_idx]
        train_geneids, train_geneproteins, train_Cas9_protein_names = geneids[train_idx], geneproteins[train_idx], Cas9_protein_names[train_idx]
        
        val_data, val_features, val_labels = data[val_idx], features[val_idx], labels[val_idx]
        val_geneids, val_geneproteins, val_Cas9_protein_names = geneids[val_idx], geneproteins[val_idx], Cas9_protein_names[val_idx]
        
        train_dataset = MyDataset(train_data, train_features, train_labels, train_geneids, train_geneproteins, train_Cas9_protein_names)
        val_dataset = MyDataset(val_data, val_features, val_labels, val_geneids, val_geneproteins, val_Cas9_protein_names)
        
        train_loader = Data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            worker_init_fn=worker_init_fn
        )
        val_loader = Data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            worker_init_fn=worker_init_fn
        )
        
        fold_loaders.append((train_loader, val_loader))
    return fold_loaders

class MyDataset(Dataset):
    def __init__(self, seq, features, labels, geneids, geneproteins, Cas9_protein_names):
        self.seq = seq
        self.features = features
        self.labels = labels
        self.geneids = geneids
        self.geneproteins = geneproteins
        self.Cas9_protein_names = Cas9_protein_names

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        return (torch.LongTensor(self.seq[index]),
                torch.FloatTensor(self.features[index]),
                torch.FloatTensor([self.labels[index]]),
                self.geneids[index],
                self.geneproteins[index],
                self.Cas9_protein_names[index])