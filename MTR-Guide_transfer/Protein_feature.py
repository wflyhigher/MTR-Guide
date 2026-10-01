import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

seed = 2026
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = 1

torch.cuda.set_device(device)

_processed_feature_dict = None

class GlobalMaxPool1d(nn.Module):
    """global max pooling""" 
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
         # x shape: (batch_size, channel, seq_len)
         # return shape: (batch_size, channel, 1)
        return F.max_pool1d(x, kernel_size=x.shape[2])

class TextCNN(nn.Module):
    """conv->relu->pool->dropout->linear->sigmoid"""
    def __init__(self, dropout_rate, embed_size, kernel_sizes, channel_nums):
        super(TextCNN, self).__init__()
        
        self.pool = GlobalMaxPool1d()
        self.dropout = nn.Dropout(dropout_rate)
        self.convs = nn.ModuleList()  
        for c, k in zip(channel_nums, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=embed_size, 
                                        out_channels=c, 
                                        kernel_size=k))

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len, embed_size)
        encoding = torch.cat([self.pool(F.relu(conv(inputs.permute(0, 2, 1).float()))).squeeze(-1) for conv in self.convs], dim=1)
        return encoding

class ProteinFeatureProcessor(nn.Module):
    
    def __init__(self):
        super(ProteinFeatureProcessor, self).__init__()
        # TextCNN for processing protein sequences
        self.text_cnn = TextCNN(dropout_rate=0.0, embed_size=1280, kernel_sizes=[5, 9, 13], channel_nums=[128, 128, 128])
        self.lin1 = nn.Linear(384, 256)
    
    def forward(self, protein):
        if len(protein.shape) == 2:
            protein = protein.unsqueeze(0)
        
        feature = self.text_cnn(protein.to(device))  # (N, 384)
        out_protein = F.relu(self.lin1(feature.float()))  # (N, 256)
        out_protein = torch.sigmoid(out_protein)
        return out_protein

def load_single_protein_file(file_path):
    
    loaded_tensor = torch.load(file_path)
    if 'label' in loaded_tensor:
        protein_name = loaded_tensor['label']
    else:
        protein_name = os.path.basename(file_path).replace('.pt', '')
    
    protein_features = loaded_tensor['representations']
    return protein_name, protein_features

def process_protein_directory(input_dir):
    
    pt_files = glob.glob(os.path.join(input_dir, '*.pt'))
    if not pt_files:
        raise ValueError(f"No .pt files found in directory: {input_dir}")
    
    #(f"Found {len(pt_files)} protein variant files in {input_dir}")
    
    processor = ProteinFeatureProcessor().to(device)
    processor.eval()
    
    all_variant_names = []
    all_processed_features = []
    
    for i, file_path in enumerate(pt_files):
        #print(f"Processing file {i+1}/{len(pt_files)}: {os.path.basename(file_path)}")
        
        protein_name, raw_features = load_single_protein_file(file_path)
        
        with torch.no_grad():
            processed_features = processor(raw_features.to(device)).cpu()
        
        all_variant_names.append(protein_name)
        all_processed_features.append(processed_features)
    
    all_processed_features = torch.cat(all_processed_features, dim=0)
    if all_processed_features.dim() == 3 and all_processed_features.shape[1] == 1:
        all_processed_features = all_processed_features.squeeze(1)
    
    #print(f"\nProcessing completed!")
    #print(f"Total variants processed: {len(all_variant_names)}")
    print(f"Combined features shape: {all_processed_features.shape}")
    
    return all_variant_names, all_processed_features

def get_protein_features(protein_dir, processed=False):
    
    if processed:
        return process_protein_directory(protein_dir)
    else:
        return load_raw_protein_features(protein_dir)

def load_raw_protein_features(protein_dir):
    
    pt_files = glob.glob(os.path.join(protein_dir, '*.pt'))
    if not pt_files:
        raise ValueError(f"No .pt files found in directory: {protein_dir}")
    
    all_variant_names = []
    all_raw_features = []
    
    for file_path in pt_files:
        protein_name, raw_features = load_single_protein_file(file_path)
        if raw_features.dim() == 2:
            raw_features = raw_features.unsqueeze(0)
        all_variant_names.append(protein_name)
        all_raw_features.append(raw_features)
    
    all_raw_features = torch.cat(all_raw_features, dim=0)
    return all_variant_names, all_raw_features

def create_protein_feature_dict(protein_names, protein_features):
    
    if len(protein_names) != protein_features.shape[0]:
        raise ValueError(f"Mismatch between number of names ({len(protein_names)}) and features ({protein_features.shape[0]})")
    
    feature_dict = {name: feature for name, feature in zip(protein_names, protein_features)}
    return feature_dict

def get_features_by_names(feature_dict, target_names):
    
    target_features = []
    missing_names = []
    
    for name in target_names:
        if name in feature_dict:
            target_features.append(feature_dict[name])
        else:
            missing_names.append(name)
    
    if missing_names:
        print(f"Warning: The following protein names were not found in the feature dict: {missing_names}")
    
    if not target_features:
        raise ValueError("No target features found. Please check your target names.")
    
    target_features = torch.stack(target_features)
    return target_features

def init_processed_feature_dict(protein_dir):
    
    global _processed_feature_dict
    
    #print(f"Initializing processed protein feature dictionary from: {protein_dir}")
    
    processed_names, processed_features = process_protein_directory(protein_dir)
    
    feature_dict = create_protein_feature_dict(processed_names, processed_features)
    
    _processed_feature_dict = feature_dict
    
    #(f"Created feature dictionary with {len(feature_dict)} entries")
    #print(f"Feature dimension: {next(iter(feature_dict.values())).shape[0]}")
    
    return feature_dict

def get_processed_feature_by_name(variant_name):
    
    if _processed_feature_dict is None:
        raise ValueError("Processed feature dictionary not initialized. Call init_processed_feature_dict() first.")
    
    if variant_name not in _processed_feature_dict:
        raise ValueError(f"Protein variant '{variant_name}' not found in processed features")
    
    return _processed_feature_dict[variant_name]

def get_processed_feature_dict():
    
    if _processed_feature_dict is None:
        raise ValueError("Processed feature dictionary not initialized. Call init_processed_feature_dict() first.")
    
    return _processed_feature_dict
'''
if __name__ == '__main__':
    input_dir = '/home/Project/PLM-CRISPR/data_protein'

    feature_dict = init_processed_feature_dict(input_dir)
    #print("111:",feature_dict)
    print("222:",feature_dict['evoCas9'])

    feature = get_processed_feature_by_name('evoCas9')
    #print("222:",feature)
    #print("333:",feature.shape)
'''
        
       