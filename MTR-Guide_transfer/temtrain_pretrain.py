# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import data_loader_transfer as dl
import train_transfer as tt
import os
import pickle
import scipy.stats as ss
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from Protein_feature import init_processed_feature_dict
from collections import Counter

random_seed = 2026
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 1
torch.cuda.set_device(device)

PRETRAIN_DATASETS = ["esp_wang", "WT_wang", "HF_wang","esp_kim", "WT_kim", "HF_kim", "WT_xiang"]
TEST_DATASETS = ["evo", "Hypa", "sniper", "xcas9"]

OVERSAMPLE_CONFIG = {
    "esp_kim": 15000,
    "WT_kim": 15000,
    "HF_kim": 15000,
    "WT_xiang": 10000
}

dataset_dir = "/home/Project/CRISPR_M3_transfer/data_M3"
unipert_embedding_dir = "/home/Project/CRISPR_M3_transfer/unipert_embedding"
GenePT_embedding_dir = "/home/Project/CRISPR_M3_transfer/GenePT_Embedding"
result_base_dir = "/home/Project/CRISPR_M3_transfer/result_pretrain"
protein_dir = '/home/Project/CRISPR_M3_transfer/data_protein'

MODEL_SAVE_PATH = os.path.join(result_base_dir, "pretrained_model.pt")

os.makedirs(result_base_dir, exist_ok=True)
feature_dict = init_processed_feature_dict(protein_dir)

basedir = ["A", "T", "C", "G", "N"]
transdir = {base: idx for idx, base in enumerate(basedir)}
idx2base = {idx: base for base, idx in transdir.items()}


class MyDatasetMultiEmbed(Dataset):
    def __init__(self, seq, features, labels, geneids, geneproteins, cas9protein, 
                 dataset_indices=None, original_indices=None, is_augmented=None):
        self.seq = seq
        self.features = features
        self.labels = labels
        self.geneids = geneids
        self.geneproteins = geneproteins
        self.cas9protein = cas9protein
        self.dataset_indices = dataset_indices
        self.original_indices = original_indices
        self.is_augmented = is_augmented
        
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, index):
        items = [
            torch.LongTensor(self.seq[index]),
            torch.FloatTensor(self.features[index]),
            torch.FloatTensor([self.labels[index]]),
            self.geneids[index],
            self.geneproteins[index],
            self.cas9protein[index]
        ]
        if self.dataset_indices is not None:
            items.append(self.dataset_indices[index])
        if self.original_indices is not None:
            items.append(self.original_indices[index])
        if self.is_augmented is not None:
            items.append(self.is_augmented[index])
        return tuple(items)


def load_original_data(dataset_name):
    
    data_path = os.path.join(dataset_dir, dataset_name)
    files = [f for f in os.listdir(data_path) if f.endswith(('.txt', '.csv'))]
    
    all_records = []
    for file_gene in files:
        file_path = os.path.join(data_path, file_gene)
        try:
            df = pd.read_csv(file_path, sep="\t")
            for idx, row in df.iterrows():
                record = row.to_dict()
                record['_source_file'] = file_gene
                record['_row_idx'] = idx
                all_records.append(record)
        except Exception as e:
            print(f"  Error loading {file_gene}: {e}")
    
    return all_records


def load_raw_datasets(dataset_names):
    
    all_records = []
    all_dataset_info = []
    
    for dataset_idx, dataset_name in enumerate(dataset_names):
        print(f"\n[{dataset_idx}] Loading raw {dataset_name}...")
        
        records = load_original_data(dataset_name)
        
        unipert_embedding_path = os.path.join(unipert_embedding_dir, f"{dataset_name}.pkl")
        GenePT_embedding_path = os.path.join(GenePT_embedding_dir, f"GenePT_gene_embedding_ada_text.pkl")
        
        with open(unipert_embedding_path, 'rb') as f:
            unipert_emb = pickle.load(f)
        with open(GenePT_embedding_path, 'rb') as f:
            genept_emb = pickle.load(f)
        
        dataset = dl.data_loader(
            dic=os.path.join(dataset_dir, dataset_name),
            GenePT_embedding_path=GenePT_embedding_path,
            unipert_embedding_path=unipert_embedding_path
        )
        
        filtered_global_indices = []
        local_idx = 0
        
        for orig_idx, record in enumerate(records):
            sgrna = record.get('Sequence', '').upper()
            eff = record.get('Value', 0)
            gene_protein = record.get('Protein_ID', '')
            gene_id = record.get('Gene_B', '')
            
            is_valid = (gene_protein in unipert_emb and 
                       gene_id in genept_emb and
                       len(sgrna) == 23 and 
                       eff <= 1000)
            
            if is_valid:
                item = {
                    'seq': dataset.allsgrna[local_idx],
                    'features': dataset.allfeature[local_idx],
                    'label': dataset.alleff[local_idx],
                    'geneid': dataset.allgeneid[local_idx],
                    'geneprotein': dataset.allgeneprotein[local_idx],
                    'cas9protein': dataset.allcas9protein[local_idx],
                    'original_record': record,
                    'original_idx': orig_idx,
                    'dataset_idx': dataset_idx,
                    'dataset_name': dataset_name,
                    'local_idx': local_idx
                }
                all_records.append(item)
                filtered_global_indices.append(len(all_records) - 1)
                local_idx += 1
        
        print(f"  Valid samples: {local_idx} / {len(records)}")
        all_dataset_info.append({
            'name': dataset_name,
            'dataset_idx': dataset_idx,
            'n_total': len(records),
            'n_valid': local_idx,
            'global_indices': filtered_global_indices,
            'unipert_emb': unipert_emb,
            'genept_emb': genept_emb
        })
    
    return all_records, all_dataset_info


def split_and_oversample(all_records, all_dataset_info, oversample_config):
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    print("\n" + "="*70)
    print("Splitting and Oversampling Strategy:")
    print("  - Each dataset: 70% train, 15% val, 15% test")
    print("  - Oversample ONLY on train set")
    print("="*70)
    
    for info in all_dataset_info:
        dataset_idx = info['dataset_idx']
        name = info['name']
        global_indices = np.array(info['global_indices'])
        n_valid = len(global_indices)
        
        if n_valid == 0:
            continue
        
        train_idx, temp_idx = train_test_split(
            global_indices, test_size=0.3, random_state=random_seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=random_seed
        )
        
        print(f"\n[{name}]")
        print(f"  Original: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        if name in oversample_config:
            target_size = oversample_config[name]
            current_size = len(train_idx)
            
            if current_size < target_size:
                n_augment = target_size - current_size
                
                np.random.seed(random_seed)
                augment_indices = np.random.choice(train_idx, size=n_augment, replace=True)
                
                train_idx = np.concatenate([train_idx, augment_indices])
                
                print(f"  Oversampled train: {current_size} -> {len(train_idx)} (+{n_augment})")
                
                for i, idx in enumerate(train_idx):
                    all_records[idx]['is_augmented'] = (i >= current_size)
            else:
                for idx in train_idx:
                    all_records[idx]['is_augmented'] = False
        else:
            for idx in train_idx:
                all_records[idx]['is_augmented'] = False
        
        for idx in val_idx:
            all_records[idx]['is_augmented'] = False
        for idx in test_idx:
            all_records[idx]['is_augmented'] = False
        
        train_indices.extend(train_idx.tolist())
        val_indices.extend(val_idx.tolist())
        test_indices.extend(test_idx.tolist())
    
    return train_indices, val_indices, test_indices


def build_data_dict(all_records, indices, dataset_info_list):
    
    seq_list = []
    features_list = []
    labels_list = []
    geneids_list = []
    geneproteins_list = []
    cas9proteins_list = []
    dataset_indices_list = []
    original_indices_list = []
    is_augmented_list = []
    original_data_dict = {}
    
    for info in dataset_info_list:
        original_data_dict[info['dataset_idx']] = []
    
    for global_idx in indices:
        record = all_records[global_idx]
        
        seq_list.append(record['seq'])
        features_list.append(record['features'])
        labels_list.append(record['label'])
        geneids_list.append(record['geneid'])
        geneproteins_list.append(record['geneprotein'])
        cas9proteins_list.append(record['cas9protein'])
        dataset_indices_list.append(record['dataset_idx'])
        original_indices_list.append(record['original_idx'])
        is_augmented_list.append(record.get('is_augmented', False))
        
        dataset_idx = record['dataset_idx']
        original_data_dict[dataset_idx].append(record['original_record'])
    
    unipert_emb_dict = {}
    genept_emb_dict = {}
    for info in dataset_info_list:
        unipert_emb_dict[info['dataset_idx']] = info['unipert_emb']
        genept_emb_dict[info['dataset_idx']] = info['genept_emb']
    
    return {
        'seq': np.array(seq_list),
        'features': np.array(features_list),
        'labels': np.array(labels_list),
        'geneids': np.array(geneids_list),
        'geneproteins': np.array(geneproteins_list),
        'cas9proteins': np.array(cas9proteins_list),
        'dataset_indices': np.array(dataset_indices_list),
        'original_indices': np.array(original_indices_list),
        'is_augmented': np.array(is_augmented_list),
        'unipert_emb': unipert_emb_dict,
        'genept_emb': genept_emb_dict,
        'original_data': original_data_dict
    }


def indices_to_sequence(seq_indices):
    
    return ''.join([idx2base.get(int(idx), 'N') for idx in seq_indices])


def save_detailed_results(data, indices, predictions, ground_truth, dataset_names, output_dir, dataset_key):
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_records = []
    
    for i, global_idx in enumerate(indices):
        dataset_idx = data['dataset_indices'][global_idx]
        original_idx = data['original_indices'][global_idx]
        is_aug = data['is_augmented'][global_idx]
        
        original_records = data['original_data'][dataset_idx]
        
        record = None
        for r in original_records:
            if r.get('_row_idx') == original_idx:
                record = r.copy()
                break
        
        if record is None:
            record = {}
        
        record['Predicted'] = predictions[i]
        record['GroundTruth'] = ground_truth[i]
        record['AbsError'] = abs(predictions[i] - ground_truth[i])
        record['Dataset'] = dataset_names[dataset_idx] if dataset_idx < len(dataset_names) else 'unknown'
        record['Is_Augmented'] = bool(is_aug)
        
        if 'Sequence' not in record or pd.isna(record.get('Sequence')):
            seq_indices = data['seq'][global_idx]
            record['Sequence'] = indices_to_sequence(seq_indices)
        
        all_records.append(record)
    
    df = pd.DataFrame(all_records)
    
    priority_cols = ['Sequence', 'Predicted', 'GroundTruth', 'AbsError', 'Dataset', 'Is_Augmented', 'Value', 'Protein_ID', 'Gene_B', 'Protein']
    other_cols = [c for c in df.columns if c not in priority_cols and not c.startswith('_')]
    final_cols = [c for c in priority_cols if c in df.columns] + other_cols + [c for c in df.columns if c.startswith('_')]
    df = df[final_cols]
    
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Detailed results saved to: {csv_path}")
    
    n_augmented = df['Is_Augmented'].sum() if 'Is_Augmented' in df.columns else 0
    n_original = len(df) - n_augmented
    print(f"  Samples: {n_original} original, {n_augmented} augmented")
    
    np.save(os.path.join(output_dir, "predictions.npy"), predictions)
    np.save(os.path.join(output_dir, "ground_truth.npy"), ground_truth)
    
    return df


def safe_get_embedding(emb_dict, d_idx, key):
    
    if isinstance(d_idx, torch.Tensor):
        d_idx = d_idx.item()
    elif hasattr(d_idx, 'item'):
        d_idx = d_idx.item()
    
    return emb_dict[d_idx][key]


def train_model(train_loader, val_loader, unipert_emb, genept_emb):
    
    net = tt.sgrna_net().to(device)
    lossf = nn.MSELoss()
    opt = torch.optim.AdamW(net.parameters(), lr=0.0005, weight_decay=0.02, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=10, factor=0.5, verbose=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    no_improve_counter = 0
    best_epoch = 0
    
    for epoch in range(100):
        torch.cuda.empty_cache()
        net.train()
        train_loss = 0
        
        for batch in train_loader:
            n_items = len(batch)
            sgrna = batch[0].to(device)
            features_batch = batch[1].to(device)
            eff = batch[2].to(device)
            geneid = batch[3]
            geneprotein = batch[4]
            cas9protein = batch[5]
            dataset_idx = batch[6] if n_items > 6 else None
            
            embed_unipert_list = []
            embed_genept_list = []
            
            for d_idx, p, g in zip(dataset_idx.tolist(), geneprotein, geneid):
                embed_unipert_list.append(unipert_emb[d_idx][p])
                embed_genept_list.append(genept_emb[d_idx][g])
            
            embed_unipert = torch.FloatTensor(np.array(embed_unipert_list)).to(device)
            embed_genept = torch.FloatTensor(np.array(embed_genept_list)).to(device)
            
            protein_tensor = torch.stack([feature_dict[p] for p in cas9protein]).to(device)
            
            pre = net(sgrna, features_batch, embed_genept, embed_unipert, protein_tensor, train=True)
            loss = lossf(pre, eff)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                n_items = len(batch)
                sgrna = batch[0].to(device)
                features_batch = batch[1].to(device)
                eff = batch[2].to(device)
                geneid = batch[3]
                geneprotein = batch[4]
                cas9protein = batch[5]
                dataset_idx = batch[6] if n_items > 6 else None
                
                embed_unipert_list = []
                embed_genept_list = []
                
                for d_idx, p, g in zip(dataset_idx.tolist(), geneprotein, geneid):
                    embed_unipert_list.append(unipert_emb[d_idx][p])
                    embed_genept_list.append(genept_emb[d_idx][g])
                
                embed_unipert = torch.FloatTensor(np.array(embed_unipert_list)).to(device)
                embed_genept = torch.FloatTensor(np.array(embed_genept_list)).to(device)
                
                protein_tensor = torch.stack([feature_dict[p] for p in cas9protein]).to(device)
                
                pre = net(sgrna, features_batch, embed_genept, embed_unipert, protein_tensor, train=False)
                val_loss += lossf(pre, eff).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = net.state_dict().copy()
            no_improve_counter = 0
            best_epoch = epoch
            print(f"✅ New best Val Loss: {best_val_loss:.4f}")
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
    
    net.load_state_dict(best_model_state)
    return net, best_val_loss, best_epoch


def evaluate_model(net, data, indices, dataset_names, save_results=False, output_dir=None):
    
    dataset = MyDatasetMultiEmbed(
        data['seq'][indices], 
        data['features'][indices], 
        data['labels'][indices],
        data['geneids'][indices], 
        data['geneproteins'][indices], 
        data['cas9proteins'][indices],
        data['dataset_indices'][indices],
        data['original_indices'][indices],
        data['is_augmented'][indices]
    )
    loader = Data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    net.eval()
    presave, effsave, dataset_indices_save = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            n_items = len(batch)
            sgrna = batch[0].to(device)
            features_batch = batch[1].to(device)
            eff = batch[2].to(device)
            geneid = batch[3]
            geneprotein = batch[4]
            cas9protein = batch[5]
            dataset_idx = batch[6] if n_items > 6 else None
            
            embed_unipert_list = []
            embed_genept_list = []
            
            for d_idx, p, g in zip(dataset_idx.tolist(), geneprotein, geneid):
                embed_unipert_list.append(data['unipert_emb'][d_idx][p])
                embed_genept_list.append(data['genept_emb'][d_idx][g])
            
            embed_unipert = torch.FloatTensor(np.array(embed_unipert_list)).to(device)
            embed_genept = torch.FloatTensor(np.array(embed_genept_list)).to(device)
            
            protein_tensor = torch.stack([feature_dict[p] for p in cas9protein]).to(device)
            
            pre = net(sgrna, features_batch, embed_genept, embed_unipert, protein_tensor, train=False)
            presave.append(pre.cpu().numpy())
            effsave.append(eff.cpu().numpy())
            dataset_indices_save.append(dataset_idx.numpy())
    
    presave = np.concatenate(presave).flatten()
    effsave = np.concatenate(effsave).flatten()
    dataset_indices_save = np.concatenate(dataset_indices_save)
    
    results = {}
    overall_spearman = ss.spearmanr(effsave, presave)[0]
    results['overall'] = {'spearman': overall_spearman, 'samples': len(indices)}
    
    for idx, name in enumerate(dataset_names):
        mask = dataset_indices_save == idx
        if np.sum(mask) > 0:
            sp = ss.spearmanr(effsave[mask], presave[mask])[0]
            results[name] = {'spearman': sp, 'samples': np.sum(mask)}
    
    if save_results and output_dir is not None:
        save_detailed_results(data, indices, presave, effsave, dataset_names, output_dir, 'test')
    
    return results, presave, effsave


def main():
    print("="*70)
    print("PRETRAINING STAGE - FIXED SPLIT (No Data Leakage)")
    print("="*70)
    print(f"Pretrain datasets: {PRETRAIN_DATASETS}")
    print(f"Test datasets: {TEST_DATASETS}")
    print(f"Oversample config: {OVERSAMPLE_CONFIG}")
    
    print("\n[Step 1] Loading raw datasets...")
    all_records, all_dataset_info = load_raw_datasets(PRETRAIN_DATASETS)
    print(f"\nTotal valid samples: {len(all_records)}")
    
    print("\n[Step 2] Splitting and oversampling...")
    train_indices, val_indices, test_indices = split_and_oversample(
        all_records, all_dataset_info, OVERSAMPLE_CONFIG
    )
    
    print("\n[Step 3] Building data dictionaries...")
    train_data = build_data_dict(all_records, train_indices, all_dataset_info)
    val_data = build_data_dict(all_records, val_indices, all_dataset_info)
    test_data = build_data_dict(all_records, test_indices, all_dataset_info)
    
    print("\n" + "="*70)
    print("Final Data Statistics:")
    print(f"  Train: {len(train_indices)} samples "
          f"({sum(train_data['is_augmented'])} augmented)")
    print(f"  Val:   {len(val_indices)} samples (no augmentation)")
    print(f"  Test:  {len(test_indices)} samples (no augmentation)")
    print("="*70)
    
    train_dataset = MyDatasetMultiEmbed(
        train_data['seq'], train_data['features'], train_data['labels'],
        train_data['geneids'], train_data['geneproteins'], train_data['cas9proteins'],
        train_data['dataset_indices'], train_data['original_indices'], train_data['is_augmented']
    )
    val_dataset = MyDatasetMultiEmbed(
        val_data['seq'], val_data['features'], val_data['labels'],
        val_data['geneids'], val_data['geneproteins'], val_data['cas9proteins'],
        val_data['dataset_indices'], val_data['original_indices'], val_data['is_augmented']
    )
    
    train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    print("\n[Step 4] Training...")
    net, best_val_loss, best_epoch = train_model(train_loader, val_loader, 
                                                 train_data['unipert_emb'], 
                                                 train_data['genept_emb'])
    
    print("\n[Step 5] Evaluating on test set (NO augmentation)...")
    test_results, _, _ = evaluate_model(
        net, test_data, range(len(test_indices)), PRETRAIN_DATASETS, 
        save_results=True, output_dir=os.path.join(result_base_dir, "pretrain_test")
    )
    
    print("\nTest Set Results (Reliable):")
    for name, res in test_results.items():
        print(f"  {name}: Spearman = {res['spearman']:.4f} ({res['samples']} samples)")
    
    torch.save(net, MODEL_SAVE_PATH)
    print(f"\n💾 Model saved to: {MODEL_SAVE_PATH}")
    
    print("\n" + "="*70)
    print("ZERO-SHOT EVALUATION ON TEST DATASETS")
    print("="*70)
    
    zero_shot_results = {}
    for test_dataset in TEST_DATASETS:
        print(f"\nEvaluating on {test_dataset}...")
        try:
            test_records, test_info = load_raw_datasets([test_dataset])
            _, _, test_idx = split_and_oversample(test_records, test_info, {})
            
            test_data_dict = build_data_dict(test_records, test_idx, test_info)
            
            output_dir = os.path.join(result_base_dir, f"test_{test_dataset}")
            results, presave, effsave = evaluate_model(
                net, test_data_dict, 
                range(len(test_idx)), 
                [test_dataset],
                save_results=True,
                output_dir=output_dir
            )
            
            sp = results[test_dataset]['spearman']
            zero_shot_results[test_dataset] = sp
            print(f"  Spearman = {sp:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            zero_shot_results[test_dataset] = None
    
    summary_path = os.path.join(result_base_dir, "pretrain_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PRETRAINING SUMMARY (FIXED - No Data Leakage)\n")
        f.write("="*70 + "\n")
        f.write(f"Pretrain datasets: {PRETRAIN_DATASETS}\n")
        f.write(f"Oversample config: {OVERSAMPLE_CONFIG}\n")
        f.write(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch})\n\n")
        
        f.write("Test Set Results (No Augmentation - Reliable):\n")
        for name, res in test_results.items():
            f.write(f"  {name}: {res['spearman']:.4f}\n")
        
        f.write("\nZero-Shot Test Results:\n")
        for name, sp in zero_shot_results.items():
            if sp is not None:
                f.write(f"  {name}: {sp:.4f}\n")
            else:
                f.write(f"  {name}: Failed\n")
    
    print(f"\n✅ Summary saved to: {summary_path}")
    print("="*70)
    print("PRETRAINING COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()