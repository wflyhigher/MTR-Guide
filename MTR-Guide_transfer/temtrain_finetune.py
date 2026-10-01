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
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from Protein_feature import init_processed_feature_dict
import copy

random_seed = 2026
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 0
torch.cuda.set_device(device)
Fold_num = 5

basedir = ["A", "T", "C", "G", "N"]
idx2base = {idx: base for idx, base in enumerate(basedir)}

PRETRAINED_MODEL_PATH = "/home/Project/CRISPR_M3_transfer/result_pretrain/pretrained_model.pt"

FINETUNE_DATASETS = ["evo", "Hypa", "sniper", "xcas9"]

dataset_dir = "/home/Project/CRISPR_M3_transfer/data_M3"
unipert_embedding_dir = "/home/Project/CRISPR_M3_transfer/unipert_embedding"
GenePT_embedding_dir = "/home/Project/CRISPR_M3_transfer/GenePT_Embedding"
result_base_dir = "/home/Project/CRISPR_M3_transfer/result_finetune"
protein_dir = '/home/Project/CRISPR_M3_transfer/data_protein'

MODULE_CONFIG = {
    'A': {'name': 'sgRNA_encoder', 'layers': ['conv1', 'conv2', 'conv3', 'conv4']},
    'B': {'name': 'gene_embedding', 'layers': ['gene_unipert_conv', 'gene_genept_conv']},
    'C': {'name': 'feature_28d', 'layers': ['feat_28d_fc1', 'feat_28d_fc2']},
    'D': {'name': 'interaction', 'layers': ['sgrna_to_32']},
    'E': {'name': 'fusion_stage1', 'layers': ['fusion_fc']},
    'F': {'name': 'protein_processing', 'layers': ['protein_fc']},
    'G': {'name': 'dynamic_weight', 'layers': ['weight_net']},
    'H': {'name': 'output', 'layers': ['stage2_fc']},
}

DATASET_CONFIG = {
    'evo': {
        'frozen_modules': ['A', 'B'],
        'lr': 0.001,
        'weight_decay': 0.05,
    },
    'Hypa': {
        'frozen_modules': ['A', 'B'],
        'lr': 0.001,
        'weight_decay': 0.05,
    },
    'sniper': {
        'frozen_modules': ['A', 'B'],
        'lr': 0.001,
        'weight_decay': 0.05,
    },
    'xcas9': {
        'frozen_modules': ['A', 'B'],
        'lr': 0.001,
        'weight_decay': 0.05,
    }
}

BATCH_SIZE = 16
USE_ENSEMBLE = True

os.makedirs(result_base_dir, exist_ok=True)
feature_dict = init_processed_feature_dict(protein_dir)


class MyDataset(Dataset):
    def __init__(self, seq, features, labels, geneids, geneproteins, cas9protein, 
                 original_indices=None):
        self.seq = seq
        self.features = features
        self.labels = labels
        self.geneids = geneids
        self.geneproteins = geneproteins
        self.cas9protein = cas9protein
        self.original_indices = original_indices if original_indices is not None else np.arange(len(seq))
        
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, index):
        return (
            torch.LongTensor(self.seq[index]),
            torch.FloatTensor(self.features[index]),
            torch.FloatTensor([self.labels[index]]),
            self.geneids[index],
            self.geneproteins[index],
            self.cas9protein[index],
            self.original_indices[index]
        )


class FreezeStrategyManager:
    
    def __init__(self, net, frozen_modules):
        self.net = net
        self.frozen_modules = frozen_modules
        self.module_map = MODULE_CONFIG
        
    def apply(self):
        
        for param in self.net.parameters():
            param.requires_grad = True
        
        frozen_names = []
        for module_key in self.frozen_modules:
            module_info = self.module_map.get(module_key, {})
            layer_names = module_info.get('layers', [])
            for name, param in self.net.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
                    frozen_names.append(name)
        
        total_params = sum(1 for _ in self.net.parameters())
        frozen_params = sum(1 for p in self.net.parameters() if not p.requires_grad)
        trainable_params = total_params - frozen_params
        
        print(f"  🔒 Frozen modules: {self.frozen_modules}")
        print(f"     ({', '.join([self.module_map[m]['name'] for m in self.frozen_modules])})")
        print(f"     Parameters: {frozen_params}/{total_params} frozen, {trainable_params}/{total_params} trainable")
        
        return self.net


def get_stratified_bins(labels, n_bins=5):
    
    try:
        bins = pd.qcut(labels, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        bins = np.digitize(labels, np.linspace(labels.min(), labels.max(), n_bins))
    return bins


def seq_to_text(seq_array):
    
    return ''.join([idx2base[int(idx)] for idx in seq_array])


def weighted_ensemble_predict(fold_models, fold_scores, test_loader, dataset):
    
    scores = np.array(fold_scores)
    weights = np.exp((scores - scores.mean()) * 10)
    weights = weights / weights.sum()
    
    print(f"    Ensemble weights: {weights.round(3)}")
    
    all_preds = []
    
    for fold_idx, model_state in enumerate(fold_models):
        net = tt.sgrna_net().to(device)
        net.load_state_dict(model_state)
        net.eval()
        
        fold_preds = []
        
        with torch.no_grad():
            for batch in test_loader:
                sgrna, features_batch, eff, geneid, geneprotein, cas9protein, indices = batch
                sgrna = sgrna.to(device)
                features_batch = features_batch.to(device)
                
                embed_unipert = torch.stack([
                    torch.from_numpy(dataset.unipert_embeddings[p]).float() 
                    for p in geneprotein
                ]).to(device)
                
                embed_genept = torch.stack([
                    torch.tensor(dataset.GenePT_embeddings[g], dtype=torch.float32) 
                    for g in geneid
                ]).to(device)
                
                protein_tensor = torch.stack([
                    feature_dict[p] for p in cas9protein
                ]).to(device)
                
                pre = net(sgrna, features_batch, embed_genept, embed_unipert, 
                         protein_tensor, train=False)
                fold_preds.append(pre.cpu().numpy())
        
        fold_preds = np.concatenate(fold_preds).flatten()
        all_preds.append(fold_preds)
    
    all_preds = np.array(all_preds)
    ensemble_preds = np.average(all_preds, axis=0, weights=weights)
    
    return ensemble_preds


def train_fold(train_loader, val_loader, dataset, config):
    
    net = torch.load(PRETRAINED_MODEL_PATH)
    net = net.to(device)
    
    frozen_modules = config.get('frozen_modules', [])
    if frozen_modules:
        manager = FreezeStrategyManager(net, frozen_modules)
        net = manager.apply()
    else:
        print(f"  🔓 Full fine-tuning (no frozen modules)")
        total_params = sum(1 for _ in net.parameters())
        trainable_params = sum(1 for p in net.parameters() if p.requires_grad)
        print(f"     Parameters: 0/{total_params} frozen, {trainable_params}/{total_params} trainable")
    
    lr = config['lr']
    wd = config['weight_decay']
    print(f"  LR: {lr}, WD: {wd}, Batch: {BATCH_SIZE}")
    
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr, 
        weight_decay=wd,
        betas=(0.9, 0.999)
    )
    
    if frozen_modules:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=5, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=2, eta_min=1e-6
        )
    
    lossf = nn.SmoothL1Loss(beta=0.5)
    
    best_val_sp = -1
    best_state = None
    patience = 20
    no_improve = 0
    
    for epoch in range(150):
        torch.cuda.empty_cache()
        
        net.train()
        train_losses = []
        for batch in train_loader:
            sgrna, features_batch, eff, geneid, geneprotein, cas9protein, indices = batch
            sgrna = sgrna.to(device)
            features_batch = features_batch.to(device)
            eff = eff.to(device)
            
            embed_unipert = torch.stack([
                torch.from_numpy(dataset.unipert_embeddings[p]).float() 
                for p in geneprotein
            ]).to(device)
            
            embed_genept = torch.stack([
                torch.tensor(dataset.GenePT_embeddings[g], dtype=torch.float32) 
                for g in geneid
            ]).to(device)
            
            protein_tensor = torch.stack([
                feature_dict[p] for p in cas9protein
            ]).to(device)
            
            pre = net(sgrna, features_batch, embed_genept, embed_unipert, 
                     protein_tensor, train=True)
            loss = lossf(pre, eff)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            train_losses.append(loss.item())
        
        scheduler.step()
        
        net.eval()
        val_preds = []
        val_truth = []
        
        with torch.no_grad():
            for batch in val_loader:
                sgrna, features_batch, eff, geneid, geneprotein, cas9protein, indices = batch
                sgrna = sgrna.to(device)
                features_batch = features_batch.to(device)
                eff = eff.to(device)
                
                embed_unipert = torch.stack([
                    torch.from_numpy(dataset.unipert_embeddings[p]).float() 
                    for p in geneprotein
                ]).to(device)
                
                embed_genept = torch.stack([
                    torch.tensor(dataset.GenePT_embeddings[g], dtype=torch.float32) 
                    for g in geneid
                ]).to(device)
                
                protein_tensor = torch.stack([
                    feature_dict[p] for p in cas9protein
                ]).to(device)
                
                pre = net(sgrna, features_batch, embed_genept, embed_unipert, 
                         protein_tensor, train=False)
                val_preds.append(pre.cpu().numpy())
                val_truth.append(eff.cpu().numpy())
        
        val_preds = np.concatenate(val_preds).flatten()
        val_truth = np.concatenate(val_truth).flatten()
        val_sp = ss.spearmanr(val_truth, val_preds)[0]
        
        if val_sp > best_val_sp:
            best_val_sp = val_sp
            best_state = copy.deepcopy(net.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  🛑 Early stopping at epoch {epoch} (best Sp: {best_val_sp:.4f})")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: Train Loss={np.mean(train_losses):.4f}, Val Sp={val_sp:.4f}")
    
    return best_state, best_val_sp


def evaluate_test_single(model_state, test_loader, dataset):
    
    net = tt.sgrna_net().to(device)
    net.load_state_dict(model_state)
    net.eval()
    
    presave, effsave = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            sgrna, features_batch, eff, geneid, geneprotein, cas9protein, indices = batch
            sgrna = sgrna.to(device)
            features_batch = features_batch.to(device)
            eff = eff.to(device)
            
            embed_unipert = torch.stack([
                torch.from_numpy(dataset.unipert_embeddings[p]).float() 
                for p in geneprotein
            ]).to(device)
            
            embed_genept = torch.stack([
                torch.tensor(dataset.GenePT_embeddings[g], dtype=torch.float32) 
                for g in geneid
            ]).to(device)
            
            protein_tensor = torch.stack([
                feature_dict[p] for p in cas9protein
            ]).to(device)
            
            pre = net(sgrna, features_batch, embed_genept, embed_unipert, 
                     protein_tensor, train=False)
            presave.append(pre.cpu().numpy())
            effsave.append(eff.cpu().numpy())
    
    presave = np.concatenate(presave).flatten()
    effsave = np.concatenate(effsave).flatten()
    spearman = ss.spearmanr(effsave, presave)[0]
    
    return presave, effsave, spearman


def load_and_align_original_data(data_path, dataset):
    
    files = [f for f in os.listdir(data_path) if f.endswith(('.txt', '.csv'))]
    
    all_data = []
    for file in files:
        file_path = os.path.join(data_path, file)
        try:
            df = pd.read_csv(file_path, sep="\t")
            all_data.append(df)
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    if not all_data:
        return None
    
    original_df = pd.concat(all_data, ignore_index=True)
    print(f"  Loaded original data: {len(original_df)} rows")
    
    valid_indices = []
    for idx, row in original_df.iterrows():
        try:
            sgrna = row['Sequence'].upper()
            eff = float(row['Value'])
            gene_protein = row['Protein_ID']
            gene_id = row['Gene_B']
            
            if (gene_protein in dataset.unipert_embeddings and 
                gene_id in dataset.GenePT_embeddings and
                len(sgrna) == 23 and 
                eff <= 1000):
                valid_indices.append(idx)
        except:
            continue
    
    aligned_df = original_df.iloc[valid_indices].copy().reset_index(drop=True)
    print(f"  After alignment: {len(aligned_df)} rows")
    
    return aligned_df


def main():
    print("="*70)
    print("FINE-TUNING STAGE (Optimal Strategy: Freeze A+B)")
    print("="*70)
    print(f"Model: {PRETRAINED_MODEL_PATH}")
    print(f"Datasets: {FINETUNE_DATASETS}")
    print(f"Strategy: Freeze A+B (sgRNA encoder + Gene embedding) for ALL datasets")
    print(f"Based on: Exploration results showing A+B optimal across all datasets")
    print(f"Config: {Fold_num}-fold, Batch={BATCH_SIZE}, Ensemble={USE_ENSEMBLE}")
    
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"❌ Model not found: {PRETRAINED_MODEL_PATH}")
        return
    
    all_results = {}
    
    for dataset_name in FINETUNE_DATASETS:
        print(f"\n{'='*70}")
        print(f"Fine-tuning on: {dataset_name}")
        print(f"{'='*70}")
        
        config = DATASET_CONFIG.get(dataset_name, {
            'frozen_modules': ['A', 'B'],
            'lr': 0.001,
            'weight_decay': 0.05,
        })
        
        print(f"Strategy: Freeze modules {config['frozen_modules']}")
        print(f"Hyperparams: LR={config['lr']}, WD={config['weight_decay']}")
        
        try:
            data_path = os.path.join(dataset_dir, dataset_name)
            unipert_embedding_path = os.path.join(unipert_embedding_dir, f"{dataset_name}.pkl")
            GenePT_embedding_path = os.path.join(GenePT_embedding_dir, f"GenePT_gene_embedding_ada_text.pkl")
            
            dataset = dl.data_loader(dic=data_path, 
                                    GenePT_embedding_path=GenePT_embedding_path, 
                                    unipert_embedding_path=unipert_embedding_path)
            
            n_samples = len(dataset.allsgrna)
            print(f"Total samples after filtering: {n_samples}")
            
            aligned_original_df = load_and_align_original_data(data_path, dataset)
            
            y_bins = get_stratified_bins(dataset.alleff, n_bins=5)
            
            train_val_idx, test_idx = train_test_split(
                range(n_samples), 
                test_size=0.2, 
                random_state=random_seed,
                stratify=y_bins
            )
            
            print(f"Train/Val: {len(train_val_idx)}, Test: {len(test_idx)}")
            
            train_val_bins = y_bins[train_val_idx]
            skf = StratifiedKFold(n_splits=Fold_num, shuffle=True, random_state=random_seed)
            
            fold_spearmans = []
            fold_models = []
            
            for fold, (fold_train_idx, fold_val_idx) in enumerate(skf.split(
                np.zeros(len(train_val_idx)), train_val_bins)):
                
                print(f"\n  Fold {fold+1}/{Fold_num}")
                
                train_idx = [train_val_idx[i] for i in fold_train_idx]
                val_idx = [train_val_idx[i] for i in fold_val_idx]
                
                train_dataset = MyDataset(
                    dataset.allsgrna[train_idx],
                    dataset.allfeature[train_idx],
                    dataset.alleff[train_idx],
                    dataset.allgeneid[train_idx],
                    dataset.allgeneprotein[train_idx],
                    dataset.allcas9protein[train_idx],
                    original_indices=np.array(train_idx)
                )
                val_dataset = MyDataset(
                    dataset.allsgrna[val_idx],
                    dataset.allfeature[val_idx],
                    dataset.alleff[val_idx],
                    dataset.allgeneid[val_idx],
                    dataset.allgeneprotein[val_idx],
                    dataset.allcas9protein[val_idx],
                    original_indices=np.array(val_idx)
                )
                
                train_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                              shuffle=True, num_workers=0)
                val_loader = Data.DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                                            shuffle=False, num_workers=0)
                
                best_state, val_sp = train_fold(train_loader, val_loader, dataset, config)
                
                fold_spearmans.append(val_sp)
                fold_models.append(best_state)
                print(f"    Val Spearman: {val_sp:.4f}")
            
            test_dataset = MyDataset(
                dataset.allsgrna[test_idx],
                dataset.allfeature[test_idx],
                dataset.alleff[test_idx],
                dataset.allgeneid[test_idx],
                dataset.allgeneprotein[test_idx],
                dataset.allcas9protein[test_idx],
                original_indices=np.array(test_idx)
            )
            test_loader = Data.DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                         shuffle=False, num_workers=0)
            
            if USE_ENSEMBLE:
                print(f"\n  Using weighted ensemble of {Fold_num} folds...")
                presave = weighted_ensemble_predict(fold_models, fold_spearmans, test_loader, dataset)
                effsave = dataset.alleff[test_idx]
                test_sp = ss.spearmanr(effsave, presave)[0]
                
                best_fold_idx = np.argmax(fold_spearmans)
                single_pred, single_truth, single_sp = evaluate_test_single(
                    fold_models[best_fold_idx], test_loader, dataset
                )
                print(f"  Single-best-fold Test Sp: {single_sp:.4f} (Fold {best_fold_idx+1})")
                print(f"  Weighted-ensemble Test Sp: {test_sp:.4f}")
                
            else:
                best_fold_idx = np.argmax(fold_spearmans)
                print(f"\n  Best fold: {best_fold_idx+1} (Spearman: {fold_spearmans[best_fold_idx]:.4f})")
                presave, effsave, test_sp = evaluate_test_single(
                    fold_models[best_fold_idx], test_loader, dataset
                )
                print(f"  Test Spearman: {test_sp:.4f}")
            
            print(f"\n  Summary:")
            print(f"    Val: {np.mean(fold_spearmans):.4f} ± {np.std(fold_spearmans):.4f}")
            print(f"    Test: {test_sp:.4f}")
            print(f"    Fold range: [{min(fold_spearmans):.4f}, {max(fold_spearmans):.4f}]")
            
            output_dir = os.path.join(result_base_dir, dataset_name)
            os.makedirs(output_dir, exist_ok=True)
            
            np.save(os.path.join(output_dir, "predictions.npy"), presave)
            np.save(os.path.join(output_dir, "ground_truth.npy"), effsave)
            
            if aligned_original_df is not None and len(aligned_original_df) == n_samples:
                test_original_df = aligned_original_df.iloc[test_idx].copy().reset_index(drop=True)
                test_original_df['Predicted_Efficiency'] = presave
                test_original_df['Absolute_Error'] = np.abs(presave - effsave)
                
                if 'Sequence' in test_original_df.columns:
                    test_original_df['Sequence_Text'] = test_original_df['Sequence'].astype(str)
                else:
                    test_seqs = [seq_to_text(dataset.allsgrna[i]) for i in test_idx]
                    test_original_df['Sequence_Text'] = test_seqs
                
                test_original_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
                print(f"  ✅ Saved full results with {len(test_original_df.columns)} columns")
            else:
                results_df = pd.DataFrame({
                    'Predicted_Efficiency': presave,
                    'Ground_Truth': effsave,
                    'Absolute_Error': np.abs(presave - effsave),
                    'Sequence_Text': [seq_to_text(dataset.allsgrna[i]) for i in test_idx]
                })
                results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
                print(f"  ⚠️  Saved prediction results only")
            
            ensemble_path = os.path.join(output_dir, "ensemble_models.pt")
            torch.save({
                'fold_models': fold_models,
                'fold_scores': fold_spearmans,
                'weights': np.exp(np.array(fold_spearmans) * 10) / 
                          np.sum(np.exp(np.array(fold_spearmans) * 10)),
                'config': config
            }, ensemble_path)
            
            all_results[dataset_name] = {
                'val_mean': np.mean(fold_spearmans),
                'val_std': np.std(fold_spearmans),
                'val_min': min(fold_spearmans),
                'val_max': max(fold_spearmans),
                'test_spearman': test_sp,
                'strategy': f"frozen_{'_'.join(config['frozen_modules'])}"
            }
            
            print(f"  ✅ Saved to: {output_dir}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    summary_path = os.path.join(result_base_dir, "finetune_summary.txt")
    with open(summary_path, "w") as f:
        f.write("FINE-TUNING SUMMARY (Optimal Strategy: Freeze A+B)\n")
        f.write("="*70 + "\n")
        f.write(f"Model: {PRETRAINED_MODEL_PATH}\n")
        f.write(f"Strategy: Freeze A+B (sgRNA encoder + Gene embedding)\n")
        f.write(f"Rationale: Exploration results showed this is optimal for all datasets\n\n")
        
        f.write(f"{'Dataset':<12} {'Val Spearman':<20} {'Test Sp':<12} {'Strategy':<20}\n")
        f.write("-"*70 + "\n")
        
        for name, res in all_results.items():
            val_str = f"{res['val_mean']:.4f} ± {res['val_std']:.4f}"
            f.write(f"{name:<12} {val_str:<20} {res['test_spearman']:<12.4f} "
                   f"{res['strategy']:<20}\n")
    
    print(f"\n{'='*70}")
    print("FINE-TUNING COMPLETED")
    print(f"Summary: {summary_path}")
    print("="*70)
    
    print("\nFinal Results:")
    print(f"{'Dataset':<12} {'Val Mean':<12} {'Val Std':<12} {'Test':<12} {'Strategy':<15}")
    print("-" * 65)
    for name, res in all_results.items():
        print(f"{name:<12} {res['val_mean']:<12.4f} {res['val_std']:<12.4f} "
              f"{res['test_spearman']:<12.4f} {res['strategy']:<15}")


if __name__ == "__main__":
    main()