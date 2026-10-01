# -*- coding: utf-8 -*-
import numpy as np
import torch 
import torch.cuda
import torch.nn as nn 
import torch.utils.data as Data
import data_loader_allM3 as dl
import train_all_nogenept as tt
import os
import pickle
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold, KFold
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

random_seed = 2026
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

basedir = ["A", "T", "C", "G", "N"]
transdir = {base: idx for idx, base in enumerate(basedir)}
reverse_transdir = {idx: base for idx, base in enumerate(basedir)}

device = 1
torch.cuda.set_device(device)

class AdaptiveConfig:
    
    
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.config = self._compute_config()
        
    def _compute_config(self):
        
        config = {}
        
        if self.n_samples < 500:
            config['n_folds'] = 5
            config['min_samples_per_fold'] = self.n_samples // 5
        elif self.n_samples < 2000:
            config['n_folds'] = 8
            config['min_samples_per_fold'] = self.n_samples // 8
        else:
            config['n_folds'] = 10
            config['min_samples_per_fold'] = self.n_samples // 10
        
        if self.n_samples < 500:
            config['batch_size'] = 32
        elif self.n_samples < 3000:
            config['batch_size'] = 128
        else:
            config['batch_size'] = 256
        
        max_bins = min(10, self.n_samples // 20)
        config['n_bins'] = max(3, max_bins)
        
        if self.n_samples < 500:
            config['max_epochs'] = 50
            config['early_stop_patience'] = 15
            config['lr_patience'] = 10
        elif self.n_samples < 3000:
            config['max_epochs'] = 100
            config['early_stop_patience'] = 20
            config['lr_patience'] = 15
        else:
            config['max_epochs'] = 150
            config['early_stop_patience'] = 25
            config['lr_patience'] = 20
        
        if self.n_samples < 500:
            config['lr'] = 0.001
        elif self.n_samples < 3000:
            config['lr'] = 0.0005
        else:
            config['lr'] = 0.0003
        
        if self.n_samples < 200:
            config['num_workers'] = 0
        elif self.n_samples < 2000:
            config['num_workers'] = 2
        else:
            config['num_workers'] = 4
        
        config['weight_decay'] = 1e-4 if self.n_samples > 1000 else 1e-3
        
        config['use_stratification'] = self.n_samples >= 200
        
        return config
    
    def get(self, key):
        return self.config.get(key)
    
    def print_config(self):
        print(f"\n{'='*60}")
        print(f"Adaptive Configuration for n_samples={self.n_samples}")
        print(f"{'='*60}")
        for k, v in self.config.items():
            print(f"  {k:<25}: {v}")
        print(f"{'='*60}\n")

class FoldDiagnostics:
    def __init__(self, output_dir, low_perf_threshold=0.5):
        self.output_dir = output_dir
        self.fold_stats = {}
        self.low_perf_threshold = low_perf_threshold
        
    def analyze_fold_distribution(self, fold, train_idx, val_idx, seq, labels, 
                                   geneids, geneproteins, features, spearman=None):
        stats = {
            'fold': fold,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_eff_mean': float(np.mean(labels[train_idx])),
            'train_eff_std': float(np.std(labels[train_idx])),
            'val_eff_mean': float(np.mean(labels[val_idx])),
            'val_eff_std': float(np.std(labels[val_idx])),
            'spearman': spearman,
            'unique_genes_train': len(np.unique(geneids[train_idx])),
            'unique_genes_val': len(np.unique(geneids[val_idx])),
        }
        
        for i in range(min(3, features.shape[1])):
            stats[f'feature_{i}_train_mean'] = float(np.mean(features[train_idx, i]))
            stats[f'feature_{i}_val_mean'] = float(np.mean(features[val_idx, i]))
        
        self.fold_stats[fold] = stats
        return stats
    
    def identify_low_perf_patterns(self):
        if not self.fold_stats:
            return {}
        
        high_perf = {k: v for k, v in self.fold_stats.items() 
                    if v['spearman'] and v['spearman'] >= self.low_perf_threshold}
        low_perf = {k: v for k, v in self.fold_stats.items() 
                   if v['spearman'] and v['spearman'] < self.low_perf_threshold}
        
        analysis = {
            'low_perf_folds': list(low_perf.keys()),
            'high_perf_folds': list(high_perf.keys()),
            'low_perf_spearmans': [v['spearman'] for v in low_perf.values()],
            'high_perf_spearmans': [v['spearman'] for v in high_perf.values()],
        }
        
        if low_perf and high_perf:
            analysis['differences'] = {}
            for key in ['val_eff_mean', 'val_eff_std', 'val_size', 'unique_genes_val']:
                low_vals = [v[key] for v in low_perf.values() if key in v]
                high_vals = [v[key] for v in high_perf.values() if key in v]
                if low_vals and high_vals:
                    analysis['differences'][key] = {
                        'low_mean': np.mean(low_vals),
                        'high_mean': np.mean(high_vals),
                        'diff': np.mean(high_vals) - np.mean(low_vals)
                    }
        
        return analysis
    
    def plot_fold_analysis(self):
        if not self.fold_stats:
            return {}
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            folds = sorted(self.fold_stats.keys())
            spearmans = [self.fold_stats[f]['spearman'] for f in folds if self.fold_stats[f]['spearman'] is not None]
            
            if len(spearmans) == 0:
                return {}
            
            colors = ['red' if s < self.low_perf_threshold else 'green' for s in spearmans]
            
            ax = axes[0, 0]
            ax.bar(folds[:len(spearmans)], spearmans, color=colors, alpha=0.7)
            ax.axhline(y=np.mean(spearmans), color='blue', linestyle='--')
            ax.set_xlabel('Fold')
            ax.set_ylabel('Spearman')
            ax.set_title('Performance per Fold')
            
            ax = axes[0, 1]
            val_sizes = [self.fold_stats[f]['val_size'] for f in folds[:len(spearmans)]]
            ax.scatter(val_sizes, spearmans, c=colors, s=100, alpha=0.7)
            ax.set_xlabel('Validation Set Size')
            ax.set_ylabel('Spearman')
            
            ax = axes[0, 2]
            val_means = [self.fold_stats[f]['val_eff_mean'] for f in folds[:len(spearmans)]]
            ax.scatter(val_means, spearmans, c=colors, s=100, alpha=0.7)
            ax.set_xlabel('Validation Efficiency Mean')
            ax.set_ylabel('Spearman')
            
            ax = axes[1, 0]
            val_stds = [self.fold_stats[f]['val_eff_std'] for f in folds[:len(spearmans)]]
            ax.scatter(val_stds, spearmans, c=colors, s=100, alpha=0.7)
            ax.set_xlabel('Validation Efficiency Std')
            ax.set_ylabel('Spearman')
            
            ax = axes[1, 1]
            val_genes = [self.fold_stats[f]['unique_genes_val'] for f in folds[:len(spearmans)]]
            ax.scatter(val_genes, spearmans, c=colors, s=100, alpha=0.7)
            ax.set_xlabel('Unique Genes in Val Set')
            ax.set_ylabel('Spearman')
            
            ax = axes[1, 2]
            feat_means = [self.fold_stats[f].get('feature_0_val_mean', 0) for f in folds[:len(spearmans)]]
            ax.scatter(feat_means, spearmans, c=colors, s=100, alpha=0.7)
            ax.set_xlabel('Feature 0 (GC) Mean in Val')
            ax.set_ylabel('Spearman')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'fold_diagnostics.png'), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate diagnostic plots: {e}")
        
        with open(os.path.join(self.output_dir, 'fold_statistics.json'), 'w') as f:
            json.dump(self.fold_stats, f, indent=2)
        
        low_perf_analysis = self.identify_low_perf_patterns()
        with open(os.path.join(self.output_dir, 'low_performance_analysis.json'), 'w') as f:
            json.dump(low_perf_analysis, f, indent=2)
        
        return low_perf_analysis

def create_stratification_labels(labels, geneids, n_bins, use_stratification=True):
    
    if not use_stratification or len(labels) < 200:
        print("Stratification disabled (too few samples)")
        return None
    
    min_per_bin = len(labels) // n_bins
    if min_per_bin < 10:
        n_bins = max(3, len(labels) // 20)
        print(f"Adjusted bins to {n_bins} (min per bin: {len(labels)//n_bins})")
    
    try:
        efficiency_bins = pd.qcut(labels, q=n_bins, labels=False, duplicates='drop')
        unique_bins = len(np.unique(efficiency_bins))
        
        if unique_bins < 2:
            print("Warning: Could not create meaningful bins, using random stratification")
            return np.random.randint(0, min(n_bins, 3), size=len(labels))
        
        print(f"Stratification: {unique_bins} unique bins from {n_bins} requested")
        return efficiency_bins
        
    except Exception as e:
        print(f"Stratification failed ({e}), using random groups")
        return np.random.randint(0, min(3, n_bins), size=len(labels))

def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)

dataset_dir = "/home/Project/MTR-Guide/Data_Embedding/data_esp_kim"
unipert_embedding_dir = "/home/Project/MTR-Guide/UniPert_Embedding"
result_base_dir = "/home/Project/MTR-Guide/result/result_single_esp_kim_nogenept"
tensorboard_base_dir = "/home/Project/MTR-Guide/result/result_single_esp_kim_nogenept"
GenePT_embedding_dir ="/home/Project/MTR-Guide/GenePT_Embedding"

datasets = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
print(f"Found {len(datasets)} datasets: {datasets}")

dataset_results = {}

for dataset_name in datasets:
    print(f"\n{'='*70}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*70}")
    
    data_path = os.path.join(dataset_dir, dataset_name)
    unipert_embedding_path = os.path.join(unipert_embedding_dir, f"{dataset_name}.pkl")
    GenePT_embedding_path = os.path.join(GenePT_embedding_dir, f"GenePT_gene_embedding_ada_text.pkl")
    
    output_subdir = os.path.join(result_base_dir, f"OUTPUT_{dataset_name}")
    model_dir = os.path.join(output_subdir, "model")
    output_dir = os.path.join(output_subdir, "test_valj")
    traj_dir = os.path.join(tensorboard_base_dir, f"OUTPUT_{dataset_name}/traj")
    tensorboard_dir = os.path.join(tensorboard_base_dir, f"OUTPUT_{dataset_name}/log")
    diag_dir = os.path.join(output_subdir, "diagnostics")
    
    for d in [output_subdir, model_dir, output_dir, traj_dir, tensorboard_dir, diag_dir]:
        os.makedirs(d, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset directory not found: {data_path}")
        continue
    
    if not os.path.exists(unipert_embedding_path):
        print(f"Error: Embedding file not found: {unipert_embedding_path}")
        continue
    
    try:
        print(f"Loading data from: {data_path}")
        dataset = dl.data_loader(dic=data_path, GenePT_embedding_path=GenePT_embedding_path, 
                                unipert_embedding_path=unipert_embedding_path)
        seq = dataset.allsgrna
        labels = dataset.alleff
        geneids = dataset.allgeneid
        geneproteins = dataset.allgeneprotein
        cas9proteins = dataset.allcas9protein
        features = dataset.all_features
        
        n_samples = len(seq)
        print(f"Loaded {n_samples} samples, 28D features shape: {features.shape}")
        
        if n_samples == 0:
            continue
        
        config = AdaptiveConfig(n_samples)
        config.print_config()
        
        Fold_num = config.get('n_folds')
        batch_size = config.get('batch_size')
        n_bins = config.get('n_bins')
        max_epochs = config.get('max_epochs')
        early_stop_patience = config.get('early_stop_patience')
        lr_patience = config.get('lr_patience')
        lr = config.get('lr')
        num_workers = config.get('num_workers')
        weight_decay = config.get('weight_decay')
        use_stratification = config.get('use_stratification')
        
        diagnostics = FoldDiagnostics(diag_dir)
        writer = SummaryWriter(tensorboard_dir)
        lossf = nn.MSELoss()
        
        stratify_labels = create_stratification_labels(labels, geneids, n_bins, use_stratification)
        
        if stratify_labels is not None:
            print(f"Using StratifiedKFold with {Fold_num} folds")
            kf = StratifiedKFold(n_splits=Fold_num, shuffle=True, random_state=random_seed)
            split_iterator = kf.split(seq, stratify_labels)
        else:
            print(f"Using standard KFold with {Fold_num} folds")
            kf = KFold(n_splits=Fold_num, shuffle=True, random_state=random_seed)
            split_iterator = kf.split(seq)
        
        fold_indices = []
        for fold, (train_idx, val_idx) in enumerate(split_iterator):
            fold_indices.append((train_idx, val_idx))
            
            fold_info = {
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'random_seed': random_seed
            }
            
            if stratify_labels is not None:
                fold_info['stratify_distribution'] = {
                    'train': np.bincount(stratify_labels[train_idx], minlength=n_bins).tolist(),
                    'val': np.bincount(stratify_labels[val_idx], minlength=n_bins).tolist()
                }
            
            fold_info_path = os.path.join(output_dir, f"fold{fold + 1}_indices.json")
            with open(fold_info_path, 'w') as f:
                json.dump(fold_info, f, indent=2)
            
            print(f"Fold {fold+1}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            train_data, train_labels = seq[train_idx], labels[train_idx]
            val_data, val_labels = seq[val_idx], labels[val_idx]
            train_geneproteins = geneproteins[train_idx]
            val_geneproteins = geneproteins[val_idx]
            train_cas9proteins = cas9proteins[train_idx]
            val_cas9proteins = cas9proteins[val_idx]
            train_geneids, val_geneids = geneids[train_idx], geneids[val_idx]
            train_features = features[train_idx]
            val_features = features[val_idx]
            
            train_dataset = dl.MyDataset(train_data, train_features, train_labels, 
                                        train_geneids, train_geneproteins, train_cas9proteins)
            val_dataset = dl.MyDataset(val_data, val_features, val_labels, 
                                      val_geneids, val_geneproteins, val_cas9proteins)
            
            train_loader = Data.DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                pin_memory=True if torch.cuda.is_available() else False
            )
            val_loader = Data.DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            folds.append((train_loader, val_loader, train_idx, val_idx))
        
        all_spearman_values = []
        
        for fold, (train_loader, val_loader, train_idx, val_idx) in enumerate(folds):
            print(f"\nFold: {fold + 1}/{Fold_num}")
            
            net = tt.sgrna_net()
            net = net.to(device)
            
            opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, net.parameters()), 
                lr=lr, 
                weight_decay=weight_decay
            )
            she = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.5, patience=lr_patience, 
                verbose=True, min_lr=1e-7
            )

            wt1 = open(f"{traj_dir}/trloss_fold{fold + 1}", "w+")
            wt2 = open(f"{traj_dir}/valoss_fold{fold + 1}", "w+")
            
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            for epoch in range(max_epochs):
                net.train()
                epoch_losses = []
                
                for sgrna, features_batch, eff, geneid, geneprotein, cas9protein in train_loader:
                    sgrna = sgrna.to(device) 
                    eff = eff.to(device)
                    features_batch = features_batch.to(device)
                    
                    embed_unipert = []
                    embed_genept = []
                    for p in geneprotein:
                        embed_unipert.append(dataset.unipert_embeddings[p])
                    for g in geneid:
                        embed_genept.append(dataset.GenePT_embeddings[g])
                    
                    embed_unipert = torch.FloatTensor(np.array(embed_unipert)).to(device)
                    embed_genept = torch.FloatTensor(np.array(embed_genept)).to(device)
                    
                    pre = net(sgrna, features_batch, embed_genept, embed_unipert, train=True)
                    loss = lossf(pre, eff)

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    opt.step()
                    
                    epoch_losses.append(loss.item())
                    
                    if n_samples > 10000 and len(epoch_losses) % 100 == 0:
                        torch.cuda.empty_cache()
                
                avg_train_loss = np.mean(epoch_losses)
                writer.add_scalar(f'train/loss_fold{fold + 1}', avg_train_loss, epoch)
                print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}", 
                      file=wt1, flush=True)

                net.eval()
                val_loss = 0
                with torch.no_grad():
                    for sgrna, features_batch, eff, geneid, geneprotein, cas9protein in val_loader:
                        sgrna = sgrna.to(device) 
                        eff = eff.to(device)
                        features_batch = features_batch.to(device)
                        
                        embed_unipert = []
                        embed_genept = []
                        for p in geneprotein:
                            embed_unipert.append(dataset.unipert_embeddings[p])
                        for g in geneid:
                            embed_genept.append(dataset.GenePT_embeddings[g])
                        
                        embed_unipert = torch.FloatTensor(np.array(embed_unipert)).to(device)
                        embed_genept = torch.FloatTensor(np.array(embed_genept)).to(device)

                        pre = net(sgrna, features_batch, embed_genept, embed_unipert, train=False)
                        val_loss += lossf(pre, eff).item()
                
                avg_val_loss = val_loss / len(val_loader)
                writer.add_scalar(f'val/loss_fold{fold + 1}', avg_val_loss, epoch)
                print(f"Epoch {epoch + 1}/{max_epochs}, Val Loss: {avg_val_loss:.4f}", 
                      file=wt2, flush=True)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = net.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        print(f"Early stopping at epoch {epoch}")
                        net.load_state_dict(best_model_state)
                        break

                she.step(avg_val_loss)

            torch.cuda.empty_cache()
            save_path = os.path.join(model_dir, f"fold{fold + 1}.pt")
            torch.save(net, save_path)

            net.eval()
            presave, effsave = [], []
            
            with torch.no_grad():
                for sgrna, features_batch, eff, geneid, geneprotein, cas9protein in val_loader:
                    sgrna, eff = sgrna.to(device), eff.to(device)
                    features_batch = features_batch.to(device)
                    
                    embed_unipert = [dataset.unipert_embeddings[p] for p in geneprotein]
                    embed_genept = [dataset.GenePT_embeddings[g] for g in geneid]
                    embed_unipert = torch.FloatTensor(np.array(embed_unipert)).to(device)
                    embed_genept = torch.FloatTensor(np.array(embed_genept)).to(device)

                    pre = net(sgrna, features_batch, embed_genept, embed_unipert, train=False)
                    presave.append(pre.detach().cpu().numpy())
                    effsave.append(eff.detach().cpu().numpy())

            presave = np.concatenate(presave)
            effsave = np.concatenate(effsave)

            np.save(os.path.join(output_dir, f"fold{fold + 1}_presave.npy"), presave)
            np.save(os.path.join(output_dir, f"fold{fold + 1}_effsave.npy"), effsave)
            
            spearman_value = ss.spearmanr(effsave.flatten(), presave.flatten())[0]
            all_spearman_values.append(spearman_value)
            writer.add_scalar(f'OUTPUT_base/fold{fold + 1}_value', spearman_value, fold + 1)
            print(f"Fold {fold + 1}: Spearman = {spearman_value:.4f}")

            diagnostics.analyze_fold_distribution(
                fold=fold+1, train_idx=train_idx, val_idx=val_idx,
                seq=seq, labels=labels, geneids=geneids, 
                geneproteins=geneproteins, features=features,
                spearman=spearman_value
            )

        print(f"\n{'='*50}")
        print("RUNNING DIAGNOSTICS...")
        low_perf_analysis = diagnostics.plot_fold_analysis()
        
        if low_perf_analysis.get('low_perf_folds'):
            print(f"⚠️  Low performance folds: {low_perf_analysis['low_perf_folds']}")

        if len(all_spearman_values) > 0:
            mean_spearman = np.mean(all_spearman_values)
            std_spearman = np.std(all_spearman_values)
            
            print(f"\n{'='*50}")
            print(f"Results: Mean={mean_spearman:.4f}, Std={std_spearman:.4f}")
            print(f"Range: [{min(all_spearman_values):.4f}, {max(all_spearman_values):.4f}]")
            
            dataset_results[dataset_name] = {
                'n_samples': n_samples,
                'spearman_values': all_spearman_values,
                'mean_spearman': mean_spearman,
                'std_spearman': std_spearman,
                'config': config.config
            }
        
        writer.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}")
print(f"{'Dataset':<20} {'N':<8} {'Mean':<10} {'Std':<10} {'Folds':<8}")
print(f"{'-'*80}")

detailed_summary = []
detailed_summary.append("="*80)
detailed_summary.append("FINAL SUMMARY - DETAILED REPORT")
detailed_summary.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
detailed_summary.append("="*80)
detailed_summary.append("")

for dataset, result in dataset_results.items():
    line = (f"{dataset:<20} {result['n_samples']:<8} {result['mean_spearman']:<10.4f} "
            f"{result['std_spearman']:<10.4f} {len(result['spearman_values']):<8}")
    print(line)
    
    detailed_summary.append(f"Dataset: {dataset}")
    detailed_summary.append(f"  Samples: {result['n_samples']}")
    detailed_summary.append(f"  Mean Spearman: {result['mean_spearman']:.4f}")
    detailed_summary.append(f"  Std Spearman: {result['std_spearman']:.4f}")
    detailed_summary.append(f"  Min: {min(result['spearman_values']):.4f}")
    detailed_summary.append(f"  Max: {max(result['spearman_values']):.4f}")
    detailed_summary.append(f"  All values: {[f'{s:.4f}' for s in result['spearman_values']]}")
    detailed_summary.append(f"  Configuration:")
    for k, v in result['config'].items():
        detailed_summary.append(f"    {k}: {v}")
    detailed_summary.append("-"*40)

detailed_summary.append("="*80)

summary_file_path = os.path.join(result_base_dir, "final_summary_detailed.txt")
with open(summary_file_path, 'w') as f:
    f.write('\n'.join(detailed_summary))
    f.write('\n')

print(f"\nDetailed summary saved to: {summary_file_path}")
print(f"{'='*80}")