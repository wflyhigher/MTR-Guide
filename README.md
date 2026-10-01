# **MTR-Guide**

MTR-Guide is a multimodal model for sgRNA activity prediction across different Cas9 variants. It combines sgRNA sequence features, UniPert protein embeddings, GenePT gene embeddings and 28 biological sequence features in a unified neural network.

## Create Environment

The repository provides a complete Conda environment file. The environment is named `MTR-Guide` and uses Python 3.9, PyTorch and CUDA 11.3 dependencies.

```bash
cd MTR-Guide
conda env create -f environment.yml
conda activate MTR-Guide
```

You can check whether PyTorch can access the GPU with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The supplied environment file contains Linux/CUDA package builds. Linux or WSL2 is recommended. When using Windows native Python, install a PyTorch build compatible with the local CUDA driver and then install the remaining packages listed in `environment.yml`.

## Get Started

### Data and Feature Files

The required data and pre-computed features are already included in the repository:

- `MTR-Guide_transfer/data_M3`: transfer-learning datasets
- `MTR-Guide_transfer/unipert_embedding`: UniPert protein embeddings
- `MTR-Guide_transfer/GenePT_Embedding`: GenePT gene embeddings
- `MTR-Guide_transfer/data_protein`: processed Cas9 protein feature files
- `MTR-Guide_base`: base-model data, embeddings and ablation scripts

The data loader expects tab-separated files containing `Sequence`, `Value`, `Protein_ID`, `Gene_B`, `Protein` and the 28 biological feature columns. Sequences are filtered to length 23, and records without matching UniPert or GenePT embeddings are skipped.

### Configure Paths and GPU

The training scripts currently keep data and result directories as constants near the top of each file. Before running them, replace the default `/home/...` paths with paths on your machine. The main files to edit are:

```text
MTR-Guide_transfer/temtrain_pretrain.py
MTR-Guide_transfer/temtrain_finetune.py
MTR-Guide_base/temtrain_adap_M3.py
MTR-Guide_base/temtrain_adap_M3_nounipert.py
MTR-Guide_base/temtrain_adap_M3_nogene.py
MTR-Guide_base/temtrain_adap_M3_nobio.py
```

Set `device` to the GPU index available on your machine. The default scripts call `torch.cuda.set_device`, so a CUDA-capable GPU is required unless the scripts are modified for CPU execution.

## Training

### Transfer Learning

The recommended workflow is to pre-train the model and then fine-tune it on the target Cas9 datasets.

First, run multi-dataset pre-training. The default pre-training datasets are `esp_wang`, `WT_wang`, `HF_wang`, `esp_kim`, `WT_kim`, `HF_kim` and `WT_xiang`. The script also performs internal test evaluation and zero-shot evaluation on `evo`, `Hypa`, `sniper` and `xcas9`.

```bash
cd MTR-Guide_transfer
python temtrain_pretrain.py
```

The pre-trained model is saved as `pretrained_model.pt` under the configured `result_pretrain` directory.

Then, update `PRETRAINED_MODEL_PATH` in `temtrain_finetune.py` and run fine-tuning:

```bash
cd MTR-Guide_transfer
python temtrain_finetune.py
```

The fine-tuning script uses five-fold training for `evo`, `Hypa`, `sniper` and `xcas9`. Its default strategy freezes the sgRNA encoder and gene-embedding modules (modules A+B) and uses a weighted ensemble for test prediction.

### Base Model and Ablation Experiments

The `train_all*.py` files define network variants. The executable training scripts are:

```bash
cd MTR-Guide_base

# Full MTR-Guide model
python temtrain_adap_M3.py

# Ablation variants
python temtrain_adap_M3_nounipert.py
python temtrain_adap_M3_nogene.py
python temtrain_adap_M3_nobio.py
```

The base scripts perform adaptive stratified K-fold training and automatically create model, prediction, TensorBoard and diagnostic result directories.

To change model hyperparameters, modify the constants in the corresponding script. Common settings include:

```text
dataset_dir: folder containing the dataset files
unipert_embedding_dir: folder containing UniPert embeddings
GenePT_embedding_dir: folder containing GenePT embeddings
result_base_dir: folder for models and evaluation results
device: GPU index
random_seed: random seed
lr: learning rate
Fold_num: number of folds for fine-tuning
BATCH_SIZE: training batch size
```

## Citation

Wenfeng He #, Yiming Li #, Yalin Hou, Chengqian Lu, Fuhao Zhang, Dabin Kuang, Min Li, Min Zeng\*, "Integrating Multimodal Target Representations to Improve sgRNA Activity Prediction across Cas9 Variants".
