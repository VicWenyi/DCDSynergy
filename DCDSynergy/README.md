# DCDSynergy: **Multimodal Cell-Drug Integration-Driven Prediction of Synergistic Drug Combinations** 

This code repository is the supporting material in the paper. In this paper, we propose a novel approach called DCDSynergy, which leverages the integration of chemical structure data and gene expression data to predict the synergistic effects of drug combinations.

![DCDSynergy流程图](C:\Users\wenyi\Desktop\GText_wy\DCDSynergy流程图.png)

## Requirements

The third-party dependencies required for model running are listed in [environment.yaml](./environment.yaml). Specifically, you can use the following command to create an environment based on conda and pip:

```bash
conda create -n DCDSynergy python=3.8
conda activate DCDSynergy
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 -c pytorch
pytorch -c nvidia
conda install -c dglteam/label/th21_cu118 dgl
conda install -c conda-forge rdkit==2024.03.5
pip install dgllife
```

## Data preparation

All data used in this paper are public and accessible. The relevant dataset has been stored in [Cloud Drive](https://drive.google.com/drive/folders/1mgCB3NJJB4RXE_KrxmdlQK7_LXtU66kh?usp=sharing) and can be downloaded to the `./data/raw/` folder. Please refer to the [DATA README](./data/raw/README.md) for the source of each file.

After downloading the relevant dataset and place it in the `./data/raw/` folder you can generate the training set and test set by running

```bash
python dataproc.py
```

## Training

After generating the traning set and test set `(`

ONEIL_train_addproteintoDruggraphyWithProteinInterconnection.pkl、ONEIL_test_addproteintoDruggraphyWithProteinInterconnection.pkl

`)` 

you can start training the model by running

```bash
python main.py
```
