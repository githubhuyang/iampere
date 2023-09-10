# IAMPERE: Fixing Privilege Escalations in Cloud Access Control with MaxSAT and Graph Neural Networks

## About the Paper

Identity and Access Management (IAM) is a critical access control service used within cloud platforms. Customers configure IAM to establish secure access control rules for their cloud organizations. However, IAM misconfigurations can lead to Privilege Escalation (PE) attacks. To address this, we introduce `IAMPERE`, an IAM Privilege Escalation Repair Engine. `IAMPERE` efficiently generates an approximately minimal patch to fix a wide range of IAM PEs. At a high level, `IAMPERE` formulates the IAM repair problem as a MaxSAT problem and employs Graph Neural Networks (GNN) to prune the MaxSAT solving search space. We first train a GNN model to produce an intermediate patch, which is relatively compact but not necessarily minimal. A MaxSAT solver then searches within the patch space defined by the intermediate patch to find the final, approximately minimal patch. This research has been accepted by [ASE'23](https://conf.researchr.org/track/ase-2023/ase-2023-papers).

### Authors
Yang Hu∗, Wenxi Wang∗, Sarfraz Khurshid, Kenneth L. McMillan, Mohit Tiwari

### BibTex

If you use any part of our tool or dataset from this repository, please kindly cite our paper:

```tex
@inproceedings{yang2023ase,
    title={Fixing Privilege Escalations in Cloud Access Control with MaxSAT and Graph Neural Networks},
    author={Hu, Yang and Wang, Wenxi and Khurshid, Sarfraz and McMillan, Kenneth L and Tiwari, Mohit},
    booktitle={38th IEEE/ACM International Conference on Automated Software Engineering (ASE 2023)},
    year={2023},
    organization={IEEE}
}
```

## About the Repository

This repository contains the source code and dataset for our prototype tool, `IAMPERE`.

### Repo Structure

```
|-data
|   |--tasks        # Folder storing IAM repair tasks
|   |--pyg          # Folder storing GNN training data
|   |--pt_model     # Folder storing GNN model
|   |--cnf          # Folder storing converted MaxSAT problems
|   |--itm_patches  # Folder storing intermediate patches
|
|-csv               # Folder storing performance CSV files
|-solvers           # Folder storing MaxSAT solvers
|-maxsat_env.sh     # Bash script to deploy the MaxSAT solver
|-meta.py           # Source code for hyperparameter settings
|-gen_pyg_data.py   # Source code for generating graph dataset for GNN
|-nn.py             # Source code defining the GNN model architecture
|-train.py          # Source code for training the GNN model
|-predict.py        # Source code to generate the intermediate patch
|-fpi_verify.py     # Source code for fixed-point iteration-based verification
|-maxsat.py         # Source code for MaxSAT-based repair
|-...
```

### Environmental Setup

#### Python 3

`IAMPERE` is developed in Python 3.10. While we haven't tested `IAMPERE` with other Python 3 versions, it doesn't rely on version-specific Python 3 features. `IAMPERE` depends on various Python packages, including `numpy`, `pandas`, and `tqdm`. We recommend using the latest [Anaconda 3](https://www.anaconda.com/download) distribution, which includes most required packages. Additionally, our tool requires `pytorch` and `pytorch geometric`, which you can install with the following commands:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
```

#### MaxSAT Solver

`IAMPERE` utilizes the `Cashwmaxsat-CorePlus` MaxSAT solver, winner of the [MaxSAT Evaluation 2022](https://maxsat-evaluations.github.io/2022/). To download and deploy it, please execute:

```bash
source maxsat_env.sh
```

The solver's executable files will be saved in the `solvers` folder and will be added to the `PATH` environment variable.

To verify the solver's successful deployment, please run:

```bash
cashwmaxsatcoreplus --help
```

### Data

This repository offers 200 repair tasks, with half used for training (`data/tasks/maxsat-train-small.zip`) and the other half for testing (`data/tasks/maxsat-test-small.zip`). Please unzip these files to access the repair tasks. We plan to release larger repair tasks and their generator in future updates.

### Usage

#### Step 1. Set Hyperparameters

Open `meta.py` to adjust hyperparameter settings. If you're new to `IAMPERE`, the default settings should suffice.

#### Step 2. Generate Graph Dataset

For convenience, we've shared the graphs we generated. Please unzip `data/pyg/maxsat-train-small.zip` and `data/pyg/maxsat-test-small.zip` to access the graph dataset.

To regenerate the graph dataset:

```bash
python3 gen_pyg_data.py
```

This command will initiate MaxSAT-based repair on training tasks and convert them, along with their patches, into a training graph dataset (`data/pyg/maxsat-train-small`). It will then convert testing tasks into a testing graph dataset (`data/pyg/maxsat-test-small`). With the default setting, generating this dataset usually takes under 20 minutes.

#### Step 3. Train a GNN Model

To start training the GNN model, please execute:

```bash
python3 train.py
```

The default setting trains the model over 100 epochs. Metrics like precision and recall will be displayed in the terminal and saved in `csv/*_gnn-train.csv`. The model will be stored as `data/pt_model/pi-best.pt`.

#### Step 4. Generate Intermediate Patches

To produce intermediate patches for testing set tasks:

```bash
python3 predict.py
```

Patches will be saved in `data/itm_patches`, while patch size and repair time metrics are stored in `csv/*_gnn-predict.csv`.

#### Step 5. Generate Final Patches

To generate final patches based on GNN-produced intermediate patches:

```bash
python3 maxsat.py
```

Patch sizes and repair time metrics will be stored in `csv/*_wcp_c.csv`.

### Contact
Yang Hu (*huyang@utexas.edu*), Wenxi Wang (*wenxiw@utexas.edu*)

### License

[MIT License](LICENSE)