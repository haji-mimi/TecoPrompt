# TecoPrompt: Temporal-Conservative Prompt Learning for Vision-Language Models

**ECCV 2026**

## Setup

Make sure [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is installed.

```bash
git clone https://github.com/haji-mimi/TecoPrompt.git
cd TecoPrompt

conda create -y -n tecoprompt python=3.8
conda activate tecoprompt

# Install PyTorch. Adjust the CUDA version for your system if necessary.
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Dassl and the remaining dependencies.
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..
```

## Dataset Preparation

Follow the instructions in [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to prepare the datasets.

Note that the [Food101N](https://www.kaggle.com/datasets/kuanghueilee/food-101n) dataset needs to be downloaded separately. Food101N uses the same test set as Food101.

## How to Run

We provide the running script in `scripts/TecoPrompt/`.

Before running an experiment, open `scripts/TecoPrompt/main.sh` and set `DATA` to your dataset root path. Run all commands from the repository root.

By default, the examples below use StanfordCars with 16 shots, a 50% noise rate, and $K=8$.

**TecoPrompt (StanfordCars, symmetric noise):**

```bash
bash scripts/TecoPrompt/main.sh stanford_cars 16 0.50 sym 196 8
```

**TecoPrompt (StanfordCars, asymmetric noise):**

```bash
bash scripts/TecoPrompt/main.sh stanford_cars 16 0.50 asym 196 8
```

The six parameters are the dataset name, number of shots, noise rate, noise type, number of classes, and temporal stability window $K$, respectively. If $K$ is omitted, it defaults to 8.

```bash
bash scripts/TecoPrompt/main.sh <DATASET> <SHOTS> <RATE> <TYPE> <CLASS> [K]
```

All results are saved to `output/`.
