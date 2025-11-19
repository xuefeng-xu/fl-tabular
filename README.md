# Federated Learning with Tabular dataset

## Installation

1. Create a Python environment:

```bash
conda create -n fl-tabular python=3.12
conda activate fl-tabular
```

2. Clone this repository:

```bash
git clone https://github.com/xuefeng-xu/fl-tabular.git && cd fl-tabular
```

3. Install dependencies:

```bash
pip install -e .
```

## Run Simulation

```bash
flwr run . --run-config 'dataset="adult" iid=true model="mlp" preprocess="federated" lr=0.001'
```

Results are saved in `./result/{dataset}-{iid}-{model}-{preprocess}-{lr}.jsonl`

| Parameter | Description | Values |
|---|---|---|
| `dataset` | Dataset name | [`"adult"`](https://archive.ics.uci.edu/dataset/2/adult), [`"bank"`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`"cover"`](https://archive.ics.uci.edu/dataset/31/covertype) |
| `iid` | IID distribution | `true` or `false` |
| `model` | Model name | `"lr"` or `"mlp"` |
| `preprocess` | Preprocessing method | `"no"`, `"local"` or `"federated"` |
| `lr` | Learning rate | Float (e.g., `0.001`) |

## Reproduction

To reproduce results for the `adult` dataset + IID setting + `lr` model:

```bash
python reproduce.py --dataset adult --iid true --model lr
```

Similarly, to reproduce results for the `bank` dataset + Non-IID setting + `mlp` model:

```bash
python reproduce.py --dataset bank --iid false --model mlp
```

Plots are saved in `./img/{dataset}-{iid}-{model}.pdf`

## Acknowledgements

This code is adapted from [Flower with a Tabular Dataset Example](https://github.com/adap/flower/tree/v1.23.0/examples/fl-tabular)
