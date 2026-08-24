# Federated Learning with Tabular dataset

This repository contains experiments on the impact of different preprocessing strategies in federated learning.
The implementation of federated data preprocessing methods is available at https://github.com/xuefeng-xu/fedps.

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

To run a simulation with 30 clients + `"adult"` dataset + IID partitioning + `"mlp"` model + `"federated"` preprocessing + learning rate of 0.001:

```bash
flwr run . --federation-config 'options.num-supernodes=30' --run-config 'dataset="adult" partition="iid" model="mlp" preprocess="federated" lr=0.001'
```

Results are saved in `./result/{dataset}-{num-supernodes}-{partition}-{model}-{preprocess}-{lr}.jsonl`

| Parameter | Description | Values |
|---|---|---|
| `num-supernodes` | Number of clients | Integer (e.g., `30`) |
| `dataset` | Dataset name | [`"adult"`](https://archive.ics.uci.edu/dataset/2/adult), [`"bank"`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`"cover"`](https://archive.ics.uci.edu/dataset/31/covertype) |
| `partition` | Data partitioning strategy | `"iid"`, `"labelskew"`, `"featureskew"` |
| `model` | Model name | `"lr"` or `"mlp"` |
| `preprocess` | Preprocessing method | `"no"`, `"local"`, `"federated"` |
| `lr` | Learning rate | Float (e.g., `0.001`) |

## Reproduction

### Main results

To reproduce results for the `adult` dataset + 30 clients + IID partitioning + `lr` model:

```bash
python reproduce.py --dataset adult --n_clients 30 --partition iid --model lr
```

Similarly, to reproduce results for the `bank` dataset + 10 clients + label skew partitioning + `mlp` model:

```bash
python reproduce.py --dataset bank --n_clients 10 --partition labelskew --model mlp
```

Plots are saved in `./img/{dataset}-{n_clients}-{partition}-{model}.pdf`

| Parameter | Description | Values |
|---|---|---|
| `dataset` | Dataset name | [`adult`](https://archive.ics.uci.edu/dataset/2/adult), [`bank`](https://archive.ics.uci.edu/dataset/222/bank+marketing), [`cover`](https://archive.ics.uci.edu/dataset/31/covertype) |
| `n_clients` | Number of clients | Integer (e.g., `30`) |
| `partition` | Data partitioning strategy | `iid`, `labelskew`, `featureskew` |
| `model` | Model name | `lr` or `mlp` |

---

### Demo (Figure 1)

```bash
python demo.py
```

Plots are saved in `./img/demo/*.pdf`

## Acknowledgements

This code is adapted from [Flower Example on Adult Census Income Tabular Dataset](https://github.com/adap/flower/tree/v1.23.0/examples/fl-tabular)

## License

[Apache License 2.0](LICENSE)
