import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pandas import read_json
from argparse import ArgumentParser

LearningRates = {
    "adult": {
        10: {
            "iid": {
                "lr": {"no": 0.001, "local": 0.01, "federated": 0.01},
                "mlp": {"no": 0.001, "local": 0.001, "federated": 0.001},
            },
            "labelskew": {
                "lr": {"no": 0.0001, "local": 0.01, "federated": 0.01},
                "mlp": {"no": 0.001, "local": 0.0001, "federated": 0.001},
            },
            "featureskew": {
                "lr": {"no": 0.001, "local": 0.001, "federated": 0.01},
                "mlp": {"no": 0.001, "local": 0.001, "federated": 0.001},
            },
        },
        30: {
            "iid": {
                "lr": {"no": 0.001, "local": 0.01, "federated": 0.01},
                "mlp": {"no": 0.01, "local": 0.001, "federated": 0.001},
            },
            "labelskew": {
                "lr": {"no": 0.001, "local": 0.1, "federated": 0.1},
                "mlp": {"no": 0.001, "local": 0.001, "federated": 0.01},
            },
            "featureskew": {
                "lr": {"no": 0.01, "local": 0.001, "federated": 0.01},
                "mlp": {"no": 0.001, "local": 0.001, "federated": 0.001},
            },
        },
    },
    "bank": {
        10: {
            "iid": {
                "lr": {"no": 0.001, "local": 0.01, "federated": 0.01},
                "mlp": {"no": 0.001, "local": 0.001, "federated": 0.001},
            },
            "labelskew": {
                "lr": {"no": 0.0001, "local": 0.01, "federated": 0.001},
                "mlp": {"no": 0.001, "local": 0.0001, "federated": 0.0001},
            },
            "featureskew": {
                "lr": {"no": 0.001, "local": 0.001, "federated": 0.001},
                "mlp": {"no": 0.01, "local": 0.0001, "federated": 0.001},
            },
        },
        30: {
            "iid": {
                "lr": {"no": 0.001, "local": 0.01, "federated": 0.01},
                "mlp": {"no": 0.001, "local": 0.001, "federated": 0.001},
            },
            "labelskew": {
                "lr": {"no": 0.001, "local": 0.01, "federated": 0.01},
                "mlp": {"no": 0.001, "local": 0.001, "federated": 0.001},
            },
            "featureskew": {
                "lr": {"no": 0.001, "local": 0.001, "federated": 0.001},
                "mlp": {"no": 0.01, "local": 0.001, "federated": 0.001},
            },
        },
    },
    "cover": {
        10: {
            "iid": {
                "lr": {"no": 0.0001, "local": 0.001, "federated": 0.01},
                "mlp": {"no": 0.0001, "local": 0.001, "federated": 0.001},
            },
            "labelskew": {
                "lr": {"no": 0.0001, "local": 0.0001, "federated": 0.001},
                "mlp": {"no": 0.0001, "local": 0.001, "federated": 0.001},
            },
            "featureskew": {
                "lr": {"no": 0.0001, "local": 0.0001, "federated": 0.0001},
                "mlp": {"no": 0.00001, "local": 0.00001, "federated": 0.0001},
            },
        },
        30: {
            "iid": {
                "lr": {"no": 0.001, "local": 0.001, "federated": 0.01},
                "mlp": {"no": 0.0001, "local": 0.001, "federated": 0.001},
            },
            "labelskew": {
                "lr": {"no": 0.0001, "local": 0.001, "federated": 0.01},
                "mlp": {"no": 0.0001, "local": 0.001, "federated": 0.001},
            },
            "featureskew": {
                "lr": {"no": 0.0001, "local": 0.0001, "federated": 0.0001},
                "mlp": {"no": 0.0001, "local": 0.0001, "federated": 0.0001},
            },
        },
    },
}


def run_experiment(dataset, n_clients, partition, model, preprocess, lr):
    cmd = [
        "flwr",
        "run",
        ".",
        "--federation-config",
        f"options.num-supernodes={n_clients} ",
    ]
    cmd += [
        "--run-config",
        f'dataset="{dataset}" '
        f'partition="{partition}" '
        f'model="{model}" '
        f'preprocess="{preprocess}" '
        f"lr={lr}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Reproduce the results of the experiment.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=["adult", "bank", "cover"],
        help="Dataset name",
    )
    parser.add_argument(
        "--n_clients",
        type=int,
        default=30,
        help="Number of clients",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="iid",
        choices=["iid", "labelskew", "featureskew"],
        help="Data partitioning strategy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["lr", "mlp"],
        help="Model name",
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).parent
    fig, ax = plt.subplots(figsize=(3, 3))

    for ls, preprocess in zip(["-", ":", "--"], ["federated", "local", "no"]):
        lr = LearningRates[args.dataset][args.n_clients][args.partition][args.model][
            preprocess
        ]

        for _ in range(5):
            run_experiment(
                dataset=args.dataset,
                n_clients=args.n_clients,
                partition=args.partition,
                model=args.model,
                preprocess=preprocess,
                lr=lr,
            )

        result_file = (
            PROJECT_ROOT
            / f"result/{args.dataset}-{args.n_clients}-{args.partition}-{args.model}-{preprocess}-{lr}.jsonl"
        )

        eval_result = read_json(result_file, lines=True)

        accuracy = []
        for acc in eval_result["accuracy"].values:
            if len(acc) == 100:
                accuracy.append(acc)

        acc_mean = np.mean(accuracy, axis=0)
        acc_std = np.std(accuracy, axis=0)

        print(
            f"preprocess: {preprocess :<10} accuracy: {acc_mean[-1] :.2f} ± {acc_std[-1] :.4f}"
        )

        ax.plot(acc_mean, ls, label=f"{preprocess.title()}")
        ax.fill_between(
            range(len(acc_mean)),
            acc_mean - acc_std,
            acc_mean + acc_std,
            alpha=0.3,
        )

    if args.dataset == "adult":
        ax.set_ylim(0.6, 0.9)
    elif args.dataset == "bank":
        ax.set_ylim(0.82, 0.92)
    elif args.dataset == "cover":
        ax.set_ylim(0.4, 0.92)

    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Test Accuracy")

    partition_map = {
        "iid": "IID",
        "labelskew": "Label Skew",
        "featureskew": "Feature Skew",
    }

    ax.set_title(
        f"{args.dataset.title()}: {partition_map[args.partition]} ({args.model.upper()})"
    )
    ax.legend()

    fig.tight_layout()
    img_file = (
        PROJECT_ROOT
        / f"img/{args.dataset}-{args.n_clients}-{args.partition}-{args.model}.pdf"
    )
    img_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_file)

    plt.show()
