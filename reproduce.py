import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pandas import read_json
from argparse import ArgumentParser

LearningRates = {
    "adult": {
        "iid": {
            "lr": {"no": 0.001, "local": 0.01, "federated": 0.01},
            "mlp": {"no": 0.01, "local": 0.001, "federated": 0.001},
        },
        "niid": {
            "lr": {"no": 0.001, "local": 0.1, "federated": 0.1},
            "mlp": {"no": 0.001, "local": 0.001, "federated": 0.01},
        },
    },
    "bank": {
        "iid": {
            "lr": {"no": 0.001, "local": 0.01, "federated": 0.01},
            "mlp": {"no": 0.001, "local": 0.001, "federated": 0.001},
        },
        "niid": {
            "lr": {"no": 0.001, "local": 0.01, "federated": 0.01},
            "mlp": {"no": 0.001, "local": 0.001, "federated": 0.001},
        },
    },
    "cover": {
        "iid": {
            "lr": {"no": 0.001, "local": 0.001, "federated": 0.01},
            "mlp": {"no": 0.0001, "local": 0.001, "federated": 0.001},
        },
        "niid": {
            "lr": {"no": 0.0001, "local": 0.001, "federated": 0.01},
            "mlp": {"no": 0.0001, "local": 0.001, "federated": 0.001},
        },
    },
}


def run_experiment(dataset, iid, model, preprocess, lr):
    cmd = ["flwr", "run", ".", "--run-config"]
    cmd += [
        f'dataset="{dataset}" '
        f"iid={iid} "
        f'model="{model}" '
        f'preprocess="{preprocess}" '
        f"lr={lr}"
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
        "--iid",
        type=str,
        default="true",
        choices=["true", "false"],
        help="IID distribution",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["lr", "mlp"],
        help="Model name",
    )
    args = parser.parse_args()
    iid = "iid" if args.iid == "true" else "niid"

    PROJECT_ROOT = Path(__file__).parent
    fig, ax = plt.subplots(figsize=(3, 3))

    for ls, preprocess in zip(["-", "--", ":"], ["federated", "local", "no"]):
        lr = LearningRates[args.dataset][iid][args.model][preprocess]

        for _ in range(5):
            run_experiment(
                dataset=args.dataset,
                iid=args.iid,
                model=args.model,
                preprocess=preprocess,
                lr=lr,
            )

        result_file = (
            PROJECT_ROOT
            / f"result/{args.dataset}_{iid}_{args.model}_{preprocess}_{lr}.jsonl"
        )

        eval_result = read_json(result_file, lines=True)

        accuracy = []
        for acc in eval_result["accuracy"].values:
            if len(acc) == 100:
                accuracy.append(acc)

        acc_mean = np.mean(accuracy, axis=0)
        print(f"preprocess: {preprocess :<10} accuracy: {acc_mean[-1] :.2f}")
        ax.plot(acc_mean, ls, label=f"{preprocess.title()}")

    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Accuracy")
    if args.iid == "true":
        ax.set_title(f"{args.dataset.title()}: IID ({args.model.upper()})")
    else:
        ax.set_title(f"{args.dataset.title()}: Non-IID ({args.model.upper()})")
    ax.legend()

    fig.tight_layout()
    img_file = PROJECT_ROOT / f"img/{args.dataset}_{iid}_{args.model}.pdf"
    img_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_file)

    plt.show()
