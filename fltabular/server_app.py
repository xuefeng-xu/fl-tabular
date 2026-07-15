"""fltabular: Flower Example on Tabular Dataset."""

import json
from pathlib import Path
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fltabular.task import get_model

# Create ServerApp
app = ServerApp()


def save_result(dataset, n_clients, partition, model, preprocess, lr, eval_result):
    PROJECT_ROOT = Path(__file__).parent.parent
    result_file = (
        PROJECT_ROOT
        / f"result/{dataset}-{n_clients}-{partition}-{model}-{preprocess}-{lr}.jsonl"
    )
    result_file.parent.mkdir(exist_ok=True)

    accuracy, loss = [], []
    for i in range(1, len(eval_result) + 1):
        accuracy.append(float(eval_result[i]["accuracy"]))
        loss.append(float(eval_result[i]["loss"]))

    with open(result_file, "a") as f:
        f.write(json.dumps({"accuracy": accuracy, "loss": loss}) + "\n")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]

    # Init global model
    net = get_model(
        dataset=context.run_config["dataset"], model=context.run_config["model"]
    )
    arrays = ArrayRecord(net.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg()

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    save_result(
        dataset=context.run_config["dataset"],
        n_clients=len(grid.get_node_ids()),
        partition=context.run_config["partition"],
        model=context.run_config["model"],
        preprocess=context.run_config["preprocess"],
        lr=context.run_config["lr"],
        eval_result=result.evaluate_metrics_clientapp,
    )
