"""fltabular: Flower Example on Tabular Dataset."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fltabular.task import load_feddata, get_model, get_device, trainer, evaluator

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Load dataset
    dataset = context.run_config["dataset"]
    partition_id = context.node_config["partition-id"]
    train_loader, _ = load_feddata(
        dataset=dataset,
        preprocess=context.run_config["preprocess"],
        iid=context.run_config["iid"],
        partition_id=partition_id,
        num_partitions=context.node_config["num-partitions"],
    )

    # Load model
    net = get_model(dataset=dataset, model=context.run_config["model"])
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Perform training
    trainer(dataset, net, train_loader, lr=context.run_config["lr"])

    # Construct and return reply Message
    model_record = ArrayRecord(net.state_dict())
    metrics = {
        "num-examples": len(train_loader),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Load dataset
    dataset = context.run_config["dataset"]
    partition_id = context.node_config["partition-id"]
    _, test_loader = load_feddata(
        dataset=dataset,
        preprocess=context.run_config["preprocess"],
        iid=context.run_config["iid"],
        partition_id=partition_id,
        num_partitions=context.node_config["num-partitions"],
    )

    # Load model
    net = get_model(dataset=dataset, model=context.run_config["model"])
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Perform evaluation
    loss, accuracy = evaluator(dataset, net, test_loader)

    # Construct and return reply Message
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "num-examples": len(test_loader),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
