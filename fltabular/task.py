"""fltabular: Flower Example on Tabular Dataset."""

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset
from flwr_datasets.partitioner import (
    IidPartitioner,
    DirichletPartitioner,
    ContinuousPartitioner,
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dataloader import load_data

fds = None  # Cache FederatedDataset


def load_feddata(
    dataset: str,
    preprocess: str,
    partition: str,
    partition_id: int,
    num_partitions: int,
):
    if preprocess not in ["no", "local", "federated"]:
        raise ValueError(f"Unknown preprocess: {preprocess}")

    global fds
    if fds is None:
        if partition == "iid":
            train_partitioner = IidPartitioner(num_partitions=num_partitions)
        elif partition == "labelskew":
            alpha = 0.5
            train_partitioner = DirichletPartitioner(
                num_partitions=num_partitions, partition_by="label", alpha=alpha
            )
        elif partition == "featureskew":
            strictness = 1.0
            feature_map = {
                "adult": "capital-gain",
                "bank": "duration",
                "cover": "Elevation",
            }
            train_partitioner = ContinuousPartitioner(
                num_partitions=num_partitions,
                partition_by=feature_map[dataset],
                strictness=strictness,
            )

        test_partitioner = IidPartitioner(num_partitions=num_partitions)

        if preprocess == "no":
            df_train, df_test = load_data(dataset, encode=True, std=False)
        elif preprocess == "local":
            df_train, df_test = load_data(dataset, encode=False, std=False)
        elif preprocess == "federated":
            df_train, df_test = load_data(dataset, encode=True, std=True)

        train_partitioner.dataset = Dataset.from_pandas(df_train, preserve_index=False)
        test_partitioner.dataset = Dataset.from_pandas(df_test, preserve_index=False)

    data_train = train_partitioner.load_partition(partition_id).with_format("pandas")[:]
    data_test = test_partitioner.load_partition(partition_id).with_format("pandas")[:]

    X_train = data_train.drop("label", axis=1)
    y_train = data_train["label"]

    X_test = data_test.drop("label", axis=1)
    y_test = data_test["label"]

    if preprocess == "local":
        objcol = X_train.select_dtypes(exclude=["float", "int"]).columns
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[objcol] = encoder.fit_transform(X_train[objcol])
        X_test[objcol] = encoder.transform(X_test[objcol])

        numcol = X_train.select_dtypes(include=["float", "int"]).columns
        scaler = StandardScaler()
        X_train[numcol] = scaler.fit_transform(X_train)
        X_test[numcol] = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

    if dataset == "cover":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    else:
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def get_model(dataset: str, model: str):
    if dataset == "adult":
        input_dim = 14
        output_dim = 1
    elif dataset == "bank":
        input_dim = 16
        output_dim = 1
    elif dataset == "cover":
        input_dim = 54
        output_dim = 7
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if model == "lr":
        net = LR(input_dim=input_dim, output_dim=output_dim)
    elif model == "mlp":
        net = MLP(input_dim=input_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown model: {model}")
    return net


class LR(nn.Module):
    def __init__(self, input_dim: int = 14, output_dim: int = 1):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int = 14, output_dim: int = 1):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_criterion(dataset, device):
    if dataset == "cover":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    return criterion


def trainer(
    dataset, model, train_loader, num_epochs=1, lr=0.001, device=torch.device("cpu")
):
    model.to(device)
    criterion = get_criterion(dataset, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to(device))
            loss = criterion(outputs, y_batch.to(device))
            loss.backward()
            optimizer.step()


def evaluator(dataset, model, test_loader, device=torch.device("cpu")):
    model.to(device)
    model.eval()
    criterion = get_criterion(dataset, device)
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(device))
            batch_loss = criterion(outputs, y_batch.to(device))
            loss += batch_loss.item()
            if dataset == "cover":
                predicted = torch.argmax(outputs, dim=1)
            else:
                predicted = (outputs > 0).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    loss = loss / len(test_loader)
    return loss, accuracy
