import zipfile
import gzip
import shutil
from pathlib import Path
from pandas import read_csv, concat
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def download(url, zip_file):
    try:
        urlretrieve(url, zip_file)
    except Exception as e:
        raise RuntimeError(f"Failed to download from {url}: {e}") from e


def extract(zip_file, extract_path):
    file_type = zip_file.suffix
    try:
        if file_type == ".zip":
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(extract_path)
        elif file_type == ".gz":
            with gzip.open(zip_file, "rb") as f_in, open(extract_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            raise ValueError(f"Unknown file type: {file_type}")

    except Exception as e:
        raise RuntimeError(f"Failed to extract {zip_file}: {e}") from e


def download_and_extract(url, zip_file, extract_path):
    download(url, zip_file)
    extract(zip_file, extract_path)


def load_adult(dataset_dir):
    file = dataset_dir / "adult.data"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "adult.zip"

        download_and_extract(
            "https://archive.ics.uci.edu/static/public/2/adult.zip",
            zip_file,
            file.parent,
        )

    X = read_csv(file, header=None)
    X.columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    y = X.pop("income").map({" >50K": 1, " <=50K": 0})

    return X, y


def load_bank(dataset_dir):
    file = dataset_dir / "bank-full.csv"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "bank+marketing.zip"

        download_and_extract(
            "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
            zip_file,
            file.parent,
        )

        zip_sub_file = file.parent / "bank.zip"
        extract(zip_sub_file, file.parent)

    X = read_csv(file, sep=";")

    y = X.pop("y").map({"yes": 1, "no": 0})

    return X, y


def load_cover(dataset_dir):
    file = dataset_dir / "covtype.data"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "covertype.zip"

        download_and_extract(
            "https://archive.ics.uci.edu/static/public/31/covertype.zip",
            zip_file,
            file.parent,
        )

        gz_file = file.parent / "covtype.data.gz"
        extract(gz_file, file)

    X = read_csv(file, header=None)
    X.columns = (
        [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
        + [f"Wilderness_Area{i}" for i in range(1, 5)]
        + [f"Soil_Type{i}" for i in range(1, 41)]
        + ["Cover_Type"]
    )

    y = X.pop("Cover_Type") - 1

    return X, y


def load_data(dataset, encode=False, std=False):
    PROJECT_ROOT = Path(__file__).parent
    dataset_dir = PROJECT_ROOT / f"dataset/{dataset}"

    uci_data = {
        "adult": lambda: load_adult(dataset_dir),
        "bank": lambda: load_bank(dataset_dir),
        "cover": lambda: load_cover(dataset_dir),
    }

    if dataset not in uci_data:
        raise ValueError(f"Unknown dataset: {dataset}")

    X, y = uci_data[dataset]()
    y.name = "label"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if encode:
        objcol = X_train.select_dtypes(exclude=["float", "int"]).columns
        encoder = OrdinalEncoder()
        X_train[objcol] = encoder.fit_transform(X_train[objcol])
        X_test[objcol] = encoder.transform(X_test[objcol])

    if std:
        numcol = X_train.select_dtypes(include=["float", "int"]).columns
        scaler = StandardScaler()
        X_train[numcol] = scaler.fit_transform(X_train)
        X_test[numcol] = scaler.transform(X_test)

    data_train = concat([X_train, y_train], axis=1)
    data_test = concat([X_test, y_test], axis=1)

    return data_train, data_test
