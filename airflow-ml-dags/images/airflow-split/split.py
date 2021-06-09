from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.option("--input-dir")
def split(input_dir: str):
    input_data_path = Path(input_dir)
    data = pd.read_csv(input_data_path / "train_data.csv")
    train_data, val_data = train_test_split(data, test_size=0.2)

    train_data.to_csv(input_data_path / "train.csv", index=False)
    val_data.to_csv(input_data_path / "val.csv", index=False)


if __name__ == '__main__':
    split()
