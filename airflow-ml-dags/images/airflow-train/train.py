import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command()
@click.option("--input-dir")
@click.option("--out_dir")
def train(input_dir: str, out_dir: str):

    input_data_path = Path(input_dir)
    dataset = pd.read_csv(input_data_path / "train.csv")

    y = dataset[['target']]
    x = dataset.drop(['target'], axis=1)

    model = LogisticRegression()
    model.fit(x, y)

    output_model_path = Path(out_dir)
    output_model_path.mkdir(exist_ok=True, parents=True)

    output_model_path = str(output_model_path / "model")
    with open(output_model_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train()
