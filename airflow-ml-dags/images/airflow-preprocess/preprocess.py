import os
from pathlib import Path

import click
import pandas as pd


@click.command()
@click.option("--input-dir")
@click.option("--out_dir")
def preprocess(input_dir: str, out_dir: str):

    x = pd.read_csv(os.path.join(input_dir, "data.csv"))
    y = pd.read_csv(os.path.join(input_dir, "target.csv"))

    y.set_axis(["target"], axis=1)

    output_dir_path = Path(out_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    data_processed = pd.concat([x, y], axis=1)
    data_processed.to_csv(os.path.join(out_dir, "train_data.csv"), index=False)


if __name__ == '__main__':
    preprocess()
