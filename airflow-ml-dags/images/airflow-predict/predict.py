import pickle
from pathlib import Path

import click
import pandas as pd


@click.command()
@click.option("--input-dir")
@click.option("--input-model-dir")
@click.option("--out_dir")
@click.option("--model_name")
@click.option("--data_file_name")
def predict(input_dir: str, input_model_dir: str, out_dir: str, model_name: str, data_file_name: str):

    input_data_path = Path(input_dir)
    input_model_path = Path(input_model_dir)
    output_dir_path = Path(out_dir)

    with open(input_model_path / model_name, "rb") as f:
        model = pickle.load(f)

    data = pd.read_csv(input_data_path / data_file_name)

    predicts = model.predict(data)

    predicts_data = pd.DataFrame(predicts, columns=["target"])

    output_dir_path.parent.mkdir(parents=True, exist_ok=True)
    predicts_data.to_csv(output_dir_path, index=False)


if __name__ == "__main__":
    predict()