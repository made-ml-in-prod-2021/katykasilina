import os

import click
from sklearn.datasets import load_breast_cancer


@click.command()
@click.option("--out_dir", required=True)
@click.option("--out_dir_data", required=True)
@click.option("--out_dir_target", required=True)
def download(out_dir: str, out_dir_data: str, out_dir_target: str):

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    os.makedirs(out_dir, exist_ok=True)
    X.to_csv(out_dir_data, index=False)
    y.to_csv(out_dir_target, index=False)



if __name__ == '__main__':
    download()
