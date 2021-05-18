import os

import click
import requests

import pandas as pd
import numpy as np

SERVICE_HOST = os.environ.get("HOST", default="0.0.0.0")
SERVICE_PORT = os.environ.get("PORT", default=8080)


def get_data(data_file_path: str):
    data = pd.read_csv(data_file_path)
    return data

@click.command()
@click.option("--data_file_path", default="tmp.csv")
@click.option("--count", default=1, help="number of requests")
def make_request(data_file_path: str, count: int):

    data = get_data(data_file_path)
    request_features = list(data.columns)

    data_len = len(data)

    if count > data_len:
        count = data_len

    for i in range(count):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        full_data = dict(zip(request_features, request_data))
        response = requests.get(
            url=f"http://{SERVICE_HOST}:{SERVICE_PORT}/predict",
            json=full_data
        )
        if response.status_code == 200:
            click.echo(f"Request:\t {full_data}")
            click.echo(f"Response:\t {response.json()}")
        else:
            click.echo(f"ERROR {response.status_code}: {response.text}")

        click.echo("---")


if __name__ == "__main__":
    make_request()
