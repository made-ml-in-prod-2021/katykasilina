import os
import logging.config
from fastapi import FastAPI, HTTPException
import uvicorn

from typing import List

from utils.prediction import predict
from utils.datamodels_cheker import InputDataModel, OutputDataModel
from utils.utils import  load_pkl_file

SOURCE_DIR = "sources"

app = FastAPI()
logger = logging.getLogger("online_inference/prediction")


@app.on_event("startup")
def load_transformer():
    global transformer
    filepath = os.path.join(SOURCE_DIR, "transformer.pkl")
    try:
        transformer = load_pkl_file(filepath)
        logger.info("Transformer is loaded")
    except FileNotFoundError as err:
        logger.error(err)
        return


@app.on_event("startup")
def load_model():
    global model
    filepath = os.path.join(SOURCE_DIR, "model.pkl")
    try:
        model = load_pkl_file(filepath)
        logger.info("Model is loaded")
    except FileNotFoundError as err:
        logger.error(err)
        return


@app.get("/")
def main():

    return "Start page"


@app.get("/health")
def health() -> bool:
    transformer_is_loaded = (transformer is not None)
    model_is_loaded = (model is not None)
    return transformer_is_loaded and model_is_loaded


@app.get("/predict", response_model=OutputDataModel)
def make_prediction(data: InputDataModel):
    if not health():
        logger.error("Transformer or model are not loaded")
        raise HTTPException(
            status_code=500,
            detail="Transformer and model should be loaded"
        )

    prediction = predict(
        data=data.convert_to_pandas(),
        transformer=transformer,
        model=model
    )
    return OutputDataModel(label=int(prediction))


if __name__ == "__main__":
    uvicorn.run("make_prediction:app", host="0.0.0.0", port=os.getenv("PORT", 8000))

