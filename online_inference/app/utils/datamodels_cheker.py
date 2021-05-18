from fastapi import HTTPException
import pandas as pd
from pydantic import BaseModel, validator


class InputDataModel(BaseModel):
    sex: int
    cp: int
    fbs: int
    restecg: int
    exang: int
    slope: int
    ca: int
    thal: int
    age: int
    trestbps: int
    chol: int
    thalach: int
    oldpeak: float

    def convert_to_pandas(self) -> pd.DataFrame:
        data = pd.DataFrame.from_dict([self.dict()], orient='columns')
        return data

    @validator("restecg", "slope")
    def check_restecg_slope(cls, value):
        if value not in (0, 1, 2):
            raise HTTPException(
                status_code=400,
                detail="Value should be in 0, 1, 2"
            )
        return value

    @validator("ca")
    def check_ca(cls, value):
        if value not in (0, 1, 2, 3, 4):
            raise HTTPException(
                status_code=400,
                detail="Value should be in 0, 1, 2, 3, 4"
            )
        return value

    @validator("age")
    def check_age(cls, value):
        if not (0 < value <= 120):
            raise HTTPException(
                status_code=400,
                detail=f"Value should be less then 120 ang greater 0"
            )
        return value

    @validator('sex', 'fbs', 'exang')
    def check_bin_features(cls, value):
        if value not in (0, 1):
            raise HTTPException(
                status_code=400,
                detail="Value should be in 0 or 1."
            )
        return value

    @validator("cp", "thal")
    def check_cp(cls, value):
        if value not in (0, 1, 2, 3):
            raise HTTPException(
                status_code=400,
                detail="Value should be in 0, 1, 2, 3"
            )
        return value

    @validator("trestbps", "thalach")
    def check_pressure(cls, value):
        if not (40 <= value <= 230):
            raise HTTPException(
                status_code=400,
                detail=f"Value should be between 40 and 230"
            )
        return value


class OutputDataModel(BaseModel):
    label: int