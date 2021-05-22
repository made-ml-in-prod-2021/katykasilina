ml_project
==============================

A short description of the project.

Project Organization
------------
    ├── configs
    │   └── feature_params     <- Configs for features
    │   └── path_config        <- Configs for all needed paths
    │   └── splitting_params   <- Configs for splitting params
    │   └── train_params       <- Configs for logreg and randomforest models parametres
    │   └── predict_config.yaml   <- Config for prediction pipline
    │   └── train_config.yaml   <- Config for train pipline
    │ 
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts for splitting dataset to train and test
    │   │   └── make_dataset.py
    │   │
    │   ├── entities       <- Scripts for creating dataclasses
    │   │    
    │   │
    │   ├── features              <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    |   |   └── custom_scaler.py  <- Custom scaler transformer
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    |   |
    │   ├── outputs       <- Hydra logs
    │   │   
    │   ├──  utils        <- Scripts for serialized models, reading data
    │   |    └── utils.py
    |   |
    |   ├── predict_pipeline.py   <- pipeline for making predictions
    |   |
    |   └── train_pipeline.py     <- pipeline for model training
    |
    ├── tests              <- tests for the project
    ├── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    ├── LICENSE
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── README.md          <- The top-level README for developers using this project.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

--------
Запуск обучения
------------
Для корректного обучения модели нужны следующие конфиги:

    ├── configs
    │   └── feature_params
    │   │   └── features.yaml   <- Конфиг с наименованием фичей и разделением на категориальные/численные/таргет фичи.
    │   │                          Так же необходимо указать требуется ли приведение к норм. распределению
    │   │                         численных фиччей
    │   │
    │   ├── path_config           
    │   │   └── path_config.yaml <- Конфиг с путями до всех нужных файлов: данные, модели и тд
    │   │
    │   ├── splitting_params
    │   │   └── splitting_params.yaml <- Конфиг с параметрами для split
    │   │
    │   ├── train_params
    |   |   └── logreg.yaml          <- Конфиг с параметрами модели logisticregression
    │   │   └── rf.yaml              <- Конфиг с параметрами модели randomforest
    │   │
    │   ├── train_config.yaml      <- Конфиг для train_pipline, которые использует Hydra

Запуск обучения модели:  `python src/train_pipeline.py`

--------
Запуск построения прогноза
--------
Для корректного построения прогноза нужны следующие конфиги:

    ├── configs
        └── predict_config.yaml <- Конфиг содержит пути до модели и трансформатора.
        
Запуск построения прогноза:  `python src/predict_pipeline.py`
