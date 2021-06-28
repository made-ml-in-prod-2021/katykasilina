FROM airflow-ml-base:latest

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY preprocess.py /preprocess.py

ENTRYPOINT ["python", "preprocess.py"]