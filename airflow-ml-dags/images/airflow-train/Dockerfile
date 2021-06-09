FROM airflow-ml-base:latest

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train.py /train.py

ENTRYPOINT ["python", "train.py"]