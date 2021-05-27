FROM python:3.9-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "make_prediction:app", "--host", "0.0.0.0", "--port", "8080"]