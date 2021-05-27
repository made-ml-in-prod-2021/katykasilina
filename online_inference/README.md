Homework # 2
==============================

Модель как REST сервис + Docker

### Сборка образа
Для сборки докер образа и последующего запуска необходимо выполнить команды: 
```
 docker build -t online_inference/predict_app:v1 .

 docker run -p 8080:8080 online_inference/predict_app:v1

```

Внутри контейнера будет запущен сервис, посылать запросы к сервису можно с помощью скрипта make_request.py

### Оптимизация образа

Для сокращения объема образа использовался python:3.9-slim. Он позволил сократить объем образа в 2 раза по сравнению с изначальным объемом
Размер образа на dockerhub - 227.52 MB

### Получение образа с DockerHub

Собранный образ можно скачать с DockerHub следующими командами:
```
 docker pull eabramova/online-infrerence:v1  

 docker run -p 8080:8080 eabramova/online-infrerence:v1

```

### Запуск скрипта генерации запросов к сервису make_request.py
Для запросов к сервису написан скрипт make_request.py.  
Скрипт принимает 2 параметра на вход: 
```
    --data_file_path  - адрес файла с данными
    --count - количество строк из файла, на которые нужно получить ответ сервиса. По умолчанию - 1 строка

    Пример запуска скрипта из папки online_inference:
        python make_request.py --data_file_path=tmp.csv

```

### Запуск тестов
```
    python -m pytest test
```
