# https://runnable.com/docker/python/dockerize-your-flask-application

FROM tensorflow/tensorflow:latest-py3

COPY ./requirements.txt /app/requirements.txt

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app"]
