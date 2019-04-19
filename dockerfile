# https://runnable.com/docker/python/dockerize-your-flask-application

FROM tensorflow/tensorflow:latest-py3

COPY ./requirements.txt /app/requirements.txt

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt


ENTRYPOINT ["python"]
CMD ["app.py"]
