FROM python:3.9-slim
RUN pip install --upgrade pip
RUN pip install tensorboard
CMD ["tensorboard", "--logdir=logs", "--host", "0.0.0.0", "--port", "6006"]