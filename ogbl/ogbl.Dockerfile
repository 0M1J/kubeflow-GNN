FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN pip install tensorboardX
RUN pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install ogb
RUN pip install networkx

RUN mkdir -p /app

WORKDIR /app/src
ADD train.py /app/src/train.py

RUN  chgrp -R 0 /app/src \
  && chmod -R g+rwX /app/src

ENTRYPOINT ["python", "/app/src/train.py"]