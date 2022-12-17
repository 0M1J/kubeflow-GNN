FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN pip install tensorboardX
RUN pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install ogb
RUN pip install networkx

RUN mkdir -p /opt/mnist

WORKDIR /opt/mnist/src
ADD mnist.py /opt/mnist/src/mnist.py
ADD train.py /opt/mnist/src/train.py

RUN  chgrp -R 0 /opt/mnist \
  && chmod -R g+rwX /opt/mnist

ENTRYPOINT ["python", "/opt/mnist/src/train.py"]