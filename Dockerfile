FROM tensorflow/tensorflow:latest-gpu
RUN python3 -m pip install --upgrade pip
RUN pip3 install matplotlib tqdm scipy pandas pyemd  sklearn tensorflow_datasets scikit-image
RUN python3 -m pip install scikit-learn
RUN mkdir -p /home/federated_gmcc/fedlearning
RUN mkdir -p /home/federated_gmcc/experiments
RUN mkdir -p /home/federated_gmcc/data
RUN mkdir -p /home/federated_gmcc/data/training
RUN mkdir -p /home/federated_gmcc/results

ADD  README.md /home/federated_gmcc/
ADD  setup.py /home/federated_gmcc/

WORKDIR /home/federated_gmcc
RUN chmod -R ugo=rwx .
RUN pip3 install .