FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

WORKDIR /exp

RUN pip install joblib
RUN pip install omegaconf
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install scikit-learn
RUN pip install pytorch_lightning
RUN pip install neptune-client