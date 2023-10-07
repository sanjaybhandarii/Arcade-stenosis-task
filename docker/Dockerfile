
FROM python:3.8-slim

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1-mesa-glx libglib2.0-0 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app \ 
    && chown user:user /opt/app 


USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python3 -m pip install --user -U pip && python3 -m pip install --user pip-tools && python3 -m pip install --user shapely
RUN python3 -m pip install --user numpy
RUN python3 -m pip install --user SimpleITK
RUN python3 -m pip install --user scikit-image
RUN python3 -m pip install --user torch
RUN python3 -m pip install --user torchvision



# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# FROM python:3.8-slim


# Install the required packages



# Install MMEngine and MMCV
RUN python3 -m pip install --user openmim && \
    mim install --user "mmengine>=0.7.1" "mmcv>=2.0.0rc4"

RUN python3 -m pip install --user mmpretrain
RUN python3 -m pip install --user mmdet

COPY --chown=user:user evaluate.py /opt/app/
COPY --chown=user:user infer.py /opt/app/
COPY --chown=user:user empt_ann.json /opt/app/
# COPY --chown=user:user imp_checkpoints /opt/app/imp_checkpoints
COPY --chown=user:user comb /opt/app/comb
COPY --chown=user:user saved_images /opt/app/saved_images


ENTRYPOINT [ "python3","evaluate.py" ]
