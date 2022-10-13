FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN python -m pip install pip==21.2.4 \
    && pip install -U --no-cache-dir \
        tqdm==4.64.0 \
        torchinfo==1.7.0 \
        matplotlib==3.5.3 \
        tensorboard==2.10.1 \
        scipy==1.9.2 \
        ml-collections==0.1.1

# RUN git clone https://github.com/NVIDIA/apex
# RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex/

