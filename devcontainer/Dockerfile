# Base pytorch image with GPU suport
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Basic package update
RUN apt-get update && apt upgrade -y

# The flag is set to avoid unnecesarry questions during installation
# Example: It asks to set timezone during instalation of python.
ARG DEBIAN_FRONTEND=noninteractive

# Install VIM (For Convenience)
RUN apt-get install -y vim

RUN pip install python-sat[aiger,approxmc,cryptosat]
RUN pip install pysat

## Huggingface installation
#RUN pip install transformers
#RUN pip install datasets
#RUN pip install --upgrade huggingface_hub