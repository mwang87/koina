FROM nvcr.io/nvidia/tritonserver:23.05-py3 AS serving-develop
# Install dependencies
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh


# set the environment variable
ENV PATH /root/miniconda3/bin:$PATH
# create a new conda environment
RUN conda create -n koina python=3.8 pip
ENV PATH /root/miniconda3/envs/koina/bin:$PATH
# activate the conda environment
RUN echo "source /root/miniconda3/bin/activate koina" >> /root/.bashrc


RUN pip install requests ms2pip psm-utils pandas pyteomics==4.6.2

RUN chmod -R 777 /root/miniconda3/envs/koina/

# Set the default command
# HEALTHCHECK --start-period=10m --interval=15s --retries=1 CMD curl --fail localhost:8501/v2/health/ready
# change user to root
USER root
# CMD ["whoami"]
CMD ["ls", "/root/miniconda3/envs/koina/bin/python" ]
# CMD ["/rot/miniconda3/envs/koina/bin/python", "/models/start.py" ]

