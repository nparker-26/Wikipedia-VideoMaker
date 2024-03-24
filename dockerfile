FROM cnstark/pytorch:2.0.1-py3.10.11-cuda11.8.0-ubuntu22.04
# FROM cnstark/pytorch:2.0.1-py3.9.17-ubuntu20.04
# cnstark/pytorch:2.0.1-py3.10.11-cuda11.8.0-ubuntu22.04
# cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04
# https://github.com/cnstark/pytorch-docker?tab=readme-ov-file

WORKDIR /

# Copy the requirements file and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install ffmpeg without interactive prompts
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*


RUN python -m nltk.downloader punkt

#testing with install for libcuda #TODO get this working since it is causing everything to be much slower
# https://github.com/tensorflow/tensorflow/issues/10776
# RUN DEBIAN_FRONTEND=noninteractive apt-get install build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev
# RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
# RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH python3 setup.py install 
# RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1

COPY . .

EXPOSE 5000

CMD ["python", "VideoMakerV10.py"]

# To Run:
# docker run -p 5000:5000 -v "C:\Users\robbi\OneDrive\Documents\GitHub\WikiVideo\Wikipedia-VideoMaker\app\data:/app/data" wikivideo:latest