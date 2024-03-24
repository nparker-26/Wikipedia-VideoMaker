FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

WORKDIR /

# Copy the requirements file and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install ffmpeg without interactive prompts
# RUN apt-get update && \
#     DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg --no-install-recommends && \
#     rm -rf /var/lib/apt/lists/*

RUN python -m nltk.downloader punkt

COPY . .

EXPOSE 5000

CMD ["python", "VideoMakerV10.py"]

# To Run:
# docker run -p 5000:5000 -v "C:\Users\robbi\OneDrive\Documents\GitHub\WikiVideo\Wikipedia-VideoMaker\app\data:/app/data" wikivideo:latest