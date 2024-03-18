FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

RUN pip install flask

CMD ["python", "VideoMakerV10.py"]
# Pull and run ffmpeg docker container
#RUN docker pull linuxserver/ffmpeg
#RUN docker run -d -p 8080:8080 linuxserver/ffmpeg

