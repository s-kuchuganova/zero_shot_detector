FROM python:3


WORKDIR /app

COPY main.py .
COPY requirements.txt .

RUN  apt-get update \
  && apt-get install -y wget \
  

CMD ["python", "main.py"]