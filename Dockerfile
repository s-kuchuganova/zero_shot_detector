FROM python:3.10


WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 wget -y
RUN pip install torch torchvision transformers

RUN wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

COPY . .
RUN pip install  -r GroundingDINO/requirements.txt
RUN pip install  -r requirements.txt
RUN pip install  -q -e GroundingDINO


CMD ["python", "main.py"]