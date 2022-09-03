FROM python:3.7

EXPOSE 8501
COPY requirements.txt .
COPY app.py .


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 -y
RUN pip install -r requirements_streamlit.txt
RUN pip install --upgrade requests


CMD streamlit run app.py