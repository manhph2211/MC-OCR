FROM python:3.7

EXPOSE 8501
COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 -y
RUN pip install -r requirements.txt
RUN pip install --upgrade requests

CMD bash setup.sh
CMD streamlit run deployment/streamlit/app.py