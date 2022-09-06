import os
from os import environ as env
from tempfile import TemporaryDirectory
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import requests

import streamlit as st
import sys
sys.path.append("../../")
from background_subtraction.maskrcnn.save_img import remove_bg 
from text_detection.craft.main import detect   
from text_recognition.main import recoginize
# from key_info_extraction.tools.inference import get_key


st.set_page_config(layout="wide", page_icon="🖱️", page_title="Interactive table app")
st.title("👨‍💻 Invoice Key Information Extraction")


def app():
    upload_file = st.file_uploader(label="Pick a file", type=["png", "jpg", "jpeg"])
    print(upload_file)
    image = None
    if upload_file is not None:
        filename = upload_file.name
        filetype = upload_file.type
        filebyte = bytearray(upload_file.read())

        # cvt byte to image
        image = np.asarray(filebyte, dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with st.spinner("🤖 Removing background... "):
            remove_bg(image)
            
        with st.spinner("🤖 Detecting angle... "):
            detect(upload_file.replace("original","bg_sub"))

        with st.spinner("🤖 Rotating invoice... "):
            detect(upload_file.replace("original","bg_sub"))

        with st.spinner("🤖 Detecting texts... "):
            remove_bg(upload_file.replace("original","bg_sub"))

        with st.spinner("🤖 Recoginizing texts... "):
            recoginize("data/demo/text_detection/data.json")

        # image = draw(image, tables)
        tab1, tab2, tab3 = st.tabs(
            ["Tabular Data", "Table Visualization", "OCR Visualization"]
        )

        with tab1:
            st.header("Tabular Information")
            st.image(image)
        st.balloons()


if __name__ == "__main__":
    app()