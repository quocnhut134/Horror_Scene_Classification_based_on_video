import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import time
import gc
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..'))
sys.path.append(project_root_dir)

from src.demo.demo_script import setup_gpu_environment, process_video_for_horror_classification
from src.demo.predictor import load_inference_models 
from src.utils.video_audio import delete_temp_file_with_retries

def load_css(file_name):
    current_dir = os.path.dirname(__file__)
    css_file_path = os.path.join(current_dir, file_name)
    with open(css_file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load CSS
load_css("style.css")

st.set_page_config(page_title="Horror Scene Classifier")

st.title("🎬 Horror Scene Classifier System")
st.markdown("This app uses machine learning models to classify the horror elements in your input video into tags: Disturbing Visuals, Jump Scare, Pscychological tension or Calm Neutral.")
st.markdown("---")

@st.cache_resource
def load_all_inference_models(project_root):

    st.write("Initializing environment and loading AI models... Please wait! :333")
    
    setup_gpu_environment() 
    
    trained_model_name = "random_forest" 
    classifier, scaler_inference, label_encoder_inference = load_inference_models(project_root, trained_model_name)
    
    if classifier and scaler_inference and label_encoder_inference:
        st.success("Complete the preparation process! ♡´･ᴗ･`♡")
        return classifier, scaler_inference, label_encoder_inference
    else:
        st.error("The system is faulty in some model! ¯\_(ツ)_/¯")
        st.stop() 
        return None, None, None

classifier_model, scaler_for_inference, label_encoder_for_inference = load_all_inference_models(project_root_dir)


uploaded_file = st.file_uploader("Upload your video here!", type=["mp4", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file) 

    if st.button("Classify!"):
        if uploaded_file is not None:
            temp_upload_dir = os.path.join(project_root_dir, 'src', 'demo', 'temp_uploaded_videos')
            os.makedirs(temp_upload_dir, exist_ok=True)
            
            temp_video_path = os.path.join(temp_upload_dir, uploaded_file.name)
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                        
            try:
                with st.spinner("Analyzing video segments to extract horror features..."):

                    final_category = process_video_for_horror_classification(
                        video_path=temp_video_path,
                        segment_duration=10, 
                        model_name="random_forest" 
                    )
                
                st.success("Complete classification!")
                st.metric(label="Your video tag is:", value=final_category)
                        
            except Exception as e:
                st.error(f"An error occurred during classification: {e}")
                st.exception(e) 
            finally:
                if os.path.exists(temp_video_path):
                    delete_temp_file_with_retries(temp_video_path)
                if os.path.exists(temp_upload_dir) and not os.listdir(temp_upload_dir):
                    os.rmdir(temp_upload_dir)
                gc.collect() 

st.markdown("---")
st.markdown("This application is a product of Computational Thinking project made by Duong Quoc Nhut, Tran Bao Tran, Tran Thai Anh Phong and Nguyen Hoang Kha.")
 


