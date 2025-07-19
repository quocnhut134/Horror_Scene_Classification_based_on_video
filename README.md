# Horror Scene Classification based on video

## Table of contents
* [General Information](https://github.com/quocnhut134/Horror_Scene_Classification_based_on_video?tab=readme-ov-file#general-information)
* [Technologies](https://github.com/quocnhut134/Horror_Scene_Classification_based_on_video?tab=readme-ov-file#technologies)
* [Fine-tuning YoloV8](https://github.com/quocnhut134/Horror_Scene_Classification_based_on_video?tab=readme-ov-file#fine-tuning-yolov8)
* [Relabeling](https://github.com/quocnhut134/Horror_Scene_Classification_based_on_video?tab=readme-ov-file#relabeling)
* [Training Classifier Model](https://github.com/quocnhut134/Horror_Scene_Classification_based_on_video?tab=readme-ov-file#training-classifier-model)
* [Demo](https://github.com/quocnhut134/Horror_Scene_Classification_based_on_video?tab=readme-ov-file#demo)
* [Demo with Streamlit](https://github.com/quocnhut134/Horror_Scene_Classification_based_on_video?tab=readme-ov-file#demo-with-streamlit)

## General Information
This project presents a robust system for classifying horror scenes within videos based on a combination of visual and auditory features. It aims to accurately identify segments containing "Disturbing Visuals," "Jump Scare," "Psychological Tension," or "Calm Neutral" content.

Utilizing a machine learning pipeline, this repository provides tools for:

* Feature Extraction: Leveraging state-of-the-art models like YOLOv8 for visual analysis (e.g., violence detection) and YAMNet for audio analysis (e.g., screams, tension-inducing sounds).

* Model Training & Evaluation: Demonstrating the process of training various classical machine learning classifiers (e.g., Logistic Regression, Random Forest, Gradient Boosting, SVC) on extracted features and evaluating their performance.

* Inference & Prediction: Providing a streamlined process to classify segments of new videos and aggregate these predictions for overall video categorization.

This system is particularly useful for content moderation, media analysis, and enhancing viewer control over potentially disturbing material.
## Technologies
This project is created with:
* opencv-python version: 4.12.0.88
* ultralytics version: 8.3.167
* scikit-learn version: 1.7.0
* tqdm version: 4.67.1
* pandas version: 2.3.1
* matplotlib version: 3.10.3
* seaborn version: 0.13.2
* ipywidgets version: 8.1.5
* torch version: 2.5.0
* torchaudio version: 2.7.1
* torchvision version: 0.22.1
* tensorflow version: 2.19.0
* tensorflow-hub version: 0.16.1
* librosa version: 0.11.0
* moviepy version: 2.2.1
* joblib version: 1.5.1
* numpy version: 2.1.2
* spyder version: 6.0.7
* streamlit version: 1.47.0

You can install all packages above by installing requirement.txt in the root directory of this project with pip:

```
pip install -r requirement.txt
```

**Warning:** Beacause of deploying the streamlit app, requirement.txt doesn't contain spyder and ipywidget. You need to install them separately after installing packages in requirement.txt.

Also, this project includes using GPU to relabel data, fine-tune **YOLOv8**, preprocess data and train **Classifier Model**. So, please ensure that you installed **CUDA Toolkit** and its suitable **CuDNN** version.

This project uses:
* **CUDA Toolkit** version: 11.8
* **CuDNN** version: 9.10.2 (Released June 2025)

Besides, this project also use ffmpeg, a cross-platform tool for splitting videos into smaller segments, extracting audio from videos, and preparing video/audio data for feature extraction and horror scene classification.

**FFMPEG Installation:**
* You need to download the [ffmpeg](https://ffmpeg.org/download.html) version being suitable to your device
* **For Windows:**
    * After downloading, paste the bin folder of ffmpeg to PATH - Environment Variables.
    * Restart your Windows device to appy changes.
* **For macOS/Linux:**
    * Open Terminal.
    * Edit your shell configuration file (Ex: ~/.bashrc, ~/.zshrc or ~/.profile).
    * Add this at the end of your file.
    ```
    export PATH=$PATH:/path/to/ffmpeg/bin
    ```
    * Save file and run this to apply changes:
    ```
    source ~/.bashrc  # or ~/.zshrc or ~/.profile
    ```
## Fine-tuning YoloV8
* This project uses **YOLOv8** to extract features being related to horror factor in images. However, because of the lack of project-related horror labels in **YOLOv8**  and horror image data to fine-tune it, I used [Real Life Violence and Non-Violence Data](https://www.kaggle.com/datasets/karandeep98/real-life-violence-and-nonviolence-data) to set a temporary standard to determine horror factor in images by calculating the violence level.
* You can find related code and finetuning result in root directory/src/YoloV8 of this project.
* You can also fine-tune again by run this:

    ```
    cd ../src/YoloV8
    python main.py
    ```
    
## Relabeling
* Because of the lack of horror video data, I used [vidharm](https://github.com/vidharm/vidharm?tab=readme-ov-file) which contains film trailers and labels are the age limit for viewers ("bt","7","11" and "15"). I relabel this data by mapping the original label domain to new label domain ("Disturbing Visuals," "Jump Scare," "Psychological Tension," or "Calm Neutral") by extracting image and sound features and calculating their scores.
* You can find related code in root directory/src and relabeled data in root directory.
* You can also rerun labeling by run this:
    ```
    cd ../src
    python main.py
    ```
    
## Training Classifier Model

This process contains below steps:
* **Multimodal Feature Extraction**: Automatically extract features such as violence score (**fine-tuned YOLOv8 model**) and multiple audio features including brightness standard deviation, scream score, silence score, noise score, music score, hum score, speech score and tension score (**YAMNet model**)
* **Classification using Machine Learning**: Uses machine learning algorithms like Logistic Regression, Random Forest, Gradient Boosting and SVC to classify movie scenes.
* **Comprehensive Evaluation**: Evaluate model performance using metrics such as F1-score, accuracy, and confusion matrix.
* **Customize and save models**: Supports hyperparameter tuning (**GridSearchCV**) and saving/loading trained models.
* **Long Video Classification**: Apply threshold classification method to determine horror genre for entire long video.
To run training, please run this:
```
cd .. #Go to root directory of this project
python -m src.train_model.train_script
```

## Demo

After passing stages above, you can classify horror video by running this:
```
cd .. #Go to root directory of this project
python -m src.demo.demo_script
```

**Warning 1:** You need to install ffmpeg that I presented above and uncomment this line in demo_script.py (Only if your ffmpeg PATH is the same as mine):
```
    # os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg\bin\ffmpeg.exe" 
```
**Warning 2:** You need to add your horror video link to code in demo_script.py first
## Demo with Streamlit

You can also visualize this classify model on Streamlit, please run this:
```
cd .. #Go to root directory of this project
streamlit run src/demo/streamlit/streamlit_app.py
```

On Streamlit platform of this product, you can upload horror video from your local device to classify it.

<img width="1913" height="858" alt="Image" src="https://github.com/user-attachments/assets/ed2037ef-8881-4c84-90d9-b55fcd1797cd" />
