# Real-Time Facial Expression Recognition

This project implements a facial expression recognition system using machine learning techniques and computer vision. The system can classify facial expressions such as happiness, sadness, anger, and more. It includes both model training and a real-time application.

## Features
- Real-time facial expression detection using a webcam
- Model training included in the `training.ipynb` notebook
- Model trained on a GPU-powered environment for faster processing
- Integration of computer vision libraries such as OpenCV and TensorFlow

## Project Overview

The project has two main parts:
1. **Model Training:**
   - The model was trained on the **FER2013 dataset**, which contains over 35,000 labeled facial expression images in 7 different categories: **Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, and **Neutral**.
   - The dataset was used to train the VGG16-based model in a Kaggle environment with GPU support.
   - TensorFlow and Keras were used to implement the VGG-based deep learning model.

3. **Real-Time Application:**
   - A trained model is used to detect facial expressions in real-time through a webcam.
   - OpenCV is used for capturing video and preprocessing input frames.


## How to Run

### Prerequisites
Ensure you have Python installed and set up a virtual environment. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 1. Clone the Repository
```bash
git clone https://github.com/begumarici/Facial-Expression-Recognition.git
cd Facial-Expression-Recognition
```

### 2. Train the Model
   - Open and run the training.ipynb notebook to train the facial expression recognition model. The training process uses the FER2013 dataset and the VGG architecture.


### 3. Run the Real-Time Application
```bash
python src/real_time.py
```

The application will start your webcam and classify facial expressions in real time.

## File Structure
```
Facial-Expression-Recognition/
├── README.md              # Project documentation
├── src/
│   ├── real-time.py          # Real-time detection script
│   └── train-model.ipynb     # Notebook for model training
├── requirements.txt          # Required libraries
```

## Technologies Used
- **Languages:** Python
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib
- **Tools:** GPU for model training

## Demo
Coming soon

## Acknowledgments
This project was developed to explore machine learning and computer vision techniques for real-time applications.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
