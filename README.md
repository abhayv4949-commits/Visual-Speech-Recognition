# End-to-End Visual Speech Recognition on the GRID Corpus

A deep learning-based Visual Speech Recognition (VSR) system that transcribes spoken language directly from video by analyzing lip movements without using audio.

This project implements and compares two deep learning architectures:

* **3D CNN + Bidirectional LSTM**
* **2D CNN + Transformer Encoder**

Both models use **Connectionist Temporal Classification (CTC)** for sequence alignment and transcription.

## Project Overview

Visual Speech Recognition, commonly known as lip reading, aims to recognize spoken language using only visual information from a speaker's mouth.

The main objective of this project is to implement and compare different deep learning architectures for visual speech recognition and evaluate their performance using standard speech recognition metrics.

## Dataset

The project uses the **GRID Audio-Visual Corpus**, specifically the **Speaker 1 (S1) subset**.

* 1,000 video samples
* 1,000 corresponding alignment files
* Approximately 3 seconds per video
* 75 frames per video
* 25 FPS
* 900 training samples
* 100 test samples

The dataset contains short, structured six-word sentences with a controlled vocabulary.

**Note:** The dataset is not included in this repository.

## Preprocessing

Each video goes through the following preprocessing steps:

1. Extract video frames using OpenCV
2. Extract the mouth Region of Interest (ROI)
3. Convert frames to grayscale
4. Apply per-video normalization
5. Parse the corresponding alignment file
6. Convert transcripts into character-level tokens
7. Build a TensorFlow `tf.data` pipeline

The processed video input has the shape:

`(75, 46, 140, 1)`

The data pipeline also uses shuffling, batching and prefetching for efficient training.

## Model Architectures

### 1. 3D CNN + Bidirectional LSTM

The first model uses 3D convolutional layers to extract spatial and temporal features from the video sequence.

The extracted features are then passed through Bidirectional LSTM layers to model sequential dependencies in both forward and backward directions.

The model was fine-tuned using pre-trained checkpoint weights.

### 2. 2D CNN + Transformer Encoder

The second architecture uses a 2D CNN to extract spatial features from individual frames.

A Transformer Encoder with multi-head self-attention is then used to model temporal relationships across the video sequence.

This model was trained from scratch.

## Training

The project was implemented using:

* Python
* TensorFlow 2.19
* OpenCV
* NumPy
* Matplotlib

Training was performed on Kaggle using **dual NVIDIA T4 GPUs**.

Due to the memory requirements of processing video tensors, the training batch size was limited to 2.

## Evaluation

The models were evaluated using:

* **Character Error Rate (CER)**
* **Word Error Rate (WER)**

### Results

| Model             |    CER |    WER |
| ----------------- | -----: | -----: |
| 3D CNN + BiLSTM   | 0.6224 | 0.8833 |
| CNN + Transformer | 0.8125 | 1.0000 |

The BiLSTM model achieved better performance on the held-out test set in this experiment.

The Transformer architecture provides greater parallelization and scalability potential but requires more training data and training time to perform effectively.

## Project Pipeline

```text
Input Video
     ↓
Frame Extraction
     ↓
Mouth ROI Extraction
     ↓
Grayscale Conversion
     ↓
Normalization
     ↓
TensorFlow Data Pipeline
     ↓
 ┌───────────────────────┐
 │                       │
3D CNN + BiLSTM    CNN + Transformer
 │                       │
 └───────────┬───────────┘
             ↓
          CTC Loss
             ↓
       CTC Decoding
             ↓
       Text Transcript
```

## Key Learning

This project provided practical experience with video preprocessing, deep learning, GPU-based training, CTC-based sequence modeling and comparison of recurrent and attention-based architectures.

## Future Work

* Train on a larger and more diverse dataset
* Use dynamic facial landmark detection instead of a fixed mouth crop
* Improve Transformer performance with more training data and epochs
* Explore advanced decoding strategies
* Evaluate performance across multiple speakers
* Develop a real-time visual speech recognition application

## Technologies

`Python` `TensorFlow` `OpenCV` `NumPy` `Matplotlib` `CNN` `BiLSTM` `Transformer` `CTC` `Deep Learning`

## Author

**Abhay Verma**
MCA — VIT Chennai

---

*Developed as part of the Deep Learning course at VIT Chennai.*
