# CIFAR-10 Image Classification using CNNs

This project implements image recognition with **Convolutional Neural Networks (CNNs)** on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The pipeline is built with **Keras (TensorFlow backend)** and compares two models:
- **Model 1:** Trained without data augmentation  
- **Model 2:** Trained with data augmentation  

## 🚀 Features
- **Data preprocessing:** one-hot encoding of labels, normalization, and scaling of images.  
- **CNN model architecture:** convolutional layers with ReLU activation, pooling, dropout, and fully connected layers.  
- **Training evaluation:** accuracy/loss curves plotted for comparison between augmented vs non-augmented models.  
- **Model deployment:** trained models are saved and later reloaded for inference on CIFAR-10 test set and custom user images.  

## 📊 Results
- Model without augmentation achieved **77.25% accuracy**.  
- Model with augmentation achieved **78.04% accuracy**, showing improved generalization.  

## 🛠️ Tech Stack
- Python, Keras, TensorFlow  
- NumPy, Matplotlib  
- CIFAR-10 Dataset  

## 📂 Project Structure
- `image_recognition.py` → Data preprocessing, CNN training, evaluation  
- `my_image_recognition.py` → Load saved model, predict on test set & user images  

## 🔮 Future Work
- Test with deeper architectures (ResNet, DenseNet)  
- Experiment with transfer learning for higher accuracy  
- Deploy model via Flask/Streamlit for interactive predictions  

---
