# 🧠 Diabetes Risk Predictor (ANN-Based)

This project is a machine learning-powered **Diabetes Risk Predictor** designed to classify individuals based on health-related features using a custom-built **Artificial Neural Network (ANN)**. The current model is trained on fully categorical and one-hot encoded data, and future plans include deploying it via a full-stack web application with interactive visualizations.

---

## 🚀 Project Overview

- ✅ **Model Type**: Deep Neural Network (11 layers)
- ✅ **Input**: Health and lifestyle data (all features categorical)
- ✅ **Output**: Binary classification (diabetes vs. non-diabetes)
- ✅ **Preprocessing**: One-hot encoding and custom binning
- ✅ **Frameworks**: TensorFlow / Keras, pandas, scikit-learn
- ✅ **Planned Deployment**: Full-stack web interface (React + Flask or Django)

---

## 📊 Dataset

All features in the dataset are **categorical**, either originally or after binning. For example:

- `HighBP`: 0 = No high blood pressure, 1 = Yes
- `BMI`: Binned into 5 categories based on health guidelines
- `MentHlth`, `PhysHlth`: Converted to classes using logical thresholds

📎 [View processed dataset](https://drive.google.com/file/d/1oAdz8yzwIxZaj8vnneNy6L6QCKL9gQ3_/view?usp=sharing)

To preserve the categorical meaning of features, we applied **one-hot encoding** instead of numerical scaling (standardization).

---

## 🧠 ANN Architecture

| Layer Type       | Details                       |
|------------------|-------------------------------|
| Input Layer      | Matches one-hot encoded input |
| Hidden Layers    | 9 layers, 30 neurons each     |
| Activation       | ReLU (hidden), Sigmoid (output) |
| Regularization   | L1 and L2, both set to 0.001  |
| Optimizer        | SGD with momentum = 0.9       |

The ANN is capable of learning both **linear** and **non-linear** relationships between features and diabetes outcome.

---

## ✅ Performance Summary

- **High test accuracy**
- Strong **precision**, **recall**, and **F1-score**
- Robust performance across various input conditions
- Generalizes well on unseen test data

> The model’s success confirms that a properly encoded, fully categorical dataset can be effectively modeled using deep learning techniques.

---

## 🛠️ Coming Soon: Full-Stack Deployment

We plan to develop a user-facing **web application** for interactive prediction and exploration. Features will include:

- 🌐 Frontend: XXXXXXXXXXXXXXXXXXXX
- 🧠 Backend: XXXXXXXXXXXXXXX serving the trained model
- 📈 Visualizations: Risk factor analysis, model outputs, confidence metrics
- 📤 User input: Form-based feature selection

---


