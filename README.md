# 🧠 Diabetes Risk Predictor (ANN-Based)

Diabetes is a **chronic metabolic disorder** characterized by prolonged elevated blood glucose levels.
Research indicates that individuals with diabetes are approximately **1.8 times more likely** to die from any cause, and over **2.3 times more likely** to die from vascular disease compared to those without the condition.

From an economic perspective, the burden is equally significant — the global cost of diabetes exceeds **USD 1.3 trillion annually**, accounting for nearly **1.8% of global GDP**.

To address this growing crisis, **early detection** and **intervention** are critical.
This project examines the intersection of **AI and healthcare** by developing a predictive model for **diabetes risk** using machine learning.

---

## 🚀 Project Overview

* ✅ **Model Type**: Deep Neural Network (11 layers total)
* ✅ **Model Selection**: ANN Model A — best performing model out of **8 models** tested
* ✅ **Input**: Health and lifestyle data (21 categorical features)
* ✅ **Output**: Binary classification — *Diabetes* vs. *Non-Diabetes*
* ✅ **Preprocessing**: One-hot encoding, feature binning
* ✅ **Frameworks**: TensorFlow/Keras, pandas, scikit-learn, Flask
* ✅ **Deployment**: Full-stack web app (Flask backend, HTML frontend)

---

## 📊 Dataset

* **Source**: CDC’s **Behavioral Risk Factor Surveillance System (BRFSS)**
* **Size**: \~253,680 survey responses (cleaned + balanced dataset)
* **Features**: 21 demographic, health, and behavioral attributes — such as **age, BMI, smoking status, blood pressure, physical activity, healthcare access**.
* **Target**: `Diabetes_binary`

  * 0 = No diabetes
  * 1 = Prediabetes or positive diabetes diagnosis

---

### Features List:

* MentHlth
* PhyHlth
* BMI
* HighBP
* HighChol
* CholCheck
* Smoker
* Stroke
* HeartDiseaseorAttack
* PhysActivity
* Fruits
* Veggies
* HvyAlcoholConsump
* AnyHealthcare
* NoDocbcCost
* GenHlth
* DiffWalk
* Sex
* Age
* Education
* Income

📎 [View processed dataset](https://drive.google.com/file/d/1oAdz8yzwIxZaj8vnneNy6L6QCKL9gQ3_/view?usp=sharing)

---

## 🧠 ANN Architecture

| Layer Type     | Details                                         |
| -------------- | ----------------------------------------------- |
| Input Layer    | Matches one-hot encoded input                   |
| Hidden Layers  | 10 hidden layers, 30 neurons each               |
| Activation     | ReLU (hidden), Sigmoid (output)                 |
| Regularization | L1 and L2, both set to `0.001`                  |
| Optimizer      | SGD with learning rate `0.01`, momentum `0.9`   |
| Training       | 50 epochs, batch size `32`, with early stopping |

The ANN is capable of learning both **linear** and **non-linear** relationships between features and diabetes outcome.

---

## ✅ Performance Summary

As part of this project, we trained and evaluated **8 different models**, including:

* **Logistic Regression (L2-regularized)**
* **Random Forest**
* **XGBoost**
* **LightGBM**
* **Bayesian Network**
* **Multiple ANN variants (ANN A, B, C, etc.)**

After extensive testing on consistent **metrics** — accuracy, precision, recall, F1 score, false negative rate — we selected **ANN Model A** as the **best overall model**:

* **Highest test accuracy**
* **Lowest false negative rate** — critical for medical applications
* **Strong precision and recall balance**
* **Robust generalization** to unseen test data

---

## 🌐 Web Application

### YouTube Demo Link:
[Demo](https://youtu.be/A3SVoX7vvxk?feature=shared)

### Features:

* Interactive **Web Form** for user input
* Real-time **Prediction**: Diabetes vs. Non-Diabetes + probability
* REST API `/api/predict` for programmatic access

### Architecture:

| Component     | Technology                                         |
| ------------- | -------------------------------------------------- |
| Backend       | Flask (Python)                                     |
| Model Serving | TensorFlow/Keras                                   |
| Frontend      | HTML templates (Jinja2) — React planned for future |
| API           | `/api/predict` (JSON POST endpoint)                |

---

## 🛠️ Running the App

1️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

2️⃣ Place model:

```
my_ann_model.h5 → project root directory
```

3️⃣ Place dataset:

```
data/DataSet.csv
```

4️⃣ Run app:

```bash
python app.py
```

5️⃣ Access app:

```
http://localhost:5000/
```

---

## 🎯 Conclusion and Discussion

This project demonstrates how a **thoughtfully designed ML system** can support **early detection of diabetes** — potentially transforming how individuals and healthcare providers approach **chronic disease prevention**.

We trained and compared **8 different models** — ultimately selecting **ANN Model A** for deployment, due to its:

* ✅ **High accuracy**
* ✅ **Balanced precision**
* ✅ **Lowest false negative rate** — key for healthcare use cases

### Real-world deployment will require addressing:

* 🔐 **Data privacy and security**
* ⚖️ **Fairness and bias mitigation**
* 🏥 **Clinical validation** to demonstrate reliability
* 🤝 **Transparency and trust** for both users and healthcare professionals

---

## 🚀 Planned Future Work

* 🌐 **React-based frontend**
* 📊 **Interactive visualizations** (risk factor analysis, feature importance)
* ☁️ **Cloud deployment** (Heroku, AWS, etc.)
* 🔄 **Model versioning** and retraining pipelines
* 📈 **Ongoing clinical evaluation**

---

**Summary:**
After evaluating **8 candidate models**, this project demonstrates that a properly tuned **deep learning model (ANN Model A)** can effectively model **categorical health data** — providing accurate, real-time **diabetes risk predictions** to support **proactive healthcare** and public health goals.

---

