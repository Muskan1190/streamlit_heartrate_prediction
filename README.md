
# 🫀 Heart Disease Risk Prediction App

A Streamlit-based web application that predicts the likelihood of heart disease based on user-provided health inputs. The model uses logistic regression trained on a clean version of a public heart disease dataset.

## 🚀 Features

- Clean, responsive sidebar interface
- Real-time prediction using a trained machine learning model
- Radar chart visualization of key health inputs
- “Reset” button to quickly clear inputs and try again
- Fully modular code (easy to extend and maintain)

## 📂 Project Structure

```
heart_disease_app/
├── heart_app.py            # Streamlit app
├── model.py                # Model training script
├── model.pkl               # Trained logistic regression model
├── columns.pkl             # Feature columns from training
├── heart.csv               # Original dataset (optional)
├── README.md               # You're here
```

## 📋 Prerequisites

- Python 3.7+
- Libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `joblib`

Install all dependencies:

```bash
pip install -r requirements.txt
```

## ⚙️ How to Use

1. Train the model (if not already done):

```bash
python model.py
```

This generates:
- `model.pkl` (trained model)
- `columns.pkl` (list of feature columns used)

2. Run the app:

```bash
streamlit run heart_app.py
```

3. Use the sidebar to input patient health information.
4. View the prediction results and radar chart.

## 📊 Inputs Collected

| Feature               | Type            |
|-----------------------|-----------------|
| Age                   | Numeric (slider) |
| Sex                   | Radio (Male/Female) |
| Chest Pain Type       | Dropdown (ATA/NAP/ASY) |
| Cholesterol Level     | Numeric (slider) |
| Fasting Blood Sugar   | Radio (Yes/No) |
| Resting ECG           | Dropdown (LVH/ST) |
| Max Heart Rate        | Numeric (slider) |
| Exercise-Induced Angina | Radio (Yes/No) |
| Oldpeak (ST Depression) | Numeric (slider) |
| ST Slope              | Dropdown (Up/Flat/Down) |

## 📈 Output

- A binary prediction (Low Risk / High Risk)
- The associated probability
- A radar chart showing the patient's numeric input profile

## 🔄 Reset Button

Use the **"Reset All Inputs"** button to quickly clear all sidebar fields and try different values.

## ✅ Model Details

- **Algorithm**: Logistic Regression
- **Preprocessing**: One-hot encoding for categorical variables, selected feature removal
- **Target variable**: `heartdisease` (1 = has disease, 0 = no disease)

## 🧠 Credits

- Dataset Source: [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Developed with 💻 by [Muskan Patel]
