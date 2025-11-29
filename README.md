# Heart Disease Prediction System (NHANES Clinical Data) 🫀

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive Machine Learning solution to predict heart disease risk using clinical biomarkers from the **NHANES (National Health and Nutrition Examination Survey)** dataset. This project has evolved from using subjective survey data (BRFSS) to rigorous biomedical data.

![Demo GIF](https://via.placeholder.com/800x400.png?text=Clinical+Prediction+Demo)

## 🚀 Key Features

*   **Clinical Accuracy**: Uses objective biomarkers (e.g., Cholesterol, HbA1c) instead of self-reported survey data.
*   **Hybrid Approach**: Includes both a custom XGBoost implementation (numpy-only) for academic study and a robust PyCaret pipeline for production.
*   **Interactive UI**: Streamlit-based web application for real-time risk assessment.
*   **Ethical Analysis**: Fairlearn analysis to detect and mitigate bias.
*   **Optimized for Recall**: Tuned to minimize False Negatives in a medical context.

## 📋 Input Variables (Biomarkers)

The model utilizes the following clinical features:

*   **Age** (Years)
*   **Sex** (0 = Female, 1 = Male)
*   **BMI** (Body Mass Index)
*   **SystolicBP** (Systolic Blood Pressure, mmHg)
*   **TotalCholesterol** (mg/dL)
*   **LDL** (LDL Cholesterol, mg/dL)
*   **Triglycerides** (mg/dL)
*   **HbA1c** (Glycated Hemoglobin, %)
*   **Glucose** (Fasting Glucose, mg/dL)
*   **UricAcid** (mg/dL)
*   **Creatinine** (Serum Creatinine, mg/dL)
*   **WaistCircumference** (cm)
*   **Smoking** (0/1)
*   **Alcohol** (0/1)
*   **PhysicalActivity** (0/1)

## 📂 Project Structure

```bash
├── data/               # Data storage (Raw, Processed)
├── docs/               # Documentation (Sprint reports, Design docs)
├── models/             # Serialized models and config
├── notebooks/          # Jupyter notebooks for EDA and Experiments
├── src/                # Source code
│   ├── app.py          # Streamlit Application
│   ├── train_pycaret.py# Model Training Pipeline
│   ├── model.py        # Custom XGBoost Implementation
│   └── ...
├── tests/              # Unit tests
└── requirements.txt    # Project dependencies
```

## 🛠️ Installation in 1 Step

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

### 1. Run the App
```bash
streamlit run src/app.py
```

### 2. Retrain the Model (Optional)
```bash
python src/train_pycaret.py
```

### 3. Run Productive Audit (Health Check)
```bash
python audit_productive.py
```

## 🧪 Testing

```bash
pytest tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed for Sprint 7 Deliverable - Clinical Data Migration*
