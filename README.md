# Heart Disease Prediction System 🫀

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive Machine Learning solution to predict heart disease risk using **NHANES (National Health and Nutrition Examination Survey)** clinical data. This project has evolved from survey-based analysis to a robust system based on **Clinical Biomarkers**, providing more objective and medically relevant predictions.

![Demo GIF](https://via.placeholder.com/800x400.png?text=Clinical+Heart+Prediction+Demo)

## 🚀 Key Features

*   **Clinical Focus**: Shifts from subjective survey responses to objective clinical measurements.
*   **Hybrid Approach**: Includes both a custom XGBoost implementation (numpy-only) and a robust PyCaret pipeline.
*   **Explainable AI**: Integrated SHAP values to explain individual predictions.
*   **Interactive UI**: Streamlit-based web application for real-time risk assessment.
*   **Ethical Analysis**: Fairlearn analysis to detect and mitigate bias.
*   **Optimized for Recall**: Tuned to minimize False Negatives in a medical context.

## 🩺 Clinical Features (NHANES)

The model uses the following biological and lifestyle markers:

*   **Age** (Years)
*   **Sex** (Male/Female)
*   **BMI** (kg/m²)
*   **SystolicBP** (mmHg)
*   **TotalCholesterol** (mg/dL)
*   **LDL** (mg/dL)
*   **Triglycerides** (mg/dL)
*   **HbA1c** (%)
*   **Glucose** (mg/dL)
*   **UricAcid** (mg/dL)
*   **Creatinine** (mg/dL)
*   **WaistCircumference** (cm)
*   **Smoking** (Yes/No)
*   **Alcohol** (Yes/No)
*   **PhysicalActivity** (Yes/No)

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

## 🛠️ Installation

Ensure you have Python 3.10 installed.

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

## 🧪 Testing

```bash
pytest tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed for Sprint 7 Deliverable*
