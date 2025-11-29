# Heart Disease Prediction System 🫀

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive Machine Learning solution to predict heart disease risk using the BRFSS dataset. This project demonstrates a journey from a "scratch" implementation of XGBoost for academic understanding to a production-grade pipeline using PyCaret and Streamlit.

![Demo GIF](https://via.placeholder.com/800x400.png?text=Insert+Demo+GIF+Here)

## 🚀 Key Features

*   **Hybrid Approach**: Includes both a custom XGBoost implementation (numpy-only) and a robust PyCaret pipeline.
*   **Explainable AI**: Integrated SHAP values to explain individual predictions.
*   **Interactive UI**: Streamlit-based web application for real-time risk assessment.
*   **Ethical Analysis**: Fairlearn analysis to detect and mitigate bias.
*   **Optimized for Recall**: Tuned to minimize False Negatives in a medical context.

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

## 🧪 Testing

```bash
pytest tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed for Sprint 7 Deliverable*
