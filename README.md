# TrustyPig – Model (MLOps)

⚠️ **Note**: This repository does **not** contain the full TrustyPig implementation.  
It only contains the **machine learning (MLOps) components** used for fraud detection.  
The main application (Flask backend, frontend templates, deployment configs, etc.) is kept separately.

---

## 📌 What’s Inside
This repo includes:
- `train.py` → training pipeline for fraud detection model  
- `export_data.py` → data preprocessing and export script  
- `feature_columns.csv` → reference for features used in training  
- `fraud_model.pkl`, `model.pkl` → trained model artifacts (example)  
- `Fraud Detection Model.ipynb` → exploratory notebook for model development  

---

## 🚀 MLOps Notes
- The model is trained using **XGBoost** and saved as `.pkl` files for deployment.  
- Data is preprocessed and exported with consistent feature engineering pipelines.  
- For production deployment:
  - Models should be stored in **cloud storage** (e.g., Firebase Storage, S3, GCS)  
  - Deployment app (`app.py`) will load the model dynamically at runtime.  
- Heavy artifacts and datasets should be **gitignored** and not versioned directly here.  

---
