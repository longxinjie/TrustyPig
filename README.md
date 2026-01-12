# TrustyPig – Model (MLOps)

<img width="890" height="524" alt="Screenshot 2025-12-30 171811" src="https://github.com/user-attachments/assets/3c2a206d-5275-42d6-b28b-29f200b8c796" />

---

⚠️ **Note**: This repository does **not** contain the full TrustyPig implementation.  
It only contains the **machine learning (MLOps) components** used for fraud detection.  
The main application (the Flask backend, frontend templates and deployment configuration) will be incrementally cleaned up and pushed as part of an effort to improve overall code readibility.

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
<img width="1889" height="873" alt="Screenshot 2025-12-30 171339" src="https://github.com/user-attachments/assets/8da4972a-521d-43f1-927a-4e84034cacce" />

<img width="1881" height="1007" alt="Screenshot 2025-12-30 171629" src="https://github.com/user-attachments/assets/0c1443b9-6e6f-48c9-8098-e7fb21431032" />

