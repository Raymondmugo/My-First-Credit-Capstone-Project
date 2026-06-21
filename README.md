# Credit Risk Prediction Engine

End-to-end XGBoost credit risk model optimizing loan approvals via SHAP explainability and a live Streamlit web application.

## Overview
This project is a machine learning solution designed to predict credit default risk and optimize portfolio profitability. By upgrading from a baseline Logistic Regression model to a cost-sensitive XGBoost classifier, the system drastically reduces false negatives (missed defaults) while maintaining transparent, explainable decision-making criteria.

## Business Impact
* Capital Protection: Caught an additional 450 hidden defaults compared to the baseline model, representing an estimated $4.5M in prevented write-offs.
* Revenue Recovery: Safely approved 1,420 previously misclassified good customers, securing approximately $2.1M in future interest payments.
* Total ROI: Added an estimated $6.6M in net value to the loan portfolio.

## Tech Stack
* Machine Learning: XGBoost, Scikit-Learn (Logistic Regression, StandardScaler)
* Explainable AI: SHAP (SHapley Additive exPlanations)
* Front-End Deployment: Streamlit
* Data Manipulation: Pandas, NumPy

## Key Features
* Cost-Sensitive Learning: Mathematically penalized the algorithm for missing bad loans using strict class weight ratios, prioritizing capital preservation.
* Data Leakage Prevention: Rigorously scrubbed post-origination features (e.g., debt settlement flags) to ensure the model only evaluates day-one application data.
* AI Explainability: Integrated SHAP Waterfall and Summary plots to eliminate the "black box" effect, allowing loan officers to generate compliant Adverse Action Notices.
* Interactive UI: Deployed a front-end Streamlit dashboard allowing risk teams to input specific applicant metrics and receive live probability scores and risk breakdowns.



   
