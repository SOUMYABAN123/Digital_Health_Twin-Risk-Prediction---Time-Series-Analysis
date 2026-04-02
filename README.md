🧬 Digital Health Twin Risk Prediction – Time Series Analysis
📌 Overview

        This project focuses on disease risk prediction using Digital Health Twin concepts and time-series-based machine learning models. It integrates multiple advanced deep learning and factorization-based techniques to predict health outcomes such as hepatitis, stroke, and depression.

The repository combines healthcare datasets, predictive modeling, and representation learning to simulate patient-specific digital twins for improved diagnosis and risk assessment.

🚀 Key Features
1.Multiple ML/DL models for healthcare prediction
2.Time-series and tabular health data analysis
3.Digital Twin-based smart healthcare framework
4.Handling imbalanced datasets (e.g., SMOTE techniques)
5.Evaluation using AUC, loss curves, and accuracy metrics
6.Models Covered:
7.DeepFM
8.FMPNN
9.AFM / NFM / DIFM
10.Logistic Regression
11.Deep Neural Networks

🗂️ Project Structure
├── data/
│ ├── ALF_Data.csv
│ ├── healthcare-dataset-stroke-data.csv
│
├── notebooks/
│ ├── deepFMtest2.ipynb
│ ├── logistic.ipynb
│ ├── deep.ipynb
│ ├── disease.ipynb
│ ├── ProtocalNetwork.ipynb
│
├── models/
│ ├── DeepFM
│ ├── FMPNN
│ ├── ENFM
│
├── results/
│ ├── evaluation_metrics
│ ├── plots
│
└── README.md

📊 Datasets Used
Hepatitis Prediction Dataset
~6000 patient records
Features include BMI, Blood Pressure, Cholesterol, Diabetes, Hypertension, Family history
Target: Hepatitis / Acute Liver Failure
Stroke Prediction Dataset
Features: Age, gender, glucose level, BMI, smoking status, heart disease
Highly imbalanced (~5% stroke cases)
Balanced using Borderline-SMOTE2
Mini-ImageNet Dataset
100 classes with 600 images each
Used for few-shot learning experiments (Prototypical Networks)

🧠 Models Implemented
DeepFM
Combines linear models, factorization machines, and deep neural networks
Learns both low-order and high-order feature interactions
FMPNN
Hybrid of Factorization Machines and Product-based Neural Networks
Captures complex feature interactions
ENFM
Extreme Neural Factorization Machine
Designed for sparse healthcare data
Used for depression prediction
Prototypical Network
Few-shot learning model
Uses Manhattan distance instead of Euclidean
Includes dropout and pooling layers

⚙️ Installation & Setup
Clone Repository

        git clone https://github.com/SOUMYABAN123/Digital_Health_Twin-Risk-Prediction---Time-Series-Analysis.git

        cd Digital_Health_Twin-Risk-Prediction---Time-Series-Analysis

        Install Dependencies

        pip install tensorflow==2.0
        pip install deepctr
        pip install numpy pandas scikit-learn matplotlib

        Run Jupyter Notebook

        jupyter notebook

▶️ How to Run
Place dataset files in the correct directory
Open any notebook (.ipynb)
Run all cells
View outputs:
Evaluation metrics
AUC scores
Loss curves

📈 Results & Evaluation
Metrics used:
Accuracy
AUC Score
Loss
Visualizations:
Training vs validation curves
Advanced models outperform baseline machine learning methods in disease prediction tasks

🧩 Applications
Smart Healthcare Systems
Personalized Medicine
Early Disease Risk Prediction
Digital Health Twin Simulation
🔮 Future Work
Integrate real-time patient data streams
Add explainability (XAI) features
Deploy as a web-based application
Extend to multimodal data (EHR + imaging + wearable sensors)
👨‍💻 Author

Soumyaban123
GitHub: https://github.com/SOUMYABAN123

📜 License

This project is intended for academic and research purposes.
