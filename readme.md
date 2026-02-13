# Credit Card Default Prediction using Machine Learning

**Group 20 | CS6471 | University of Limerick**

This project applies various machine learning algorithms to predict whether a credit card client will default on their payment in the next month. Using the UCI Credit Card Default dataset, the project explores data analysis, preprocessing techniques (including SMOTE), and model evaluation to identify the most effective predictive model.

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Team Members](#team-members)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
- [Installation & Usage](#installation--usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## 📖 About the Project
Financial institutions face significant risks when extending credit. Predicting defaults is crucial for risk management, reducing financial losses, and optimizing credit approval processes. 

This project was developed as part of the **CS6471** module. [cite_start]Initially considering the Pima Indians Diabetes dataset, the team pivoted to the **UCI Credit Card Default Dataset** to leverage a larger, more complex dataset suitable for robust machine learning techniques[cite: 7].

**Key Objectives:**
* Perform Exploratory Data Analysis (EDA) to understand feature distributions and correlations.
* Address class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
* Train and evaluate multiple machine learning models: Logistic Regression, Decision Trees, Random Forest, XGBoost, and CatBoost.
* Compare models using metrics like Accuracy, Confusion Matrices, and ROC-AUC.

## 👥 Team Members
**Group 20**
* **Ben Ryan** (ID: 21330786)
* **Hima Ambalagere Sudarshan** (ID: 25335464)
* **Kesavarapu Vivek Reddy** (ID: 25269933)

## 📊 Dataset
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) / Kaggle
* **Size:** 30,000 instances, 25 variables.
* **Target Variable:** `default.payment.next.month` (1 = Default, 0 = No Default).
* **Features:**
    * **Demographics:** Age, Sex, Education, Marriage status.
    * **Financial History:** Credit limit (`LIMIT_BAL`), Repayment status (PAY_0 to PAY_6), Bill amounts (BILL_AMT1 to 6), and Previous payments (PAY_AMT1 to 6).

## 🛠 Technologies Used
* **Language:** Python
* **Environment:** Jupyter Notebook
* **Libraries:**
    * `pandas`, `numpy`: Data manipulation and numerical operations.
    * `matplotlib`, `seaborn`: Data visualization.
    * `scikit-learn`: Model building, preprocessing, and evaluation.
    * `xgboost`, `catboost`: Advanced gradient boosting models.
    * `imblearn`: For handling class imbalance (SMOTE).

## ⚙️ Methodology
[cite_start]The project follows a structured data science pipeline[cite: 3]:

1.  **Data Loading & Inspection:** Loading the UCI dataset and verifying structure/types.
2.  **Exploratory Data Analysis (EDA):**
    * Target variable distribution analysis.
    * Correlation heatmaps for numerical variables.
    * Visualizing relationships between demographics and default risk.
3.  **Preprocessing:**
    * Handling outliers using IQR capping.
    * Encoding categorical variables (One-Hot/Label Encoding).
    * **SMOTE:** Applied to balance the defaulting vs. non-defaulting classes.
    * Feature Scaling using StandardScaler/MinMaxScaler.
4.  **Model Training:**
    * Logistic Regression (Baseline)
    * Decision Tree Classifier
    * Random Forest Classifier
    * XGBoost
    * CatBoost
5.  **Evaluation:**
    * Confusion Matrix (True Positives vs. False Positives).
    * ROC Curve and AUC score.

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/truly-vivek/credit-card-default-prediction.git](https://github.com/truly-vivek/credit-card-default-prediction.git)
    cd credit-card-default-prediction
    ```

2.  **Install dependencies:**
    Ensure you have Python installed. Install the required libraries using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost imbalanced-learn
    ```

3.  **Run the Notebook:**
    Open the Jupyter Notebook to view the code and execution steps.
    ```bash
    jupyter notebook Group_20_AI_ML_Notebook.ipynb
    ```

4.  **Load Data:**
    Ensure `UCI_Credit_Card.csv` is in the same directory as the notebook.

## 📈 Results
* **Baseline:** Logistic Regression provided a baseline understanding of linear relationships.
* **Ensemble Methods:** Tree-based ensemble models like **XGBoost** and **CatBoost** generally outperformed simpler models in handling non-linear relationships and feature interactions.
* **Class Imbalance:** Implementing SMOTE significantly improved the model's ability to detect the minority class (Defaulters).

## 🎓 Acknowledgments
* **University of Limerick** - Module CS6471.
* **Professor Annette** - For guidance and feedback during the project timeline.
* **UCI Machine Learning Repository** - For providing the dataset.
