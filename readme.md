\# Credit Card Default Prediction using Machine Learning



\*\*Group 20 | CS6471 | University of Limerick\*\*



This project applies various machine learning algorithms to predict whether a credit card client will default on their payment in the next month. Using the UCI Credit Card Default dataset, the project explores data analysis, preprocessing techniques (including SMOTE), and model evaluation to identify the most effective predictive model.



\## 📋 Table of Contents

\- \[About the Project](#about-the-project)

\- \[Team Members](#team-members)

\- \[Dataset](#dataset)

\- \[Technologies Used](#technologies-used)

\- \[Methodology](#methodology)

\- \[Installation \& Usage](#installation--usage)

\- \[Results](#results)

\- \[Acknowledgments](#acknowledgments)



\## 📖 About the Project

Financial institutions face significant risks when extending credit. Predicting defaults is crucial for risk management, reducing financial losses, and optimizing credit approval processes. 



This project was developed as part of the \*\*CS6471\*\* module. \[cite\_start]Initially considering the Pima Indians Diabetes dataset, the team pivoted to the \*\*UCI Credit Card Default Dataset\*\* to leverage a larger, more complex dataset suitable for robust machine learning techniques\[cite: 7].



\*\*Key Objectives:\*\*

\* Perform Exploratory Data Analysis (EDA) to understand feature distributions and correlations.

\* Address class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

\* Train and evaluate multiple machine learning models: Logistic Regression, Decision Trees, Random Forest, XGBoost, and CatBoost.

\* Compare models using metrics like Accuracy, Confusion Matrices, and ROC-AUC.



\## 👥 Team Members

\*\*Group 20\*\*

\* \*\*Ben Ryan\*\* (ID: 21330786)

\* \*\*Hima Ambalagere Sudarshan\*\* (ID: 25335464)

\* \*\*Kesavarapu Vivek Reddy\*\* (ID: 25269933)



\## 📊 Dataset

\* \*\*Source:\*\* \[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) / Kaggle

\* \*\*Size:\*\* 30,000 instances, 25 variables.

\* \*\*Target Variable:\*\* `default.payment.next.month` (1 = Default, 0 = No Default).

\* \*\*Features:\*\*

&nbsp;   \* \*\*Demographics:\*\* Age, Sex, Education, Marriage status.

&nbsp;   \* \*\*Financial History:\*\* Credit limit (`LIMIT\_BAL`), Repayment status (PAY\_0 to PAY\_6), Bill amounts (BILL\_AMT1 to 6), and Previous payments (PAY\_AMT1 to 6).



\## 🛠 Technologies Used

\* \*\*Language:\*\* Python

\* \*\*Environment:\*\* Jupyter Notebook

\* \*\*Libraries:\*\*

&nbsp;   \* `pandas`, `numpy`: Data manipulation and numerical operations.

&nbsp;   \* `matplotlib`, `seaborn`: Data visualization.

&nbsp;   \* `scikit-learn`: Model building, preprocessing, and evaluation.

&nbsp;   \* `xgboost`, `catboost`: Advanced gradient boosting models.

&nbsp;   \* `imblearn`: For handling class imbalance (SMOTE).



\## ⚙️ Methodology

\[cite\_start]The project follows a structured data science pipeline\[cite: 3]:



1\.  \*\*Data Loading \& Inspection:\*\* Loading the UCI dataset and verifying structure/types.

2\.  \*\*Exploratory Data Analysis (EDA):\*\*

&nbsp;   \* Target variable distribution analysis.

&nbsp;   \* Correlation heatmaps for numerical variables.

&nbsp;   \* Visualizing relationships between demographics and default risk.

3\.  \*\*Preprocessing:\*\*

&nbsp;   \* Handling outliers using IQR capping.

&nbsp;   \* Encoding categorical variables (One-Hot/Label Encoding).

&nbsp;   \* \*\*SMOTE:\*\* Applied to balance the defaulting vs. non-defaulting classes.

&nbsp;   \* Feature Scaling using StandardScaler/MinMaxScaler.

4\.  \*\*Model Training:\*\*

&nbsp;   \* Logistic Regression (Baseline)

&nbsp;   \* Decision Tree Classifier

&nbsp;   \* Random Forest Classifier

&nbsp;   \* XGBoost

&nbsp;   \* CatBoost

5\.  \*\*Evaluation:\*\*

&nbsp;   \* Confusion Matrix (True Positives vs. False Positives).

&nbsp;   \* ROC Curve and AUC score.



\## 🚀 Installation \& Usage



1\.  \*\*Clone the repository:\*\*

&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/truly-vivek/credit-card-default-prediction.git](https://github.com/truly-vivek/credit-card-default-prediction.git)

&nbsp;   cd credit-card-default-prediction

&nbsp;   ```



2\.  \*\*Install dependencies:\*\*

&nbsp;   Ensure you have Python installed. Install the required libraries using pip:

&nbsp;   ```bash

&nbsp;   pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost imbalanced-learn

&nbsp;   ```



3\.  \*\*Run the Notebook:\*\*

&nbsp;   Open the Jupyter Notebook to view the code and execution steps.

&nbsp;   ```bash

&nbsp;   jupyter notebook Group\_20\_AI\_ML\_Notebook.ipynb

&nbsp;   ```



4\.  \*\*Load Data:\*\*

&nbsp;   Ensure `UCI\_Credit\_Card.csv` is in the same directory as the notebook.



\## 📈 Results

\* \*\*Baseline:\*\* Logistic Regression provided a baseline understanding of linear relationships.

\* \*\*Ensemble Methods:\*\* Tree-based ensemble models like \*\*XGBoost\*\* and \*\*CatBoost\*\* generally outperformed simpler models in handling non-linear relationships and feature interactions.

\* \*\*Class Imbalance:\*\* Implementing SMOTE significantly improved the model's ability to detect the minority class (Defaulters).



\## 🎓 Acknowledgments

\* \*\*University of Limerick\*\* - Module CS6471.

\* \*\*Professor Annette\*\* - For guidance and feedback during the project timeline.

\* \*\*UCI Machine Learning Repository\*\* - For providing the dataset.

