# Banking Customer Churn Prediction & Retention Simulator

A machine-learning project that predicts customer churn using synthetic banking data and explores how predictive models can support customer retention decisions.

The project covers exploratory data analysis, feature engineering, machine-learning model comparison, probability thresholds, individual prediction explanations, and an interactive Streamlit application that allows users to simulate how changes in customer behaviour could affect predicted churn risk.

> **Note:** The dataset used in this project is synthetic and was created specifically for demonstrating a data-science workflow. It does not contain real customer information.

---

## Project Overview

Customer churn is an important problem for banks and other subscription-based businesses. Being able to identify customers who may be at higher risk of leaving can allow organisations to investigate potential issues and consider appropriate retention strategies.

This project asks:

- Which customer characteristics are associated with churn?
- Can customer churn be predicted using machine-learning models?
- Which model performs best on the synthetic dataset?
- How does changing the probability threshold affect precision and recall?
- What factors are contributing to an individual customer's prediction?
- How could a prediction model be incorporated into an interactive retention tool?

The final result is a **Customer Retention Simulator** built with Streamlit.

---

## Project Workflow

```text
Synthetic Banking Data
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Threshold Analysis
        ↓
Individual Customer Predictions
        ↓
Interactive Retention Simulator
```

This project focuses primarily on the modelling and application side of the data-science workflow. Because the dataset was generated specifically for the project, a separate data-cleaning or SQL pipeline was not required.

---

## Dataset

The dataset contains 20,000 synthetic banking customers.

| Feature | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `age` | Customer age |
| `income` | Annual customer income |
| `tenure_years` | Number of years the customer has been with the bank |
| `account_balance` | Current account balance |
| `savings_balance` | Savings account balance |
| `monthly_app_logins` | Number of monthly banking app logins |
| `monthly_transactions` | Number of monthly transactions |
| `complaints` | Number of recorded complaints |
| `customer_satisfaction` | Customer satisfaction score |
| `balance_change_pct` | Percentage change in customer balance |
| `churned` | Target variable indicating whether the customer churned |

The churn outcome was generated using relationships between customer engagement, satisfaction, complaints, tenure and changes in account balances.

Because the data is synthetic, the results should be interpreted as demonstrating a modelling workflow rather than as evidence about real banking customers.

---

## Exploratory Data Analysis

Exploratory analysis was used to understand the structure of the dataset and investigate relationships between customer characteristics and churn.

Areas explored included:

- Customer age
- Income
- Account balances
- Customer tenure
- App engagement
- Transaction activity
- Complaints
- Customer satisfaction
- Balance changes
- Differences between customers who stayed and customers who churned

The analysis helped identify behavioural patterns that could be incorporated into the predictive models.

For example, customers who churned tended to show:

- Lower app engagement
- Lower customer satisfaction
- More complaints
- Shorter tenure
- More negative changes in balances

---

## Feature Engineering

Additional features were created to give the models more useful representations of customer behaviour and financial activity.

The engineered features include:

### Total Balance

Combines current account and savings balances:

```text
total_balance = account_balance + savings_balance
```

### Savings Ratio

Measures the proportion of a customer's total balance held in savings.

### Transactions per Login

Measures transaction activity relative to app engagement:

```text
transactions_per_login =
    monthly_transactions / monthly_app_logins
```

### Complaint Indicator

Creates a binary variable indicating whether a customer has made at least one complaint.

### Low Engagement Indicator

Identifies customers with particularly low levels of digital engagement.

### Declining Balance Indicator

Identifies customers whose balances have decreased.

These features were then used alongside the original variables when training the models.

---

## Machine Learning

Three classification models were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The dataset was split into:

- **16,000 training observations**
- **4,000 test observations**

The models were evaluated on the unseen test dataset.

---

## Model Evaluation

Several metrics were used rather than relying solely on accuracy.

This is important for churn prediction because the cost of incorrectly classifying a customer as unlikely to churn may be different from incorrectly flagging a customer as at risk.

### Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.813 | 0.635 | 0.290 | 0.802 |
| Decision Tree | 0.802 | 0.546 | 0.272 | 0.756 |
| Random Forest | 0.810 | 0.637 | 0.251 | 0.776 |

Logistic Regression achieved the highest ROC-AUC and accuracy of the three models, while Random Forest produced a very similar precision.

For this project, Logistic Regression was used for the individual prediction and interactive simulation components.

---

## Why Accuracy Isn't Enough

The default classification threshold for a binary classifier is commonly 0.5.

However, the choice of threshold can have a significant effect on the balance between precision and recall.

For example, using the Logistic Regression model:

| Threshold | Precision | Recall | Customers Flagged |
|---:|---:|---:|---:|
| 0.1 | 0.316 | 0.924 | 2,493 |
| 0.2 | 0.404 | 0.740 | 1,560 |
| 0.3 | 0.482 | 0.574 | 1,017 |
| 0.4 | 0.568 | 0.423 | 636 |
| 0.5 | 0.635 | 0.290 | 389 |
| 0.6 | 0.711 | 0.199 | 239 |
| 0.7 | 0.822 | 0.124 | 129 |

This demonstrates the trade-off between identifying more potentially at-risk customers and reducing the number of customers incorrectly flagged.

For example:

- A **lower threshold** identifies more potential churners, increasing recall.
- A **higher threshold** produces fewer alerts, but risks missing more customers who eventually churn.

The appropriate threshold would therefore depend on the business objective and the relative costs of false positives and false negatives.

The Streamlit application allows this threshold to be adjusted interactively.

---

## Individual Customer Predictions

The application can generate a churn probability for an individual customer based on their characteristics.

Rather than displaying only a probability, the application also provides an explanation of the prediction.

For the Logistic Regression model, feature contributions are calculated using the customer's standardised feature values and the model's learned coefficients.

These contributions are visualised to show which features are pushing the prediction towards or away from churn.

This provides a more interpretable view of the model's output than simply displaying a single probability.

---

## Interactive Retention Simulator

The project includes a Streamlit application that allows users to interact with the trained model.

The application allows a user to enter customer characteristics such as:

- Age
- Income
- Tenure
- Account balance
- Savings balance
- App logins
- Monthly transactions
- Complaints
- Customer satisfaction
- Balance change

The application then:

1. Applies the same feature engineering used by the model.
2. Generates a predicted probability of churn.
3. Classifies the customer according to the selected probability threshold.
4. Visualises the prediction.
5. Shows the factors contributing to the prediction.
6. Provides example retention considerations based on the customer's characteristics.

### What-if Analysis

The simulator can also be used to explore hypothetical scenarios.

For example, a user could increase a customer's app engagement or satisfaction score and observe how the model's predicted churn probability changes.

This makes the model more interactive and demonstrates how predictive modelling could be incorporated into a decision-support tool.

> These scenarios demonstrate model behaviour and should not be interpreted as proof that changing a particular customer characteristic would directly cause their churn probability to change.

---

## Retention Recommendations

The application provides illustrative retention considerations based on customer characteristics.

These include:

- Reviewing service issues where complaints are present
- Following up with customers showing low satisfaction
- Encouraging useful digital engagement where app activity is low
- Reviewing customer needs where transaction activity is low
- Investigating declining balances
- Providing additional engagement during the early customer relationship

These recommendations are deliberately presented as **suggestions rather than automated decisions**.

In a real banking environment, retention actions would require additional customer information, business rules, human judgement and appropriate governance.

---

## Technology Stack

### Python

- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn

### Application

- Streamlit

### Development

- Visual Studio Code
- Git
- GitHub

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/josephnewell01/banking_churn_data_science.git
cd banking_churn_data_science
```

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
```

Activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application should then open in your browser.

---

## Responsible Use & Limitations

This project is intended as a demonstration of a data-science workflow.

There are several important limitations.

### Synthetic Data

The dataset is entirely synthetic and does not represent real banking customers.

The relationships between variables and churn were deliberately created for the purposes of the project. Consequently, the model's performance should not be interpreted as representative of a real banking environment.

### Correlation Does Not Imply Causation

The model identifies patterns associated with churn. It does not demonstrate that changing a particular characteristic would cause a customer to stay.

For example, if higher engagement is associated with lower churn in the synthetic data, this does not establish that simply increasing app logins would prevent churn.

### Predictions Should Support Rather Than Replace Decisions

In a real-world setting, a churn model should be used as one source of information rather than an automatic decision-making system.

Additional considerations would include:

- Data quality
- Model validation
- Fairness and bias
- Privacy and data protection
- Model monitoring
- Human oversight
- Business costs and benefits

---

## What I Learned

This project helped develop practical experience across several areas of the data-science workflow.

### Data Analysis

- Exploring relationships between variables
- Comparing groups
- Visualising distributions and patterns
- Interpreting statistical differences

### Feature Engineering

- Creating useful derived variables
- Handling division-by-zero situations
- Converting behavioural information into model features
- Understanding how feature design affects predictive modelling

### Machine Learning

- Training classification models
- Comparing different algorithms
- Evaluating models using multiple metrics
- Understanding precision, recall and ROC-AUC
- Working with probability thresholds

### Model Interpretation

- Understanding Logistic Regression coefficients
- Calculating individual feature contributions
- Communicating why a model produces a particular prediction

### Application Development

- Connecting a trained model to a Streamlit interface
- Creating interactive inputs
- Building what-if scenarios
- Presenting model outputs visually
- Turning a predictive model into a practical decision-support tool

### Software Development

- Structuring a Python project
- Separating functionality into modules
- Managing dependencies
- Using Git and GitHub for version control

---

## AI-Assisted Development

AI tools were used as part of the development process, particularly for coding support, debugging and exploring implementation approaches.

The project was developed iteratively, with generated suggestions being tested, adapted and integrated into the codebase.

This reflects the increasing role of tools such as GitHub Copilot and Claude in modern software and data-science workflows, while retaining human responsibility for understanding, testing and evaluating the resulting code and models.

---

## Future Improvements

Possible future extensions include:

- Hyperparameter tuning
- More advanced model comparison
- SHAP-based model explanations
- Cross-validation
- Model monitoring
- Automated testing
- CI/CD
- Additional customer segmentation
- Deployment of the Streamlit application
- Evaluation using a real-world banking dataset

These features were not required for the current version of the project but would provide opportunities to extend the project further.

---

## Author

**Joseph Newell**

First-Class Master's degree in Mathematics, Durham University.

This project was developed as part of my portfolio to demonstrate practical skills in Python, data analysis, machine learning, model interpretation and interactive data applications.