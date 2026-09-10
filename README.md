# Banking Customer Churn Prediction & Retention Simulator

A machine-learning project that predicts customer churn and explores how changes in customer behaviour could affect predicted churn risk.

The project uses a synthetic banking dataset to investigate **which customers are most likely to leave, which characteristics are associated with churn, and how a predictive model could support customer retention decisions**.

The final output is an interactive **Streamlit retention simulator**, allowing users to enter customer characteristics, generate a churn probability, explore the factors contributing to the prediction, and test "what-if" scenarios.

> **Important:** This project uses entirely synthetic data. The predictions and scenario analysis are for demonstration purposes and should not be interpreted as causal predictions or recommendations for real customers.

---

## Project Overview

Customer churn is an important business problem for banks and other subscription-based financial services.

Retaining an existing customer can be more valuable than acquiring a new one, but identifying customers who are genuinely at risk can be difficult.

This project explores whether customer behavioural and engagement data can be used to identify customers with an increased likelihood of churn.

The project asks:

* Which customer characteristics are associated with churn?
* Can customer churn be predicted using machine learning?
* Which classification model performs best?
* How does changing the prediction threshold affect the customers identified as being at risk?
* What factors contribute to an individual customer's predicted churn risk?
* How could a predictive model be used to explore potential retention scenarios?

---

## Dataset

The dataset is **synthetically generated** for this project rather than collected from real banking customers.

It contains 20,000 customer records with variables including:

| Feature                 | Description                          |
| ----------------------- | ------------------------------------ |
| `customer_id`           | Unique customer identifier           |
| `age`                   | Customer age                         |
| `income`                | Annual income                        |
| `tenure_years`          | Length of time with the bank         |
| `account_balance`       | Current account balance              |
| `savings_balance`       | Savings account balance              |
| `monthly_app_logins`    | Number of monthly banking app logins |
| `monthly_transactions`  | Number of monthly transactions       |
| `complaints`            | Number of customer complaints        |
| `customer_satisfaction` | Customer satisfaction score          |
| `balance_change_pct`    | Percentage change in account balance |
| `churned`               | Whether the customer churned         |

The synthetic churn outcome was generated using relationships between customer behaviour and engagement. In particular, lower engagement and satisfaction, more complaints, shorter tenure and falling balances were designed to be associated with greater churn risk.

This provides a controlled dataset for demonstrating the modelling process without exposing real customer information.

---

## Project Workflow

The project follows a machine-learning workflow:

```text
Synthetic Banking Data
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Prediction & Explainability
        ↓
Churn Threshold Analysis
        ↓
Interactive Streamlit Simulator
```

---

## Exploratory Data Analysis

Initial analysis was used to investigate relationships between customer characteristics and churn.

Examples include examining differences between customers who stayed and customers who churned across:

* App engagement
* Transaction activity
* Customer satisfaction
* Complaints
* Age
* Tenure
* Account balances
* Changes in account balance

The analysis suggested meaningful differences in several behavioural variables.

For example, customers who churned tended to have:

* Lower monthly app engagement
* Lower customer satisfaction
* More complaints
* Shorter tenure
* More negative changes in account balance

These observations informed the subsequent feature engineering and modelling.

---

## Feature Engineering

Additional features were created to provide the models with more useful representations of customer behaviour.

Examples include measures designed to capture:

* Customer engagement
* Relative financial behaviour
* Interaction between behavioural variables
* Low engagement indicators

The feature-engineering process also included handling edge cases such as zero balances when calculating ratio-based features.

---

## Machine Learning Models

Three classification models were developed and compared:

### 1. Logistic Regression

Used as an interpretable baseline model.

### 2. Decision Tree

Used to investigate a non-linear tree-based approach.

### 3. Random Forest

Used as an ensemble model capable of capturing more complex relationships between features.

The models were evaluated using a held-out test set rather than relying solely on training performance.

---

## Model Evaluation

The test dataset contained 4,000 customers, including 853 customers who churned and 3,147 who did not.

The results were:

| Model               | Accuracy | Precision | Recall |   ROC-AUC |
| ------------------- | -------: | --------: | -----: | --------: |
| Logistic Regression |    0.813 |     0.635 |  0.290 | **0.802** |
| Decision Tree       |    0.802 |     0.546 |  0.272 |     0.756 |
| Random Forest       |    0.810 |     0.637 |  0.251 |     0.776 |

### Key finding

Logistic Regression achieved the highest ROC-AUC of the three models at approximately **0.802**.

This was particularly interesting because the more complex Random Forest model did not outperform the simpler Logistic Regression model on this dataset.

This demonstrates an important modelling principle:

> A more complex model is not necessarily a better model.

Model selection should consider the business objective, interpretability and appropriate evaluation metrics rather than simply choosing the most sophisticated algorithm.

---

## Why Accuracy Isn't Enough

Because the project is focused on identifying customers who may churn, simply maximising accuracy isn't necessarily the best objective.

A false negative occurs when the model predicts that a customer will stay when they actually churn.

If the purpose of the model is to help identify customers for potential retention activity, failing to identify a customer at risk could be more costly than contacting some customers who ultimately remain.

For this reason, the project explores the relationship between **precision and recall at different probability thresholds**.

For the Logistic Regression model:

| Threshold | Precision | Recall | Customers Flagged |
| --------: | --------: | -----: | ----------------: |
|       0.1 |     0.316 |  0.924 |             2,493 |
|       0.2 |     0.404 |  0.740 |             1,560 |
|       0.3 |     0.482 |  0.574 |             1,017 |
|       0.4 |     0.568 |  0.423 |               636 |
|       0.5 |     0.635 |  0.290 |               389 |
|       0.6 |     0.711 |  0.199 |               239 |
|       0.7 |     0.822 |  0.124 |               129 |

This demonstrates the trade-off involved in choosing a classification threshold.

A lower threshold identifies more potential churners and produces higher recall, but also results in more false positives.

A higher threshold produces fewer flagged customers and higher precision, but risks missing more genuine churners.

---

## Individual Customer Predictions

The project goes beyond evaluating the overall model by allowing predictions to be made for individual customers.

For an individual customer, the application produces a **predicted probability of churn** and provides an explanation of the characteristics contributing to the prediction.

For example, a customer with:

* Low app engagement
* Low transaction activity
* Multiple complaints
* Low satisfaction
* Falling account balances
* Shorter tenure

may receive a substantially higher predicted churn probability than a highly engaged and satisfied customer.

This makes the model easier to interpret from a business perspective rather than treating it simply as a classification algorithm.

---

# Interactive Retention Simulator

The final stage of the project is an interactive **Streamlit application**.

The simulator allows users to enter or adjust customer characteristics and immediately see the resulting predicted churn probability.

The application can be used to explore questions such as:

> "What happens to the model's predicted risk if this customer's engagement increases?"

or:

> "How does the predicted risk change if the customer has fewer complaints?"

This creates a simple **what-if scenario analysis tool** rather than just displaying a static model prediction.

### Example workflow

```text
Enter customer characteristics
            ↓
Create model features
            ↓
Generate churn probability
            ↓
Display customer risk
            ↓
Explore alternative scenarios
            ↓
Compare predicted risk
```

### Important limitation

The simulator demonstrates **modelled scenarios, not causal relationships**.

For example:

> "Increasing app engagement reduces the model's predicted churn probability"

does **not** mean:

> "Increasing app engagement will definitely cause this customer not to churn."

The model is trained on synthetic observational relationships, so the scenario analysis should be interpreted as an exploration of model behaviour rather than evidence that a particular intervention will cause a particular outcome.

---

## Technology Stack

### Programming & Data

* Python
* pandas
* NumPy
* Matplotlib

### Machine Learning

* scikit-learn
* Logistic Regression
* Decision Trees
* Random Forest

### Application

* Streamlit

### Development

* VS Code
* Git
* GitHub

---

## Responsible Use & Limitations

This project is intended as a **portfolio demonstration of data science and machine-learning techniques**.

There are several important limitations:

### Synthetic data

The dataset is entirely synthetic and does not represent real banking customers.

Consequently, the model's performance should not be interpreted as evidence that the same approach would achieve similar results on real banking data.

### Correlation vs causation

The model identifies patterns associated with churn. It does not establish that changing a particular customer characteristic will cause churn to increase or decrease.

### Model performance

The models achieve useful predictive performance on the synthetic test data, but there is still substantial uncertainty in individual predictions.

### Responsible decision-making

A real-world churn model should not automatically determine how customers are treated.

Predictions should instead be considered alongside appropriate human judgement, business context, fairness considerations and regulatory requirements.

---

## What I Learned

This project provided practical experience across the machine-learning workflow, including:

* Generating and working with synthetic data
* Exploratory data analysis
* Feature engineering
* Training classification models
* Comparing different modelling approaches
* Evaluating classification performance
* Understanding precision, recall and ROC-AUC
* Investigating probability thresholds
* Producing individual customer predictions
* Building interactive data-science applications with Streamlit
* Thinking about model outputs in a business context
* Considering the difference between prediction and causation
* Using Git and GitHub to manage and present a data-science project

One of the main lessons from the project was that **model performance needs to be considered in context**. A model with strong accuracy is not automatically the most useful model, particularly when the cost of false positives and false negatives differs.

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/josephnewell01/banking_churn_data_science.git
```

Move into the project directory:

```bash
cd banking_churn_data_science
```

Create and activate a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application should then open in your browser.

---

## AI-Assisted Development

AI coding tools were used during development to assist with areas such as code generation, debugging, explanation, and refinement.

Generated code was reviewed, tested and adapted during development rather than being used without verification.

This project was built as a learning exercise, with particular emphasis on understanding the underlying Python, machine-learning and statistical concepts rather than simply producing working code.

---

## Future Improvements

Potential future development could include:

* More extensive model explainability using SHAP
* Hyperparameter tuning
* Cross-validation
* More sophisticated customer segmentation
* Model monitoring and drift detection
* Automated testing
* CI/CD using GitHub Actions
* Deployment of the Streamlit application
* Testing the approach on a larger or more realistic dataset

These are deliberately left as potential extensions rather than being presented as features already implemented.

---

## Author

**Joseph Newell**

First-Class Master's degree in Mathematics, Durham University.

This project was developed as part of my portfolio to demonstrate practical skills in **Python, data analysis, statistics, machine learning and data-driven problem solving**.
