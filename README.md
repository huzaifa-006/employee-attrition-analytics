# Employee Attrition Analytics

An end-to-end Data Science project focused on analyzing employee attrition, identifying factors associated with employee turnover, and developing machine learning models to predict employee attrition risk.

---

## Project Overview

Employee attrition is an important business problem because high employee turnover can increase recruitment costs, reduce productivity, and affect organizational performance.

This project analyzes employee data to understand attrition patterns and identify factors associated with employee turnover.

The project combines:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SQL
- SQLite
- Tableau
- Statistical Analysis
- Machine Learning

---

## Business Objectives

The main objectives of this project are:

1. Calculate the overall employee attrition rate.
2. Identify departments with higher attrition.
3. Analyze attrition across different job roles.
4. Investigate the relationship between overtime and attrition.
5. Analyze job satisfaction and employee turnover.
6. Examine salary and experience differences between employees who stayed and left.
7. Analyze promotion history and attrition.
8. Identify important features associated with attrition prediction.
9. Build and evaluate machine learning classification models.
10. Develop an interactive Tableau dashboard.
11. Perform SQL-based HR analytics.
12. Generate actionable business recommendations.

---

## Dataset

The project uses an employee dataset containing demographic, professional, compensation, workload, satisfaction, and employment information.

### Main Features

- Employee_ID
- Age
- Gender
- Department
- Job_Role
- Experience
- Education
- Salary
- Performance_Score
- Job_Satisfaction
- Overtime
- Work_Hours
- Remote_Work
- Training_Hours
- Projects
- Promotion_Last_5Yrs
- Joining_Date
- Attrition
- City
- Employment_Type
- Manager_Rating
- Sick_Leaves

### Target Variable

`Attrition`

- `Yes` → Employee left the organization
- `No` → Employee remained with the organization

---

## Project Workflow

The project follows an end-to-end Data Science workflow:

Raw Data
↓
Data Quality Assessment
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Statistical Analysis
↓
Feature Engineering
↓
Machine Learning
↓
Model Evaluation
↓
Feature Importance
↓
SQL Analysis
↓
Tableau Dashboard
↓
Business Insights
↓
Recommendations

---

## Exploratory Data Analysis

The exploratory analysis investigates:

- Employee demographics
- Department distribution
- Job-role distribution
- Salary distribution
- Experience
- Job satisfaction
- Overtime
- Work hours
- Remote work
- Promotion history
- Attrition patterns

---

## Statistical Analysis

Statistical analysis was performed to investigate relationships between employee characteristics and attrition.

The analysis includes comparisons involving:

- Job satisfaction
- Salary
- Experience
- Overtime
- Promotion history
- Employee groups

Statistical results are interpreted as evidence of association rather than proof of causation.

---

## Machine Learning

Employee attrition was treated as a binary classification problem.

### Target

```text
0 → Employee stayed
1 → Employee left