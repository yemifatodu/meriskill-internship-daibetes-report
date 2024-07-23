# meriskill-internship-daibetes-report

Based on the provided code and your request to generate a report, I will create a detailed report using the information from your code, focusing on the exploratory data analysis and the logistic regression model.

Below is a structured report based on your code:

---

# Exploratory Data Analysis and Logistic Regression Model Report

## 1. Introduction
This report presents an exploratory data analysis (EDA) and logistic regression model development for a diabetes dataset. The dataset is analyzed to understand its characteristics, distribution, and relationship between variables. A logistic regression model is then built to predict the outcome variable.

## 2. Data Loading and Overview

### Loading the Dataset
The dataset is loaded from a CSV file using Pandas.

```python
import pandas as pd

data = pd.read_csv(r"C:\Users\HP\Downloads\diabetesdataset.csv")
```

### Data Overview
The dataset is inspected to understand its structure and initial content.

```python
print("data type:", type(data))
print("data shape:", data.shape)
print(data.head())
```

### Checking for Missing Values
The dataset is checked for missing values in each column.

```python
missing_values = data.isnull().sum()
print("Missing values in each column:")
print(missing_values)
```

### Visualizing Missing Values
A heatmap is used to visualize the distribution of missing values in the dataset.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(data.isnull())
plt.show()
```

## 3. Exploratory Data Analysis

### Histograms
Histograms are created for each column to visualize the distribution of values.

```python
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(data.columns):
    data[col].hist(bins=30, ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()
```

### Box Plots
Box plots are created to identify outliers in each column.

```python
plt.figure(figsize=(15, 10))
for i, column in enumerate(data.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=data[column])
    plt.title(column)
    plt.ylabel("Value")  
plt.tight_layout()
plt.show()
```

### Removing Outliers
Outliers are removed from the dataset using the Interquartile Range (IQR) method.

```python
def remove_outliers(df):
    numeric_columns = df.select_dtypes(include=[float, int]).columns
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

cleaned_data = remove_outliers(data)
print("Original data shape:", data.shape)
print("Cleaned data shape:", cleaned_data.shape)
print(cleaned_data.head())
```

### Statistics Before and After Outlier Removal
Descriptive statistics are compared before and after removing outliers.

```python
print("Original data statistics:\n", data.describe())
print("Cleaned data statistics:\n", cleaned_data.describe())
```

## 4. Feature Engineering

### Defining Target and Features
The target variable and features are defined for the logistic regression model.

```python
target = "Outcome"
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
y_train = data[target]  
X_train = data[features]
print(y_train.head())
print(X_train.head())
```

### Splitting the Dataset
The dataset is split into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
```

### Correlation Matrix
A heatmap is created to visualize the correlation matrix of the training data.

```python
train_data = pd.concat([X_train, y_train], axis=1)
correlation_matrix = train_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix - Training Data')
plt.show()
```

## 5. Model Development

### Baseline Performance
Baseline performance is evaluated using mean absolute error (MAE).

```python
from sklearn.metrics import mean_absolute_error

y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
print("Baseline MAE:", mae_baseline)
```

### Feature Scaling and Model Training
The feature variables are scaled, and a logistic regression model is trained.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train_scaled, y_train)
```

### Making Predictions
Predictions are made on the test set.

```python
y_pred = logistic_regression_model.predict(X_test_scaled)
print("Predicted Labels:")
print(y_pred)
```

### Evaluating Model Accuracy
The accuracy of the model is evaluated.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### Feature Importances
The feature importances are displayed and visualized.

```python
feature_importance = logistic_regression_model.coef_[0]
print("Feature Importances:", feature_importance)

import matplotlib.pyplot as plt

importance = logistic_regression_model.coef_[0]
feature_importance = pd.Series(importance, index=features)
feature_importance.nlargest(8).plot(kind='barh')
plt.title('Feature Importance')
plt.show()
```

## 6. Interactive Visualization with Dash

### Setting up the Dash App
A Dash app is set up to provide an interactive visualization of the data.

```python
import dash
from dash import dcc, html
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Exploratory Data Analysis"),
    dcc.Dropdown(
        id='dropdown-column',
        options=[{'label': col, 'value': col} for col in data.columns],
        value='Pregnancies',  
        style={'width': '50%'}
    ),
    dcc.Graph(id='scatter-plot'),
])

@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('dropdown-column', 'value')]
)
def update_scatter_plot(selected_column):
    fig = px.scatter(data, x=selected_column, y='Pregnancies', color='Outcome',
                     marginal_y='violin', marginal_x='box', trendline='ols')
    fig.update_layout(title=f'Scatter plot of {selected_column} vs BMI')
    return fig

if __name__ == '__main__':
    app.run_server(port=8051, debug=True)
```

## 7. Conclusion
This report provided a comprehensive exploratory data analysis and built a logistic regression model to predict diabetes outcomes. 
---

