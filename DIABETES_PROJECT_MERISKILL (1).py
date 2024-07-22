#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Installing the Necessary Packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
import pickle
from dash.dependencies import Input, Output

from dash import dcc, html
import dash
import plotly.express as px


# In[2]:


#loading the dataset

data = pd.read_csv(r"C:\Users\HP\Downloads\diabetesdataset.csv")
data


# In[3]:


# Data Overview

# Checking the data type
print("data type:", type (data))

# Checking the shape of the dataset (number of rows and columns)
print("data shape:", data.shape)

#Display of the first few rows of the dataset
print(data.head())


# In[4]:


# Checking for missing values in each column

missing_values = data.isnull().sum()
print("Missing values in each column:")
print(missing_values)


# In[5]:


# heatmap to visualize missing values

sns.heatmap(data.isnull())


# In[6]:


# Ploting histograms, This section creates histograms chart for each column in the dataset to visualize the distribution of values. 
#This helps in understanding the data distribution and identifying any potential issues.

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(data.columns):
    data[col].hist(bins=30, ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()



# In[7]:


# Creating a box plots for each column in the dataset to check for outliersing

plt.figure(figsize=(15, 10))

for i, column in enumerate(data.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=data[column])
    plt.title(column)
    plt.ylabel("Value")  

plt.tight_layout()
plt.show()


# In[8]:


# Identify numeric columns for the remover of outliers

def remove_outliers(df):
    numeric_columns = df.select_dtypes(include=[float, int]).columns

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

       # Calculate the lower and upper bound for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df


# In[9]:


# Removed_outliers from the dataset to be renamed cleaned_data
cleaned_data = remove_outliers(data)

# Checking the shape of the data before and after removing outliers
print("Original data shape:", data.shape)
print("Cleaned data shape:", cleaned_data.shape)


#Display of the first few rows of the new dataset
print(cleaned_data.head())


# In[10]:


# Some statistics to verify outliers removal
print("Original data statistics:\n", data.describe())
print("Cleaned data statistics:\n", cleaned_data.describe())


# In[11]:


# Exploring the cleaned data

plt.figure(figsize=(15, 10))

for i, column in enumerate(cleaned_data.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=cleaned_data[column])
    plt.title(column)
    plt.ylabel("Value")

plt.tight_layout()
plt.show()



# In[12]:


# Defining the target variable and features

target = "Outcome"
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

y_train = data[target]  
X_train = data[features]


# In[13]:


#Target variable

target = "Outcome"
y_train = data[target] 
print(y_train.head())


# In[14]:


#Feature variables

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X_train = data[features]
print(X_train.head())


# In[15]:


# Splitting the Dataset into Training and Testing Sets


X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Printing the shapes of the training and testing sets

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# In[16]:


# Putting together the features and target variable into one DataFrame
# And computing the correlation matrix

train_data = pd.concat([X_train, y_train], axis=1)
correlation_matrix = train_data.corr()

#heatmap of the correlation matrix

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix - Training Data')
plt.show()


# In[17]:


# Evaluating Baseline Performance


y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("MAE:", y_mean)
print("Baseline MAE :", mean_absolute_error(y_train, y_pred_baseline))


# In[18]:


# Feature Scaling and Model Training
# This section scales the feature variables using `StandardScaler` to normalize the data,
# The `LogisticRegression` model is initialized with a maximum of 1000 iterations and trained on the scaled training data.


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train_scaled, y_train)


# In[19]:


# Making Predictions, assuming 'scaler' is already fitted on X_train
# The `predict` method of the Logistic Regression model is then used to generate predictions for the test set. 

X_test_scaled = scaler.transform(X_test)  
y_pred = logistic_regression_model.predict(X_test_scaled)

print("Predicted Labels:")
print(y_pred)


# In[20]:


# Evaluating Model Accuracy
# A higher accuracy score generally indicates that the model is making correct predictions more often.

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[21]:


# Displaying Feature Importances
# the `coef_` attribute contains the coefficients of the features, which indicate their importance in predicting the target variable.

feature_importance = logistic_regression_model.coef_[0]
print(feature_importance)


# In[22]:


# Visualizing Feature Importances


importance = logistic_regression_model.coef_[0]
feature_importance = pd.Series(importance, index=features)
feature_importance.nlargest(8).plot(kind='barh')
plt.title('Feature Importance')
plt.show()


# In[23]:


app = dash.Dash(__name__)


# In[24]:


from dash.dependencies import Input, Output


# Initialize Dash app

app.layout = html.Div([
    html.H1("Exploratory Data Analysis"),
    
    # Dropdown for selecting columns
    dcc.Dropdown(
        id='dropdown-column',
        options=[{'label': col, 'value': col} for col in data.columns],
        value='Pregnancies',  
        style={'width': '50%'}
    ),
    
    # Scatter plot based on selected column
    dcc.Graph(id='scatter-plot'),
])

# Callback to update scatter plot based on dropdown selection
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('dropdown-column', 'value')]
)
def update_scatter_plot(selected_column):
    fig = px.scatter(data, x=selected_column, y='Pregnancies', color='Outcome',
                     marginal_y='violin', marginal_x='box', trendline='ols')
    fig.update_layout(title=f'Scatter plot of {selected_column} vs BMI')
    return fig

# Running the app
if __name__ == '__main__':
    app.run_server(port=8051, debug=True)


# In[ ]:




