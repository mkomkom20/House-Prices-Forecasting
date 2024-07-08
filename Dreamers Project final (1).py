#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib as mpl          
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
import customtkinter as ctk
from tkinter import ttk
from tkinter import messagebox


# In[3]:


# 1.- DATA COLLECTION

#Let's start by loading and exploring the data 
# Set the directory location
path = 'C:/Users/Jmcharles/Desktop/MBDA/BSAD/Final Project/'

# Load the housing dataset into a Pandas DataFrame
Data = pd.read_excel(path + 'Housing_information.xlsx')
Data.head()


# In[4]:


Data.info()


# In[5]:


# Check for missing values
Data.isnull().sum()


# In[6]:


# Read the data description file using pandas
data_description = pd.read_excel(path + 'data_description.xlsx')

# Split the "Variable: Description" column into two separate columns
data_description[['Variable', 'Description']] = data_description['Variable:Description'].str.split(':', n=1, expand=True)

# Drop the original "Variable: Description" column
data_description.drop(columns=['Variable:Description'], inplace=True)

# Display the updated DataFrame
data_description.head()


# In[7]:


# Convert variable names from data description file to lowercase
data_description['Variable'] = data_description['Variable'].str.lower()

# Rename columns to 'Variable' and 'Description'
data_description.columns = ['Variable', 'Description']

# Convert column names in the Data DataFrame to lowercase
Data.columns = Data.columns.str.lower()

# Define the data dictionary as an empty list
data_dict = []

# Iterate over each column in the DataFrame
for column in Data.columns:
    # Define the data type of the variable
    data_type = Data[column].dtype
    
    # Identify potential values or range
    potential_values_range = Data[column].unique()
    
    # Identify missing values
    missing_values = Data[column].isnull().sum()
    
    # Find the description for the current variable
    description = data_description[data_description['Variable'] == column]['Description'].values
    
    # If description is not available, set it to empty string
    if len(description) == 0:
        description = ''
    else:
        description = description[0]
    
    # Append variable information to data dictionary
    data_dict.append({
        'Variable': column,
        'Description': description,
        'Data Type': data_type,
        'Potential Values Range': potential_values_range,
        'Missing Values': missing_values
    })

# Convert data_dict to a DataFrame
data_dict_df = pd.DataFrame(data_dict)

# Display the first few rows of the DataFrame
data_dict_df.head()


# In[8]:


# Save the DataFrame to a CSV file
data_dict_df.to_csv('C:/Users/Jmcharles/Desktop/MBDA/BSAD/Final Project/house_dictionary.csv', index=False)


# In[9]:


Dict=pd.read_csv(path + 'house_dictionary.csv')
Dict.head()


# In[10]:


# 2.- DATA CLEANING
    
# Print missing values percentage for all the columns
for column in Data.columns:
    missing_percentage = (Data[column].isnull().sum() / len(Data[column])) * 100
    print(f"Percentage of missing values in {column}: {missing_percentage:.2f}%")

# Calculate missing values percentage after printing
missing_percentage = (Data.isnull().sum() / len(Data)) * 100

# Filter columns with missing percentage greater than 10%
columns_to_drop = missing_percentage[missing_percentage > 10].index

# Drop the columns
Data.drop(columns_to_drop, axis=1, inplace=True)


# In[11]:


# Compute missing values for numerical columns with their median
numerical_columns = Data.select_dtypes(include='number').columns

for column in numerical_columns:
    median_value = Data[column].median()
    Data[column].fillna(median_value, inplace=True)


# In[12]:


# Compute missing values for categorical columns with their mode
categorical_columns = Data.select_dtypes(include='object').columns

for column in categorical_columns:
    mode_value = Data[column].mode().iloc[0]  # Get the mode value
    Data[column].fillna(mode_value, inplace=True)


# In[13]:


Data.info()


# In[14]:


# 3 EPLORATORY DATA ANALYSIS (EDA)

# Histogram of numerical features.
numeric_data = Data.select_dtypes(include=['number']).columns
Data[numeric_data].hist(bins=20, figsize=(12, 6))
plt.suptitle('Histograms of Numerical Features', y=1.02)
plt.tight_layout()
plt.show()


# In[15]:


# Bar plot for categorical features
categorical_data = Data.select_dtypes(include=['object']).columns
num_plots = len(categorical_data)
num_cols = 3
num_rows = num_plots // num_cols + (1 if num_plots % num_cols > 0 else 0)

plt.figure(figsize=(18, num_rows * 5))

for i, feature in enumerate(categorical_data, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.countplot(data=Data, x=feature)
    plt.title(f'Count Plot of {feature}')
    plt.xlabel(None)  # Remove x-axis label for clarity
    plt.ylabel('Count')  # Add y-axis label
    plt.xticks(rotation=45, ha='right')  # Adjust rotation and alignment of x-axis ticks
    plt.legend([], [], frameon=False)  # Remove legend

plt.tight_layout()
plt.show()


# In[16]:


#Let's check the number of house sold per month
# Group by 'YrSold' and 'MoSold' columns, count the number of houses sold in each month
houses_sold_per_month = Data.groupby(['yrsold', 'mosold']).size().reset_index(name='housessold')
palette = sns.color_palette('husl', n_colors=len(houses_sold_per_month['yrsold'].unique()))

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=houses_sold_per_month, x='mosold', y='housessold', hue='yrsold', palette=palette, marker='o')
plt.title('Number of Houses Sold per Month')
plt.xlabel('Month')
plt.ylabel('Number of Houses Sold')
plt.legend(title='Year Sold', loc='upper left', fontsize='small')  # Adjust legend title font size
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])  # Add month names
plt.grid(True)  # Add gridlines
plt.show()


# In[17]:


#Correlation Analysis:
#Compute and visualize the correlation matrix between numerical features.
numeric_data = Data.select_dtypes(include=['number'])
numeric_data.corr()


# In[18]:


# Let's visualize the correlation matrix as a heatmap
plt.figure(figsize=(35,20))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.7)
plt.title('Correlation Matrix')
plt.show()


# In[19]:


# 3 FEATURE ENGINEERING

# Before creating somes news features let's convert categorical into numericals features

# Select categorical columns
categorical_cols = Data.select_dtypes(include=['object']).columns

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply Label Encoding to each categorical column
for col in categorical_cols:
    Data[col] = label_encoder.fit_transform(Data[col])


# In[20]:


# Let's create somes new features by Combining information from existing ones

# Area size 
Data['area_size'] = Data['1stflrsf'] + Data['2ndflrsf'] + Data['grlivarea'] + Data['lotarea'] + Data['wooddecksf'] + Data['openporchsf'] + Data['enclosedporch'] + Data['3ssnporch'] + Data['screenporch'] + Data['poolarea']
# House age category 
Data['house_age'] = Data['yrsold'] - Data['yearbuilt']
# Total bathrooms in the house
Data['totalbathrooms'] = Data['fullbath'] + (0.5 * Data['halfbath']) + Data['bsmtfullbath'] + (0.5 * Data['bsmthalfbath'])

# Display the first few rows of the DataFrame
Data.head()


# In[21]:


# Let's scaling all the variables
Data_Scaling = Data
# Min-Max scaling
scaler_minmax = MinMaxScaler()
Data_Scaling = scaler_minmax.fit_transform(Data)


# In[22]:


# Convert scaled NumPy array back to a DataFrame
Data_Scaling = pd.DataFrame(Data_Scaling, columns=Data.columns)

# Display the first few rows of the DataFrame
Data_Scaling.head()


# In[23]:


# 4 MODEL TRAINNING

# X and y are your features and target variable, respectively
# Since my target variable is 'SalePrice' let's predict it using other features
X = Data_Scaling.drop(columns=['saleprice'])  # Features
y = Data_Scaling['saleprice']  # Target variable

# Set the random seed for reproducibility
np.random.seed(123456)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456)

# Print the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[24]:


# Define hyperparameters
learning_rate = 0.1  # Also known as eta
max_depth = 5
n_estimators = 100  # Number of trees
reg_alpha = 0  # L1 regularization term
reg_lambda = 1  # L2 regularization term

# Initialize with specified hyperparameters
xgb_model = xgb.XGBRegressor(
    learning_rate=learning_rate,
    max_depth=max_depth,
    n_estimators=n_estimators,
    reg_alpha=reg_alpha,
    reg_lambda=reg_lambda
)

# Train the XGBoost model on the training data
xgb_model.fit(X_train, y_train)

# Optionally, you can make predictions on the test data
y_pred = xgb_model.predict(X_test)


# In[25]:


y_pred


# In[26]:


# 5 MODEL EVALUATION

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)


# In[27]:


# These values indicate that our model's predictions are relatively close to the actual house prices, with RMSE being approximately 0.03891.
# which means that, on average, your model's predictions deviate by around $0.03891 from the actual prices. 
# Similarly, the MAE of approximately 0.02148 indicates that, on average, the absolute difference between the predicted and actual prices is around $0.02148.

# Overall, these evaluation metrics suggest that our XGBoost model is performing well in predicting house prices, with relatively low errors.


# In[28]:


# Train a Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame to store feature importances along with feature names
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print or visualize the feature importances
feature_importance_df.head()


# In[29]:


# Generate a color gradient based on feature importances
norm = plt.Normalize(min(feature_importance_df['Importance']), max(feature_importance_df['Importance']))
cmap = plt.cm.viridis

# Plotting feature importances with color gradient
plt.figure(figsize=(10, 6))
bar_plot = plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10], color=cmap(norm(feature_importance_df['Importance'][:10])))
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()  # Invert y-axis to display the most important features at the top

# Create a colorbar to display the color gradient legend
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical')
cbar.set_label('Importance')

plt.show()


# In[30]:


#In order to perform the GUI let's retrain the model 

# X and y are your features and target variable, respectively
# Since my target variable is 'SalePrice' let's predict it using other features
X = Data.drop(columns=['saleprice'])  # Features
y = Data['saleprice']  # Target variable

# Set the random seed for reproducibility
np.random.seed(123456)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456)

# Print the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Define hyperparameters
learning_rate = 0.1  # Also known as eta
max_depth = 5
n_estimators = 100  # Number of trees
reg_alpha = 0  # L1 regularization term
reg_lambda = 1  # L2 regularization term

# Initialize with specified hyperparameters
xgb_model = xgb.XGBRegressor(
    learning_rate=learning_rate,
    max_depth=max_depth,
    n_estimators=n_estimators,
    reg_alpha=reg_alpha,
    reg_lambda=reg_lambda
)

# Train the XGBoost model on the training data
xgb_model.fit(X_train, y_train)


# In[32]:


# Function to perform prediction
def predict_sale_price(input_values):
    # Create a DataFrame with zeros for all features
    input_data = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
    
    # Update the values for the features present in input_values
    for feature, value in input_values.items():
        if feature in input_data.columns:
            input_data[feature] = value
    
    # Use the trained model to make predictions
    predicted_price = xgb_model.predict(input_data)
    
    # Return the predicted sale price
    return predicted_price[0]

# Function to handle button click event
def on_predict_click():
    try:
        # Get input values from entry fields and convert to float
        input_values = {
            'overallqual': float(overallqual_entry.get()),
            'grlivarea': float(grlivarea_entry.get()),
            '2ndflrsf': float(secondflr_entry.get()),
            'totalbsmtsf': float(totalbsmtsf_entry.get()),
            'area_size': float(area_size_entry.get()),
            'totalbathrooms': float(totalbath_entry.get())
        }
        
        # Perform prediction
        predicted_price = predict_sale_price(input_values)
        
        # Display predicted price in output label
        predicted_price_label.config(text=f"Your house is estimated at : ${predicted_price:,.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Set up the main window using customtkinter
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

window = ctk.CTk()
window.title("Property Sale Price Estimator")
window.geometry("800x600")

# Use a canvas to draw a gradient
canvas = tk.Canvas(window, width=800, height=600)
canvas.grid(row=0, column=0, rowspan=8, columnspan=3, sticky="nsew")

# Draw a simple background
canvas.create_rectangle(0, 0, 800, 600, fill="blue")

overallqual_label = ttk.Label(window, text="Rate the quality of the house:", font=('Helvetica', 14))
overallqual_entry = ttk.Entry(window, font=('Helvetica', 14))

grlivarea_label = ttk.Label(window, text="Size of the ground living area:", font=('Helvetica', 14))
grlivarea_entry = ttk.Entry(window, font=('Helvetica', 14))

secondflr_label = ttk.Label(window, text="Size of the second floor:", font=('Helvetica', 14))
secondflr_entry = ttk.Entry(window, font=('Helvetica', 14))

totalbsmtsf_label = ttk.Label(window, text="Total of the basement:", font=('Helvetica', 14))
totalbsmtsf_entry = ttk.Entry(window, font=('Helvetica', 14))

area_size_label = ttk.Label(window, text="Total size area:", font=('Helvetica', 14))
area_size_entry = ttk.Entry(window, font=('Helvetica', 14))

totalbath_label = ttk.Label(window, text="Total bathrooms:", font=('Helvetica', 14))
totalbath_entry = ttk.Entry(window, font=('Helvetica', 14))

predict_button = ttk.Button(window, text="Predict", command=on_predict_click)

predicted_price_label = ttk.Label(window, text="Your house is estimated at:", font=('Helvetica', 16, 'bold'))

# Place widgets using grid layout for left alignment
overallqual_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
overallqual_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

grlivarea_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
grlivarea_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')

secondflr_label.grid(row=2, column=0, padx=10, pady=10, sticky='w')
secondflr_entry.grid(row=2, column=1, padx=10, pady=10, sticky='w')

totalbsmtsf_label.grid(row=3, column=0, padx=10, pady=10, sticky='w')
totalbsmtsf_entry.grid(row=3, column=1, padx=10, pady=10, sticky='w')

area_size_label.grid(row=4, column=0, padx=10, pady=10, sticky='w')
area_size_entry.grid(row=4, column=1, padx=10, pady=10, sticky='w')

totalbath_label.grid(row=5, column=0, padx=10, pady=10, sticky='w')
totalbath_entry.grid(row=5, column=1, padx=10, pady=10, sticky='w')

predict_button.grid(row=6, column=0, columnspan=2, padx=10, pady=20)

predicted_price_label.grid(row=7, column=0, columnspan=2, padx=10, pady=20)

# Run the GUI application
window.mainloop()


# In[ ]:




