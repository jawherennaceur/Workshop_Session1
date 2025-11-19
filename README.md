# AI Workshop: Introduction to Machine Learning

Welcome to the AI Workshop! This repository contains hands-on exercises for learning the fundamentals of Machine Learning, focusing on **Regression** and **Classification** problems.

## 📋 Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Notebooks Explanation](#notebooks-explanation)
- [Datasets](#datasets)
- [Technologies Used](#technologies-used)

---

## 🎯 Overview

This workshop provides practical experience with:
- **Classification**: Predicting categorical outcomes using K-Nearest Neighbors (KNN) with hyperparameter tuning
- **Regression**: Predicting continuous values using Linear and Polynomial Regression models

Each notebook walks through the complete machine learning pipeline from data loading to model evaluation.

---

## 📚 Prerequisites

Before starting, ensure you have:
- Python 3.7 or higher installed
- Visual Studio Code (VS Code) installed
- Basic understanding of Python programming
- Familiarity with data analysis concepts (helpful but not required)

---

## 🚀 Getting Started

Follow these steps to set up the project on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/jawherennaceur/Workshop_Session1.git
cd Workshop_Session1
```

### 2. Open in Visual Studio Code

```bash
code .
```

Or open VS Code manually and use `File > Open Folder` to select the project directory.

### 3. Open the Terminal

In VS Code, open the integrated terminal:
- **Windows/Linux**: `Ctrl + `` (backtick)
- **Mac**: `Cmd + `` (backtick)

Or go to `Terminal > New Terminal` from the menu.

### 4. Create a Virtual Environment

Creating a virtual environment keeps your project dependencies isolated.

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your terminal prompt, indicating the virtual environment is active.

### 5. Install Required Packages

With the virtual environment activated, install all dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- pandas: Data manipulation and analysis
- scikit-learn: Machine learning algorithms and tools
- jupyter: For running notebook files
- numpy: Numerical computing

### 6. Open a Notebook

In VS Code, navigate to either `Classification.ipynb` or `Regression.ipynb` and open it.

### 7. Select the Virtual Environment Kernel

When you open the notebook:
1. Click on the kernel picker in the top-right corner (it might say "Select Kernel")
2. Choose "Python Environments"
3. Select the virtual environment you just created (it should be labeled `venv` or show the path to your `venv` folder)

### 8. Run the Notebook

Now you're ready to run the code! You can:
- Run cells individually by clicking the play button next to each cell
- Run all cells using `Run All` from the top menu
- Use `Shift + Enter` to run the current cell and move to the next one

---

## 📁 Project Structure

```
AI-Workshop/
│
├── Data/
│   ├── Classification/
│   │   ├── Drug.csv                    # Drug classification dataset
│   │   └── README.md                   # Dataset description and features
│   │
│   └── Regression/
│       ├── House_Pricing.csv           # California housing dataset
│       └── README.md                   # Dataset description and features
│
├── Classification.ipynb                # Classification tutorial notebook
├── Regression.ipynb                    # Regression tutorial notebook
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---
## 📊 Datasets

### Classification Dataset
Located in `Data/Classification/`

**Purpose**: Predict which drug should be prescribed to a patient based on their characteristics.

**Features**:
- Age
- Sex
- Blood Pressure levels
- Cholesterol levels
- Sodium to Potassium ratio

**Target**: Drug type (categorical)

For detailed information, see `Data/Classification/README.md`

---

### Regression Dataset
Located in `Data/Regression/`

**Purpose**: Predict the median house value in California districts.

**Features**:
- Longitude and Latitude
- Housing median age
- Total rooms and bedrooms
- Population
- Households
- Median income
- Ocean proximity

**Target**: Median house value (continuous)

For detailed information, see `Data/Regression/README.md`


---
## 📓 Notebooks Explanation

### Classification.ipynb

This notebook demonstrates **supervised classification** using the K-Nearest Neighbors (KNN) algorithm.

#### What You'll Learn:
- Loading and exploring categorical data
- Preprocessing data with one-hot encoding
- Splitting data into training and testing sets
- Hyperparameter tuning using GridSearchCV
- Training a KNN classifier
- Evaluating model accuracy

#### Key Steps:

1. **Data Loading**
   ```python
   data = pd.read_csv("Data\\Classification\\Drug.csv")
   ```
   The dataset contains patient information used to predict which drug should be prescribed.

2. **Feature Engineering**
   ```python
   X = data.drop(columns='Drug')  # Features
   y = data['Drug']               # Target variable
   X = pd.get_dummies(X, dtype=float, drop_first=True)  # One-hot encoding
   ```
   - Separates features (patient characteristics) from the target (drug type)
   - Converts categorical variables into numerical format using one-hot encoding
   - `drop_first=True` prevents multicollinearity by dropping one dummy variable

3. **Train-Test Split**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```
   - 80% of data for training, 20% for testing
   - `random_state=42` ensures reproducible results

4. **Hyperparameter Tuning**
   ```python
   param_knn = {
       'n_neighbors': list(range(1, 31)),  # Number of neighbors to consider
       'p': (1, 2),                         # Distance metric (1=Manhattan, 2=Euclidean)
       'weights': ('uniform', 'distance'),  # Weight function
       'metric': ('minkowski', 'chebyshev') # Distance calculation method
   }
   ```
   GridSearchCV tests all combinations to find the best parameters using 10-fold cross-validation.

5. **Model Training & Evaluation**
   ```python
   knn = KNeighborsClassifier(n_neighbors=4, metric='minkowski', p=1, weights='distance')
   knn.fit(X_train, y_train)
   y_pred = knn.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred) * 100
   ```
   - Trains the model with optimal parameters
   - Makes predictions on test data
   - Calculates accuracy percentage

#### Expected Outcome:
The model achieves high accuracy in predicting the appropriate drug for patients based on their characteristics.

---

### Regression.ipynb

This notebook covers **supervised regression** for predicting continuous values using Linear and Polynomial Regression.

#### What You'll Learn:
- Loading and cleaning numerical data
- Handling missing values
- Feature scaling and normalization
- Training linear regression models
- Implementing polynomial regression
- Evaluating models using R² score and MSE

#### Key Steps:

1. **Data Loading**
   ```python
   data = pd.read_csv("Data\\Regression\\House_Pricing.csv")
   ```
   The California housing dataset contains information about houses to predict median house values.

2. **Data Cleaning**
   ```python
   X = data.drop('median_house_value', axis=1)  # Features
   y = data.iloc[:, 8:9]                        # Target variable
   
   # Handle missing values
   X['total_bedrooms'] = X['total_bedrooms'].fillna(X['total_bedrooms'].mean())
   
   # Encode categorical features
   X = pd.get_dummies(data=X)
   ```
   - Separates features from the target variable
   - Fills missing bedroom values with the mean
   - Converts categorical variables (like ocean proximity) into numerical format

3. **Data Splitting**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   ```
   - 70% training data, 30% testing data

4. **Feature Scaling**
   ```python
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)
   ```
   - Standardizes features to have mean=0 and standard deviation=1
   - Critical for algorithms sensitive to feature scales
   - `fit_transform` on training data, `transform` on test data to prevent data leakage

5. **Linear Regression**
   ```python
   model = LinearRegression()
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   ```
   - Fits a straight line to the data
   - Assumes linear relationship between features and target

6. **Model Evaluation**
   ```python
   r2 = r2_score(y_test, y_pred)           # R-squared (0 to 1, higher is better)
   mse = mean_squared_error(y_test, y_pred) # Mean Squared Error (lower is better)
   ```
   - **R² Score**: Explains how much variance in the target is captured by the model
   - **MSE**: Average squared difference between predictions and actual values

7. **Polynomial Regression**
   ```python
   poly_features = PolynomialFeatures(degree=2, include_bias=False)
   X_train_poly = poly_features.fit_transform(X_train)
   X_test_poly = poly_features.transform(X_test)
   
   lin_reg = LinearRegression()
   lin_reg.fit(X_train_poly, y_train)
   ```
   - Creates polynomial features (x², x*y, etc.) for capturing non-linear relationships
   - `degree=2` means squared terms and interactions
   - Often provides better fit than simple linear regression

#### Expected Outcome:
- Linear regression provides a baseline model
- Polynomial regression captures non-linear patterns, typically achieving better R² scores and lower MSE

---

