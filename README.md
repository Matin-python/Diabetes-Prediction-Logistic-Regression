# 🩺 Diabetes Prediction using Logistic Regression

A machine learning classification project that uses **Logistic Regression** to predict whether a patient has diabetes based on medical diagnostic measurements. The project also uses data visualization to explore the distribution of diabetes outcomes and relationships between features.

## Overview

This project uses the **Diabetes dataset** to build a binary classification model.

The target variable, `Outcome`, indicates whether a patient has diabetes:

* `0` → No diabetes
* `1` → Diabetes

A **Logistic Regression** model is trained using the available medical features and evaluated on a test set.

The project also includes exploratory data analysis using **Seaborn** and **Matplotlib** to visualize the target distribution and correlations between features.

## Features

* 🩺 Diabetes prediction using Logistic Regression
* 🤖 Binary classification
* 📊 Train/test data splitting
* 📈 Prediction accuracy evaluation
* 📉 Mean Squared Error (MSE) calculation
* 📊 Diabetes outcome distribution visualization
* 🔥 Feature correlation heatmap
* 🐼 Dataset handling using Pandas
* 📚 Beginner-friendly machine learning implementation

## Technologies Used

* Python 3
* NumPy
* Pandas
* Scikit-learn
* Seaborn
* Matplotlib

## Dataset

The project uses the **Diabetes dataset** stored in `diabetes2.csv`.

The dataset contains medical diagnostic measurements that are used to predict the `Outcome` of a patient.

The features include:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

The target variable is:

* `Outcome`

where:

```text
0 = No Diabetes
1 = Diabetes
```

## Data Visualization

Before training the model, the project performs basic exploratory data analysis.

### Outcome Distribution

A count plot is used to visualize the number of patients in each outcome category.

```python
sns.countplot(x='Outcome', data=df)
```

### Feature Correlation

A correlation heatmap is used to examine relationships between the dataset's features.

```python
sns.heatmap(df.corr(), annot=True)
```

These visualizations provide a basic understanding of the dataset before training the machine learning model.

## Machine Learning Workflow

1. Load the dataset using Pandas.
2. Visualize the distribution of diabetes outcomes.
3. Calculate and visualize feature correlations.
4. Separate the input features from the target variable.
5. Split the dataset into training and testing sets.
6. Train a Logistic Regression classifier.
7. Predict outcomes for the test dataset.
8. Calculate prediction accuracy.
9. Calculate Mean Squared Error.

## Model

The project uses **Logistic Regression** for binary classification.

```python
reg_logestic = linear_model.LogisticRegression()

reg_logestic.fit(x_train, y_train)

out_pred = reg_logestic.predict(x_test)
```

Logistic Regression is suitable for this problem because the target variable contains two possible outcomes: `0` or `1`.

## Project Structure

```text
Diabetes-Prediction-Logistic-Regression/
│
├── screenshots/
│   ├── outcome_distribution.png
│   └── correlation_heatmap.png
│
├── diabetes_prediction.py
├── diabetes.csv
├── diabetes2.csv
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Matin-python/diabets.git
```


## Dataset

This project uses the **Digits Dataset** from **scikit-learn**, which contains **1,797 handwritten digit samples** represented as **8×8 grayscale images**.

The trained model is also tested on custom digit images stored in the **test dataset** folder.


## Contributing
Contributions, suggestions, and bug reports are welcome. Feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Author
Mohammad Reza Bakhshandeh

Electrical Engineering (Electronics) Graduate

Interested in Python Development, Machine Learning, Deep Learning, Computer Vision, Artificial Intelligence, and Game Development.
