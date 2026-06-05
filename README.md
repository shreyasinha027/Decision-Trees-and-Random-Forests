# Decision Trees and Random Forests

## Objective

The objective of this project is to learn and implement tree-based machine learning models for classification using Decision Trees and Random Forests.

## Dataset

Healthcare Dataset containing patient information such as age, gender, blood type, medical condition, medication, admission details, and test results.

## Tools and Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Tasks Performed

### 1. Data Preprocessing

* Loaded the healthcare dataset.
* Removed unnecessary columns.
* Converted date columns into numerical format.
* Applied One-Hot Encoding to categorical features.

### 2. Decision Tree Classifier

* Trained a Decision Tree model.
* Evaluated performance using:

  * Accuracy
  * Confusion Matrix
  * Classification Report
* Visualized the decision tree structure.

### 3. Overfitting Analysis

* Trained Decision Trees with different maximum depths.
* Compared training and testing accuracy.
* Visualized the effect of tree depth on model performance.

### 4. Random Forest Classifier

* Trained a Random Forest model.
* Compared its accuracy with the Decision Tree model.
* Evaluated performance using classification metrics.

### 5. Feature Importance Analysis

* Extracted feature importance scores from the Random Forest model.
* Identified the most influential features affecting predictions.

### 6. Cross Validation

* Applied 5-Fold Cross Validation.
* Calculated mean accuracy and standard deviation to assess model stability.

## Results

* Successfully implemented Decision Tree and Random Forest classifiers.
* Compared model performance using accuracy metrics.
* Analyzed overfitting behavior through depth control.
* Interpreted important features influencing predictions.
* Evaluated model reliability using cross-validation.

## Project Structure

Decision Trees and Random Forests/

├── healthcare_dataset.csv

├── main.py

├── README.md

└── requirements.txt

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the program:
   python main.py

## Author

Shreya Sinha
