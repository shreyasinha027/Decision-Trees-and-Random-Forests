import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


file_path = r"C:\Users\Shreya\Documents\Decision Trees and Random Forests\healthcare_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())


columns_to_drop = [
    "Name",
    "Doctor",
    "Hospital"
]

for col in columns_to_drop:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)


date_cols = ["Date of Admission", "Discharge Date"]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = df[col].map(lambda x: x.toordinal() if pd.notnull(x) else 0)


target_column = "Test Results"

X = df.drop(target_column, axis=1)
y = df[target_column]



categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)



dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ))
])

dt_pipeline.fit(X_train, y_train)

y_pred_dt = dt_pipeline.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\n" + "="*50)
print("DECISION TREE RESULTS")
print("="*50)

print("Accuracy:", round(dt_accuracy, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))


X_encoded = preprocessor.fit_transform(X)

feature_names = preprocessor.get_feature_names_out()

tree_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

tree_model.fit(X_encoded, y)

plt.figure(figsize=(20,10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=np.unique(y).astype(str),
    filled=True,
    fontsize=8
)

plt.title("Decision Tree Visualization")
plt.show()



depths = range(1, 21)

train_scores = []
test_scores = []

for depth in depths:

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    train_scores.append(
        accuracy_score(y_train, model.predict(X_train))
    )

    test_scores.append(
        accuracy_score(y_test, model.predict(X_test))
    )

plt.figure(figsize=(8,5))

plt.plot(depths, train_scores, marker="o", label="Train Accuracy")
plt.plot(depths, test_scores, marker="o", label="Test Accuracy")

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting Analysis")
plt.legend()
plt.grid(True)

plt.show()


rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)

y_pred_rf = rf_pipeline.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\n" + "="*50)
print("RANDOM FOREST RESULTS")
print("="*50)

print("Accuracy:", round(rf_accuracy, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))



plt.figure(figsize=(6,4))

plt.bar(
    ["Decision Tree", "Random Forest"],
    [dt_accuracy, rf_accuracy]
)

plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.show()



rf_model = rf_pipeline.named_steps["classifier"]

feature_names = preprocessor.fit(X).get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

plt.figure(figsize=(10,6))

top10 = importance_df.head(10)

plt.barh(
    top10["Feature"],
    top10["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.title("Top 10 Feature Importances")

plt.show()



cv_scores = cross_val_score(
    rf_pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n" + "="*50)
print("CROSS VALIDATION")
print("="*50)

print("Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())

print("\nTask Completed Successfully!")