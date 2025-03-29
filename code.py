import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv("chronic_kidney_disease_cleaned.csv")

column_names = [
    "age", "blood_pressure", "specific_gravity", "albumin", "sugar", "red_blood_cells",
    "pus_cell", "pus_cell_clumps", "bacteria", "blood_glucose_random", "blood_urea", "serum_creatinine",
    "sodium", "potassium", "hemoglobin", "packed_cell_volume", "white_blood_cell_count",
    "red_blood_cell_count", "hypertension", "diabetes_mellitus", "coronary_artery_disease",
    "appetite", "pedal_edema", "anemia", "class"
]
df.columns = column_names
df_cleaned = df.iloc[1:].reset_index(drop=True)
df_cleaned.info(), df_cleaned.head()


X = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.replace('?', np.nan)
X_test = X_test.replace('?', np.nan)    

# Define numerical and categorical columns
numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod',
                'pot', 'hemo', 'pcv', 'wc', 'rc']
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Numeric transformer: Impute missing values with median and scale data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer: Impute missing values with most frequent and encode as numbers
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Create full pipeline with preprocessing and classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
svm_pred = pipeline.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")

# Save the pipeline
joblib.dump(pipeline, 'pipeline.pkl')
print("Pipeline saved as pipeline.pkl")
