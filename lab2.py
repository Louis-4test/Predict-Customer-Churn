import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
import joblib

# Load your dataset
data = pd.read_csv('/home/fola/Downloads/customer_churn_data.csv')

# Save customer_id for later use and drop from feature set
if 'customer_id' in data.columns:
    customer_ids = data['customer_id']  # Store customer_id
    data = data.drop(columns=['customer_id'])  # Drop for modeling

# Define features and target variable
X = data.drop(columns=['churn'])  # Use 'churn' as the target column
y = data['churn']

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add back customer_id to the test set
X_test['customer_id'] = customer_ids.loc[X_test.index].values  # Ensure IDs are aligned correctly

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# Create and fit the adapted model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the adapted model
model_pipeline.fit(X_train, y_train)

# Create baseline model using DummyClassifier (as a simple baseline)
baseline_model = DummyClassifier(strategy='most_frequent', random_state=42)
baseline_model.fit(X_train, y_train)

# Function to evaluate model performance
def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Evaluate the adapted model
print("Evaluating Adapted Model:")
evaluate_model(model_pipeline, X_test.drop(columns=['customer_id']), y_test)

# Evaluate the baseline model
print("\nEvaluating Baseline Model:")
evaluate_model(baseline_model, X_test.drop(columns=['customer_id']), y_test)

# Now split test set into older and newer customers
# Assuming 'customer_id' is ordered chronologically
median_id = X_test['customer_id'].median()  # Now we can access 'customer_id' safely
older_customers = X_test[X_test['customer_id'] < median_id]
newer_customers = X_test[X_test['customer_id'] >= median_id]

# Evaluate on older customers
print("\nEvaluating Adapted Model on Older Customers:")
evaluate_model(model_pipeline, older_customers.drop(columns=['customer_id']), y_test[older_customers.index])

print("\nEvaluating Baseline Model on Older Customers:")
evaluate_model(baseline_model, older_customers.drop(columns=['customer_id']), y_test[older_customers.index])

# Evaluate on newer customers
print("\nEvaluating Adapted Model on Newer Customers:")
evaluate_model(model_pipeline, newer_customers.drop(columns=['customer_id']), y_test[newer_customers.index])

print("\nEvaluating Baseline Model on Newer Customers:")
evaluate_model(baseline_model, newer_customers.drop(columns=['customer_id']), y_test[newer_customers.index])

# Optional: Save the models
joblib.dump(model_pipeline, 'adapted_model_pipeline.pkl')
joblib.dump(baseline_model, 'baseline_model.pkl')