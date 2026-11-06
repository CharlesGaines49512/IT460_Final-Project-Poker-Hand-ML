import pandas as pd
import numpy as np

# Define column names since the file has no headers
column_names = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']

# Load the training dataset
df = pd.read_csv('poker-hand-training-true.data', names=column_names, header=None)

# Display the first 5 rows
print("First 5 rows:")
print(df.head())

# Check the shape (rows, columns)
print("\nShape of dataset:")
print(df.shape)

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Check the distribution of the CLASS column
print("\nClass distribution:")
print(df['CLASS'].value_counts().sort_index())

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Separate features (X) from target (y)
X = df.drop('CLASS', axis=1)
y = df['CLASS']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Linear Regression model
model = LinearRegression()

# Fit (train) the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Linear Regression Results ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# Show a few actual vs predicted values
print("\nSample of Actual vs Predicted:")
comparison = pd.DataFrame({'Actual': y_test.values[:10], 'Predicted': y_pred[:10]})
print(comparison)

# Compare model parameters using the same held-out validation data
print("\n--- Parameter Comparison: fit_intercept ---")

parameter_results = []

for intercept_setting in [True, False]:
    experiment_model = LinearRegression(
        fit_intercept=intercept_setting
    )

    experiment_model.fit(X_train, y_train)
    experiment_predictions = experiment_model.predict(X_test)

    experiment_mse = mean_squared_error(
        y_test, experiment_predictions
    )
    experiment_r2 = r2_score(
        y_test, experiment_predictions
    )

    parameter_results.append({
        "fit_intercept": intercept_setting,
        "Validation MSE": experiment_mse,
        "Validation R2": experiment_r2
    })

results_df = pd.DataFrame(parameter_results)

print(results_df.to_string(index=False, float_format="%.6f"))

results_df.to_csv("parameter_comparison.csv", index=False)
print("\nResults saved to parameter_comparison.csv")

# Compare unrestricted and nonnegative coefficients
print("\n--- Parameter Comparison: positive ---")

positive_results = []

for positive_setting in [False, True]:
    positive_model = LinearRegression(
        fit_intercept=True,
        positive=positive_setting
    )

    positive_model.fit(X_train, y_train)
    positive_predictions = positive_model.predict(X_test)

    positive_results.append({
        "fit_intercept": True,
        "positive": positive_setting,
        "Validation MSE": mean_squared_error(
            y_test, positive_predictions
        ),
        "Validation R2": r2_score(
            y_test, positive_predictions
        )
    })

positive_results_df = pd.DataFrame(positive_results)

print(
    positive_results_df.to_string(
        index=False, float_format="%.6f"
    )
)

positive_results_df.to_csv(
    "positive_parameter_comparison.csv", index=False
)

print("\nResults saved to positive_parameter_comparison.csv")

# Final evaluation using the separate testing dataset
print("\n--- Final Evaluation on Separate Testing Dataset ---")

testing_df = pd.read_csv(
    "poker-hand-testing.data",
    names=column_names,
    header=None
)

print("Testing dataset shape:", testing_df.shape)
print(
    "Total missing values in testing dataset:",
    testing_df.isnull().sum().sum()
)

X_final_test = testing_df.drop("CLASS", axis=1)
y_final_test = testing_df["CLASS"]

# Retrain the selected configuration on all training records
final_model = LinearRegression(
    fit_intercept=True,
    positive=True
)

final_model.fit(X, y)

# Predict on the separate testing records
final_predictions = final_model.predict(X_final_test)

final_mse = mean_squared_error(
    y_final_test, final_predictions
)
final_r2 = r2_score(
    y_final_test, final_predictions
)

print("Selected parameters: fit_intercept=True, positive=True")
print("Training records:", len(X))
print("Testing records:", len(X_final_test))
print(f"Final Test MSE: {final_mse:.6f}")
print(f"Final Test R2: {final_r2:.6f}")

# Compare with a baseline that predicts the training target mean
mean_predictions = np.full(
    len(y_final_test), y.mean()
)

mean_baseline_mse = mean_squared_error(
    y_final_test, mean_predictions
)

print(f"Training-Mean Baseline Test MSE: {mean_baseline_mse:.6f}")

# Save final metrics
final_results_df = pd.DataFrame([{
    "fit_intercept": True,
    "positive": True,
    "Training Records": len(X),
    "Testing Records": len(X_final_test),
    "Test MSE": final_mse,
    "Test R2": final_r2,
    "Training-Mean Baseline Test MSE": mean_baseline_mse
}])

final_results_df.to_csv(
    "final_test_results.csv", index=False
)

# Display and save a small prediction sample
final_comparison = pd.DataFrame({
    "Actual": y_final_test.iloc[:10].to_numpy(),
    "Predicted": final_predictions[:10]
})

print("\nFinal Test Sample of Actual vs Predicted:")
print(final_comparison.to_string(index=False))

final_comparison.to_csv(
    "final_prediction_sample.csv", index=False
)

print("\nSaved final_test_results.csv and final_prediction_sample.csv")