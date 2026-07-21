





























import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Download the dataset (student-mat.csv)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"
df = pd.read_csv(url, sep=";")

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn names:\n{df.columns.tolist()}")










# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Dataset info
print("\nDataset Info:")
print(df.dtypes)

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())






# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")



# Define target variable (G3 - final grade)
target = "G3"

# Remove G1 and G2 to avoid data leakage (G3 is our target)
features = [col for col in numerical_cols if col not in ["G1", "G2", "G3"]] + categorical_cols

print(f"Features to use ({len(features)}): {features}")
print(f"Target: {target}")






X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")






class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def _compute_loss(self, X, y):
        n = len(y)
        predictions = X.dot(self.weights) + self.bias
        loss = (1 / (2 * n)) * np.sum((predictions - y) ** 2)
        return loss
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        for i in range(self.n_iterations):
            predictions = X.dot(self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * X.T.dot(predictions - y)
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Record loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            
        return self
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias

print("LinearRegressionGD class defined successfully!")



# Create preprocessing pipeline
numeric_features = [col for col in features if col in numerical_cols]
categorical_features = [col for col in features if col in categorical_cols]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Fit preprocessor on training data and transform both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed training shape: {X_train_processed.shape}")
print(f"Processed testing shape: {X_test_processed.shape}")






# Train with different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5]
models = {}

for lr in learning_rates:
    model = LinearRegressionGD(learning_rate=lr, n_iterations=1000)
    model.fit(X_train_processed, y_train.values)
    models[lr] = model
    print(f"Learning Rate {lr}: Final Loss = {model.loss_history[-1]:.4f}")



# Plot loss convergence for different learning rates
plt.figure(figsize=(12, 6))
for lr, model in models.items():
    plt.plot(model.loss_history, label=f"LR = {lr}")

plt.xlabel("Iterations")
plt.ylabel("Loss (MSE/2)")
plt.title("Convergence of Loss for Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()










# Select the best model (lowest final loss)
best_lr = min(models.keys(), key=lambda lr: models[lr].loss_history[-1])
best_model = models[best_lr]
print(f"Best learning rate: {best_lr}")

# Make predictions
y_train_pred = best_model.predict(X_train_processed)
y_test_pred = best_model.predict(X_test_processed)

# Calculate metrics
def evaluate_model(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{dataset_name} Metrics:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    return mae, mse, rmse, r2

train_metrics = evaluate_model(y_train.values, y_train_pred, "Training Set")
test_metrics = evaluate_model(y_test.values, y_test_pred, "Testing Set")



# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Student Grades (Test Set)")
plt.grid(True)
plt.show()



# Plot residual distribution
residuals = y_test.values - y_test_pred
plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.grid(True)
plt.show()
























from sklearn.linear_model import LinearRegression as SKLinearRegression

# Train scikit-learn model
sk_model = SKLinearRegression()
sk_model.fit(X_train_processed, y_train)

# Predictions
y_test_pred_sk = sk_model.predict(X_test_processed)

# Evaluate
mae_sk = mean_absolute_error(y_test, y_test_pred_sk)
mse_sk = mean_squared_error(y_test, y_test_pred_sk)
rmse_sk = np.sqrt(mse_sk)
r2_sk = r2_score(y_test, y_test_pred_sk)

print("Scikit-Learn Linear Regression Results:")
print(f"  MAE  : {mae_sk:.4f}")
print(f"  MSE  : {mse_sk:.4f}")
print(f"  RMSE : {rmse_sk:.4f}")
print(f"  R²   : {r2_sk:.4f}")

print(f"\nOur GD Model (LR={best_lr}):")
print(f"  MAE  : {test_metrics[0]:.4f}")
print(f"  MSE  : {test_metrics[1]:.4f}")
print(f"  RMSE : {test_metrics[2]:.4f}")
print(f"  R²   : {test_metrics[3]:.4f}")
