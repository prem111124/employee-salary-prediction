# Employee Salary Prediction using Linear Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Generate synthetic dataset
np.random.seed(42)
n_samples = 100

# Features
experience = np.random.normal(5, 2, n_samples).clip(0)
education = np.random.choice([0, 1, 2], size=n_samples)  # 0=High School, 1=Bachelor's, 2=Master's
age = (experience + np.random.randint(22, 35, n_samples)).clip(22, 60)

# Target (Salary)
salary = 30 + experience * 7 + education * 10 + np.random.normal(0, 5, n_samples)
salary = salary.round(2)

# 2. Create DataFrame
df = pd.DataFrame({
    'Experience': experience,
    'Education': education,
    'Age': age,
    'Salary': salary
})

print("Sample Data:")
print(df.head())

# 3. Prepare features and target
X = df[['Experience', 'Education', 'Age']]
y = df['Salary']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model training
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Absolute Error (MAE): ₹{mae:.2f}k")
print(f"Mean Squared Error (MSE): ₹{mse:.2f}k²")
print(f"R² Score: {r2:.2f} ({r2*100:.2f}%)")

# 8. Visualization (Optional)
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.grid(True)
plt.show()
