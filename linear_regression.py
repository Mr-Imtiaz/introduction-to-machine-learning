A simple implementation of linear regression using Scikit-learn.

2. **Create `linear_regression.py`:**
   - Create a new file named `linear_regression.py`.
   - Add the following content:

```python
# linear_regression.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset (replace 'your_dataset.csv' with your actual dataset)
data = pd.read_csv('your_dataset.csv')  # Ensure you have a dataset for this example

# Define features and target variable
X = data[['feature1', 'feature2']]  # Replace with actual feature names
y = data['target']  # Replace with actual target variable name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y

