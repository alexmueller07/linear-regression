import pandas as pd 
import matplotlib.pyplot as plt 

# Calculates the average squared error for a given line (y = mx + b) over the dataset
def loss_function(m, b, points):
    x_col = points.columns[0]
    y_col = points.columns[1]
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i][x_col]
        y = points.iloc[i][y_col]
        total_error += (y - (m * x + b))**2  # Squared error for each point
    return total_error / float(len(points))  # Mean squared error

# Performs one step of gradient descent and returns the updated slope (m) and intercept (b)
def gradient_descent(m_now, b_now, points, L):
    x_col = points.columns[0]
    y_col = points.columns[1]
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i][x_col]
        y = points.iloc[i][y_col]
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))  # Partial derivative w.r.t m
        b_gradient += -(2/n) * (y - (m_now * x + b_now))      # Partial derivative w.r.t b

    m = m_now - m_gradient * L  # Update m
    b = b_now - b_gradient * L  # Update b

    return m, b

# Loads and standardizes data from a CSV file, returns both standardized and original values
def get_data(fileName):
    data = pd.read_csv(fileName)

    # Ensure data is float type for calculations
    data.iloc[:, 0] = data.iloc[:, 0].astype(float)
    data.iloc[:, 1] = data.iloc[:, 1].astype(float)

    # Keep original data for later plotting
    original_x = data.iloc[:, 0].copy()
    original_y = data.iloc[:, 1].copy()

    # Compute mean and std for standardization
    x_mean = original_x.mean()
    x_std = original_x.std()
    y_mean = original_y.mean()
    y_std = original_y.std()

    # Apply z-score normalization
    data.iloc[:, 0] = (original_x - x_mean) / x_std
    data.iloc[:, 1] = (original_y - y_mean) / y_std

    return data, x_std, y_std, x_mean, y_mean, original_x, original_y

# Load and preprocess the data
data, x_std, y_std, x_mean, y_mean, original_x, original_y = get_data('data.csv')

# Initialize model parameters
m = 0  # Initial slope
b = 0  # Initial intercept
L = 0.01  # Learning rate
epochs = 1000  # Number of iterations

# Perform gradient descent
for i in range(epochs):
    if i % 100 == 0: 
        print(f"Epoch: {i}, Loss: {loss_function(m, b, data):.6f}")
    m, b = gradient_descent(m, b, data, L)

print(f"Final m (scaled): {m}, b (scaled): {b}")

# Convert the final scaled parameters back to original units
new_m = (m * y_std) / x_std
new_b = (b * y_std) + y_mean - (m * y_std * x_mean) / x_std

print(f"Final m (original): {new_m}, b (original): {new_b}")

# Plot original data with best fit line
plt.scatter(original_x, original_y, color="black")  # Plot raw data points

# Generate x/y pairs for plotting the regression line
x_vals = [original_x.min(), original_x.max()]
y_vals = [new_m * x + new_b for x in x_vals]

plt.plot(x_vals, y_vals, color='red')  # Plot regression line
plt.xlabel(original_x.name)
plt.ylabel(original_y.name)
plt.title('Linear Regression on Original Data')
plt.show()
