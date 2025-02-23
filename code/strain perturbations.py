import numpy as np
import pandas as pd

# Set parameters (adjustable as needed)
λ = 1.55e-6  # Wavelength, can be adjusted as needed
ζ = 0.78     # Damping ratio, can be adjusted as needed
n = 1.467    # Refractive index, can be adjusted as needed
L = 10       # Length, can be adjusted as needed

# Define the formula function
def calculate_theta(phi, λ, ζ, n, L):
    return (λ * phi) / (4 * np.pi * ζ * n * L)

# Read the CSV file
file_path = r'D:\landslide-data\2022-09-15-2-guohua\09-15-2-0-24-500-down.csv'
data = pd.read_csv(file_path, header=None)

# Extract the header (first row) and index (first column)
header = data.iloc[0, 1:]  # First row (sensor IDs)
index = data.iloc[1:, 0]   # First column (time)

# Extract sensor measurement data, ignoring the first row and first column
sensor_data = data.iloc[1:, 1:].astype(float)

# Apply the formula to transform sensor measurement data
theta_data = sensor_data.applymap(lambda phi: calculate_theta(phi, λ, ζ, n, L))

# Restore the header and index
theta_data.insert(0, 'Time', index)  # Restore the time column
theta_data.columns = ['Time'] + header.tolist()  # Restore the sensor IDs row

# Save the results to a new CSV file
output_path = r'D:\landslide-data\2022-09-15-2-guohua\09-15-2-0-24-500-down-strain.csv'
theta_data.to_csv(output_path, index=False)

print(f"Transformed sensor data has been saved to: {output_path}")