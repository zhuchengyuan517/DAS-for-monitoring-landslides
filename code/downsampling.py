import pandas as pd

# Read the CSV file
data = pd.read_csv("D:\\landslide-data\\2022-09-14-2-guohua\\09-14-2-0-24-500.csv")  # Replace with your file path

# Create a time index for each column (if it doesn't exist, you need to create a suitable time index based on your data)
time_index = pd.date_range(start='2022-09-14', periods=len(data), freq='300ms')  # Assuming the start time is 2022-09-14, replace as needed
data.index = time_index

# Fill in missing data to ensure no missing time points
data = data.asfreq('300ms', method='pad')  # Use forward fill to handle missing data

# Downsample to 1Hz
resampled_data = data.resample('1S').mean()  # Use mean for downsampling; can be replaced with .max(), .min(), .median(), etc., as needed

# Ensure the total data count is 86400 (i.e., from 00:00:00 to 23:59:59)
# Generate a complete time index
full_time_index = pd.date_range(start='2022-09-14', periods=86400, freq='1S')

# Reindex and interpolate missing values
final_data = resampled_data.reindex(full_time_index).interpolate(method='linear')

# Save the result back to a CSV file
final_data.to_csv("D:\\landslide-data\\2022-09-14-2-guohua\\09-14-2-0-24-500-down.csv")  # Replace with your desired output file name