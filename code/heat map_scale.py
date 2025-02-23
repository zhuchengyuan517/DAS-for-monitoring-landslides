import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt

# Read the CSV file
df = pd.read_csv('D:\\landslide-data\\14-15-1-2.csv')

# Ensure the 'time' column is in datetime format
df['time'] = pd.to_datetime(df['time'])

# Design a lowpass filter
def lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Assuming a time interval of 1s, adjust the sampling rate and cutoff frequency as needed
sampling_rate = 1
cutoff_frequency = 0.1

# Apply lowpass filtering to each sensor
df['filtered_energy_gradient'] = df.groupby('sensor')['energy_gradient'].transform(
    lambda x: lowpass_filter(x.values, cutoff_frequency, sampling_rate))

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
df['filtered_energy_gradient'] = scaler.fit_transform(df[['filtered_energy_gradient']])

# Filter sensors within a specific range
df_filtered = df[(df['sensor'] >= 0) & (df['sensor'] <= 100)]

# Create a pivot table for the heatmap
heatmap_data = df_filtered.pivot_table(index='sensor', columns='time', values='filtered_energy_gradient')

# Plot the heatmap
plt.figure(figsize=(18, 9))  # Increase the figure size
ax = sns.heatmap(heatmap_data, cmap='RdYlBu_r', annot=False, fmt='.1f',
                 cbar_kws={'label': 'Energy Gradient'})

# Adjust the color bar font
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=8))  # Adjust the color bar tick font size

# Set the color bar tick font to Times New Roman
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties('Times New Roman')

font = {'family': 'Times New Roman', 'size': 18, 'weight': 'bold'}
cbar.set_label('Energy Gradient', fontdict=font, labelpad=10))  # Set the color bar label font and size

# Hide the x and y axis ticks
ax.set_xticks([]))  # Hide x-axis ticks
ax.set_yticks([]))  # Hide y-axis ticks

# Set the title and axis labels
font = {'family': 'Times New Roman', 'size': 32, 'weight': 'bold'}
plt.ylabel('Distance', fontdict=font, labelpad=10))
plt.xlabel('Time', fontdict=font, labelpad=10))  # Add x-axis label

# Optimize time ticks if needed
# plt.xticks(rotation=45))  # Rotate x-axis labels if they overlap

# Display the plot
plt.tight_layout()
plt.show()

# Save the figure with high resolution if needed
plt.savefig('heatmap.png', dpi=1200))