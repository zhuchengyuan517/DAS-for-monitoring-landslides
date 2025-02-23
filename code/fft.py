import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Read the CSV file
signal = pd.read_csv(r"D:\landslide-data\all-14-15.csv", header=None)

# Assume the first column is timestamps, and the remaining columns are time series of spatial points
time_stamps = signal.iloc[:, 0]
space_time = signal.iloc[:, 1:]
time_points, space_points = space_time.shape

# Store the frequency-space matrix after Fourier transform
spectra_matrix = []

for i in range(space_points):
    # Compute the Fourier transform
    fourier_transform = np.fft.fft(space_time.iloc[1:, i].values)
    # Take the positive frequency part, indices from 0 to num_points // 2 + 1
    spectra_matrix.append(fourier_transform[:space_points // 2 + 1])

# Compute the amplitude of the spectra (frequency intensity)
amplitude_spectra = np.abs(np.array(spectra_matrix).T)

# Compute the frequency axis (sampling rate: T = dT = 0.006)
freqs = np.fft.fftfreq(len(time_stamps), d=0.006)[:space_points // 2 + 1]

# Save the amplitude spectra to a CSV file
np.savetxt(r'D:\landslide-data\09-15-1-.csv', amplitude_spectra, delimiter=',')

# Normalize the amplitude spectra for visualization
vmin = np.min(amplitude_spectra)  # Set the minimum value for the color map
vmax = np.max(amplitude_spectra)  # Set the maximum value for the color map
amplitude_spectra = (amplitude_spectra - vmin) / (vmax - vmin)

# Plot the heatmap
plt.figure(figsize=(16, 4))
im = plt.imshow(
    amplitude_spectra,
    aspect='auto',
    origin='lower',
    extent=[0, space_points, 0, freqs[-1]],
    cmap='coolwarm',
    vmin=vmin,
    vmax=vmax,
    interpolation='none'
)

# Add a color bar
cbar = plt.colorbar(im, label="Frequency Intensity")
cbar.ax.tick_params(labelsize=12, labelcolor='black')  # Set the size and color of the color bar tick labels
cbar.set_label("Frequency Intensity", fontsize=24, fontname='Times New Roman')  # Set the font and size of the color bar label

# Label the axes
plt.xlabel("Distance (m)", fontsize=24, fontname='Times New Roman')
plt.ylabel("Frequency (Hz)", fontsize=24, fontname='Times New Roman')

# Set x-axis tick labels to 0, 10000, 20000, 30000, 40000, and adjust tick size and font
xticks = [0, 10000, 20000, 30000, 40000]
plt.xticks(ticks=np.arange(0, space_points + 1, 100), labels=xticks, fontsize=20, fontname='Times New Roman')

# Set y-axis tick size and font
plt.yticks(fontsize=20, fontname='Times New Roman')

# Set the number of x-axis ticks to 5
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

# Save the figure
output_path = "frequency_09-15-1.png"  # Path to save the image
plt.savefig(output_path, dpi=1200)  # Save as a high-resolution image
print(f"Image saved as: {output_path}")

# Display the plot
plt.show()