import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from matplotlib.colors import LinearSegmentedColormap

colors = ["blue", "white", "red"]
cmap = LinearSegmentedColormap.from_list("mycmap", colors)

# Read csv file
file_paths = ['D:\\landslide-data\\2022-09-14-2-guohua\\09-14-2-0-24-500-down.csv', 'D:\\landslide-data\\2022-09-15-2-guohua\\09-15-2-0-24-500-down.csv']  # Replace with the paths to your CSV files

# Read and concatenate your CSV files
dataframes = [pd.read_csv(file_path, index_col=0) for file_path in file_paths]
df = pd.concat(dataframes)


data = df.iloc[1:, :].values.T  # skip the first row of sensor identifiers and transpose the dataframe

# Window parameters
t_win = min(3600, data.shape[1])  # Time window (each row now, after transposing), no greater than data length
s_win = 10  # Space window (each column)
t_overlap = t_win // 2
s_overlap = s_win // 2

# Spectrogram parameters
nfft = 2**np.ceil(np.log2(t_win)).astype(int)  # Round up to nearest power of 2
window_type = 'hanning'

# Sensor signals are assumed to be in columns after transposing
for i in range(data.shape[0]):
    signal = data[i, :]

    # Apply sliding window spectrogram S-transform
    frequencies, times, Sxx = spectrogram(signal, window=window_type, nperseg=t_win, noverlap=t_overlap, nfft=nfft)

    # Plot S-transform
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), cmap=cmap)  # Display in dB
    plt.title(f'S-transform of Signal {i+1}')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')

    # Save the plot
    plt.savefig(f'D:\landslide-data\\14-15-2\\Spectrogram_Sensor_{i+1}.png')
    plt.close()