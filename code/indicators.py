import numpy as np
import pandas as pd
from scipy.signal import stft
from scipy.ndimage import gaussian_filter1d

# Parameters
time_window_size = 3600  # 1 hour in seconds
time_overlap = 1800  # no overlap for hourly windows
space_window_size = 20
threshold = 0.1
secondary_time_window_size = 600
secondary_time_overlap = 300
secondary_space_window_size = 10
secondary_space_overlap = 5
time_sigma = 0.05
space_sigma = 0.05

# Read data
data_path_1 = 'D:\\landslide-data\\2022-09-14-2-guohua\\09-14-2-0-24-500-down.csv'
data_path_2 = 'D:\\landslide-data\\2022-09-15-2-guohua\\09-15-2-0-24-500-down.csv'

data_1 = pd.read_csv(data_path_1)
data_2 = pd.read_csv(data_path_2)

# Extract time and signals
time_1 = pd.to_datetime(data_1.iloc[:, 0]).values
signals_1 = data_1.iloc[:, 1:].values

time_2 = pd.to_datetime(data_2.iloc[:, 0]).values
signals_2 = data_2.iloc[:, 1:].values

# Concatenate signals vertically
signals = np.vstack((signals_1, signals_2))

# Total time array
time = np.concatenate((time_1, time_2))

results = []
# Calculate windows
num_time_windows = (len(time) - time_window_size) // time_overlap + 1
num_space_windows = (signals.shape[1] - space_window_size) // space_window_size + 1

for t_win in range(num_time_windows):
    start_time_idx = t_win * time_overlap
    end_time_idx = start_time_idx + time_window_size

    for s_win in range(num_space_windows):
        start_space = s_win * space_window_size
        end_space = start_space + space_window_size

        window_signals = signals[start_time_idx:end_time_idx, start_space:end_space]

        f, t, Zxx = stft(window_signals, axis=0, nperseg=256)
        power_spectrum = np.abs(Zxx) ** 2

        # Ensure power_spectrum has enough dimensions for gradient calculation
        if power_spectrum.ndim < 3:
            continue

        gradient = np.gradient(power_spectrum, axis=2)
        energy_gradient = np.sum(np.abs(gradient), axis=(0, 1))

        # Maximum power spectrum
        max_power = np.max(power_spectrum, axis=(0, 1))

        # Calculate durations and space ranges
        durations = np.zeros(power_spectrum.shape[2], dtype=int)
        space_ranges = np.zeros(power_spectrum.shape[2], dtype=int)

        for idx in range(power_spectrum.shape[2]):
            power_slice = power_spectrum[:, :, idx]

            # Calculate duration
            duration = 0
            for time_idx in range(power_slice.shape[1]):
                if (power_slice[:, time_idx] >= max_power[idx] * 0.5).any():
                    duration += 1
                else:
                    break
            durations[idx] = duration * secondary_time_overlap

            # Calculate affected sensors
            space_range = 0
            for space_idx in range(power_slice.shape[0]):
                if (power_slice[space_idx, :] >= max_power[idx] * 0.5).any():
                    space_range += 1
            space_ranges[idx] = space_range * secondary_space_overlap

        # Energy gradient
        high_energy_indices = np.where(energy_gradient > threshold)[0]
        if len(high_energy_indices) == 0:
            continue

        for i, idx in enumerate(high_energy_indices):
            result = {
                'time': time[start_time_idx + idx],  # Use the correct timestamp
                'sensor': start_space + idx,
                'energy_gradient': energy_gradient[idx],
                'duration': durations[idx],
                'affected_sensors': space_ranges[idx]
            }
            results.append(result)

# Create DataFrame
results_df = pd.DataFrame(results)

# Calculate thresholds
energy_gradient_threshold = results_df['energy_gradient'].quantile(0.99)
duration_threshold = 1800
affected_sensors_threshold = 10

# Rating
results_df['rating'] = (
    (results_df['energy_gradient'] > energy_gradient_threshold) &
    (results_df['duration'] > 0) & (results_df['duration'] <= duration_threshold) &
    (results_df['affected_sensors'] > 1) & (results_df['affected_sensors'] <= affected_sensors_threshold)
).astype(int)

# Save to CSV
results_df.to_csv('14-15-2.csv', index=False)

print("Processing complete. Results saved to '14-15-2.csv'.")
