import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read CSV file
def read_csv(file_path):
    return pd.read_csv(file_path, index_col=0)  # Use the first column as the index


# Perform frequency-wavenumber analysis
def fk_analysis(data):
    fft_data = np.fft.fft2(data.values)  # Perform 2D Fourier transform on the 2D array
    fft_shifted = np.fft.fftshift(fft_data)  # Shift zero frequency to the center of the spectrum

    freqs = np.fft.fftfreq(data.shape[0], d=1.0)  # Calculate frequencies
    wavenums = np.fft.fftfreq(data.shape[1], d=1.0)  # Calculate wavenumbers

    return freqs, wavenums, np.abs(fft_shifted)  # Return frequencies, wavenumbers, and amplitude spectrum


# Visualize the frequency-wavenumber analysis results
def plot_fk_analysis(freqs, wavenums, amplitude_spectrum):
    plt.figure(figsize=(10, 5))

    # Adjust vmin and vmax parameters
    vmin = np.percentile(amplitude_spectrum, 5)
    vmax = np.percentile(amplitude_spectrum, 95)

    # Use log scale if needed
    amplitude_spectrum_log = np.log1p(amplitude_spectrum)

    c = plt.contourf(wavenums, freqs, amplitude_spectrum_log, levels=10, cmap='jet', vmin=vmin, vmax=vmax)

    plt.colorbar(c, label='Log Amplitude')

    # Set labels and title
    plt.xlabel('Wavenumber (m⁻¹)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Frequency-Wavenumber Analysis')

    # Customize ticks
    plt.xticks(np.linspace(-0.1, 0.1, 5))
    plt.yticks(np.linspace(0, 0.5, 5))

    plt.show()


# Example usage
if __name__ == "__main__":
    file_path = "D:\\data\\2022-09-14-2-guohua\\14-2-0-24-down.csv"  # Replace with your CSV file path
    data = read_csv(file_path)
    data = data.iloc[:, 1:]  # Remove the index column
    freqs, wavenums, amplitude_spectrum = fk_analysis(data)
    plot_fk_analysis(freqs, wavenums, amplitude_spectrum)