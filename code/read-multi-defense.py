import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define directory and parameters
dirct = 'D:\\landslide-data\\2022-09-14-2-guohua\\2022-09-14'  # Date directory
starttime = '00:00:00'  # Start time
endtime = '23:59:59'  # End time
index = int(31 / 1)  # Zone number
savefile = 1  # Set to 1 to save the waveform as a CSV file
fangqu = np.arange(int(1), int(500))  # Range of zones


# Function to view waveforms for multiple consecutive zones
def duogefangqu(dirct, starttime, endtime, fangqu, savefile):
    f = open(dirct + '.txt')  # Open the data file
    lines = f.readlines()  # Read all lines

    # Find the start and end indices based on the specified time range
    starindex = 0
    endindex = 0
    for i in range(len(lines)):
        if lines[i].split(',')[0][2:-1] == starttime:
            starindex = i
        if lines[i].split(',')[0][2:-1] == endtime:
            endindex = i
            break

    # Extract data for each zone
    data_dict = {}
    for index in fangqu:
        indexdata = []
        for i in np.arange(starindex, endindex, 1):
            indexdata.append(float(lines[i].split(',')[index + 1]))  # Extract data for the current zone
        data_dict[str(index)] = indexdata  # Store data in a dictionary

    # Save the data to a CSV file if savefile is set to 1
    if savefile == 1:
        dataframe = pd.DataFrame(data_dict)
        dataframe.to_csv("D:\\landslide-data\\2022-09-14-2-guohua\\09-14-2-0-24-500.csv", index=False, sep=',')

    print("============ Continuous waveform CSV file saved at: D:\\连续防区波形 =================")


# Print the current zone and range of zones
print('Single zone number:', index, 'Continuous zone range:', fangqu)

# Call the function to process multiple zones
duogefangqu(dirct, starttime, endtime, fangqu, savefile)