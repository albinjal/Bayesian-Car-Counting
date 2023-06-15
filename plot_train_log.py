import pandas as pd
import matplotlib.pyplot as plt
import re

def plot_stats(file_path):
    # Initialize lists to store data
    epochs = []
    losses = []
    mses = []
    maes = []
    phases = []

    # Define the regex patterns
    pattern_train = re.compile(r"\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2}\s*Epoch\s*(\d+)\s*Train,\s*Loss:\s*([\d\.]+),\s*MSE:\s*([\d\.]+)\s*MAE:\s*([\d\.]+),\s*Cost\s*([\d\.]+)\s*sec.*")
    pattern_val = re.compile(r"\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2}\s*Epoch\s*(\d+)\s*Val,\s*MSE:\s*([\d\.]+)\s*MAE:\s*([\d\.]+),\s*Cost\s*([\d\.]+)\s*sec.*")

    # Open and read the text file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Loop over the lines and extract data
    for line in lines:
        match_train = pattern_train.match(line)
        match_val = pattern_val.match(line)

        if match_train:
            epoch, loss, mse, mae, cost = match_train.groups()
            phase = "Train"
        elif match_val:
            epoch, mse, mae, cost = match_val.groups()
            phase = "Val"
            loss = None

        if match_train or match_val:
            epochs.append(int(epoch))
            losses.append(None if loss is None else float(loss))
            mses.append(float(mse))
            maes.append(float(mae))
            phases.append(phase)

    # Create a DataFrame from the lists
    data = pd.DataFrame({
        'Epoch': epochs,
        'Loss': losses,
        'MSE': mses,
        'MAE': maes,
        'Phase': phases
    })

    # Plot the data
    plt.figure(figsize=(12, 8))

    for phase in ["Train", "Val"]:
        plt.plot(data[data['Phase'] == phase]['Epoch'], data[data['Phase'] == phase]['Loss'], label=f'{phase} Loss')
        #plt.plot(data[data['Phase'] == phase]['Epoch'], data[data['Phase'] == phase]['MSE'], label=f'{phase} MSE')
        plt.plot(data[data['Phase'] == phase]['Epoch'], data[data['Phase'] == phase]['MAE'], label=f'{phase} MAE')

    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.title('Training and Validation Losses')
    plt.show()

# Call the function
plot_stats("vgg/bl-sh/traininglog.txt")
