from matplotlib import pyplot as plt

def plot_multiple_channel_eeg_data(data):
    plt.figure(figsize=(16, 8))

    for i in range(data.shape[1]):
        plt.plot(data[i], label=f"channel {i}")
        plt.title("EEG Data")
        plt.xlabel("Time")

