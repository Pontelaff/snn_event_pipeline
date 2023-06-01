import matplotlib.pyplot as plt
import numpy as np
from utils import cropLogs



def plotNeuronActivity(neuronInputSpikes, neuronOutputSpikes):
    """
    This function plots the spiking behavior of a single neuron, including a heatmap of weighted input
    spikes separated by channel and a bar graph of output spikes.

    @param neuronInputSpikes A 2D numpy array representing the accumulated weighted input spikes per
    channel for a single neuron over time.
    @param neuronOutputSpikes The output spikes of a single neuron over time, which is a 1D numpy array.
    """
    neuronInputSpikes = np.transpose(neuronInputSpikes)
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot the stacked graph
    # ax1.stackplot(range(num_bins), data, colors=plt.cm.viridis(np.linspace(0, 1, num_subgraphs)))
    # ax1.set_xlim(0, num_bins)
    # ax1.set_ylim(0, 25)
    # ax1.set_ylabel('Weighted Input Spikes\nper Channel')


    # Plot the input heatmap
    im = ax1.imshow(neuronInputSpikes, cmap='viridis', aspect='auto')
    # Set the x and y-axis labels
    #ax3.set_xlabel('Time Bins')
    ax1.set_ylabel('Weighted Input Spikes\nper Channel')

    # Add a colorbar
    #cbar = fig.colorbar(im, ax=ax1)
    #cbar.set_label('Input Current')
    # Add a title to the plot
    #ax1.set_title('Input Spike Heatmap')

    # Set the width and position of the bars
    num_bins = len(neuronOutputSpikes)
    bar_positions = np.arange(num_bins)

    # Plot the output graph
    ax2.bar(bar_positions, neuronOutputSpikes, color='blue', alpha=0.7, width=0.7, label='Own (event based)')
    ax2.set_xlim(0, num_bins)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel('Time Bins')
    ax2.set_ylabel('Output Spikes')
    ax2.legend()

    # Add a title to the figure
    fig.suptitle('Spiking behaviour of a single neuron')
    # Display the figure
    plt.show()

def compNeuronInput(layerName):
    ownArrPath = "test_sequences/" + layerName + "_inLog.npy"
    pytorchArrPath = "test_sequences/" + layerName + "_input_seq.npy"
    pytorch = np.load(pytorchArrPath)
    pytorchSum = np.sum(pytorch, axis= (-1,-2))
    own = np.load(ownArrPath)

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot the stacked graph
    # ax1.stackplot(range(num_bins), data, colors=plt.cm.viridis(np.linspace(0, 1, num_subgraphs)))
    # ax1.set_xlim(0, num_bins)
    # ax1.set_ylim(0, 25)
    # ax1.set_ylabel('Weighted Input Spikes\nper Channel')


    # Plot the pytorch input heatmap
    im = ax1.imshow(np.transpose(pytorchSum), cmap='viridis', aspect='auto')
    # Set the x and y-axis labels
    #ax3.set_xlabel('Time Bins')
    ax1.set_ylabel('Input Spikes\nper Channel')

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Input Spikes')
    # Add a title to the plot
    ax1.set_title('Pytorch Input Spike Heatmap')

    # Plot the input heatmap of the own implementation
    im = ax2.imshow(np.transpose(own), cmap='viridis', aspect='auto')
    # Set the x and y-axis labels
    #ax3.set_xlabel('Time Bins')
    ax2.set_ylabel('Weighted Input Spikes\nper Channel')

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Input Current')
    # Add a title to the plot
    ax2.set_title('Own Input Spike Heatmap')


    # Add a title to the figure
    #fig.suptitle('Spiking behaviour of a single neuron')
    # Display the figure
    #plt.show()

def compNeuronLogs(layerName, channel):
    ownArrInPath = "test_sequences/" + layerName + "_inLog.npy"
    ownArrOutPath = "test_sequences/" + layerName + "_outLog.npy"
    pytorchArrInPath = "test_sequences/" + layerName + "_input_seq.npy"
    pytorchArrOutPath = "test_sequences/" + layerName + "_output_seq.npy"

    pytorchIn = np.load(pytorchArrInPath)
    ownIn = np.load(ownArrInPath)
    pytorchOutAll = np.load(pytorchArrOutPath)
    ownOutAll = np.load(ownArrOutPath)

    ownOutAll, pytorchOutAll = cropLogs(ownOutAll, pytorchOutAll)
    pytorchInSum = np.sum(pytorchIn, axis= (-1,-2))
    pytorchOut = pytorchOutAll[:, channel]
    ownOut = ownOutAll[:, channel]

     # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot the pytorch input heatmap
    im = ax1.imshow(np.transpose(pytorchInSum), cmap='viridis', aspect='auto')
    # Set the x and y-axis labels
    #ax3.set_xlabel('Time Bins')
    ax1.set_ylabel('Input Spikes\nper Channel')

    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Input Spikes')
    # Add a title to the plot
    ax1.set_title('Pytorch Input Spike Heatmap')

    # # Plot the input heatmap of the own implementation
    # im = ax2.imshow(np.transpose(own), cmap='viridis', aspect='auto')
    # # Set the x and y-axis labels
    # #ax3.set_xlabel('Time Bins')
    # ax2.set_ylabel('Weighted Input Spikes\nper Channel')

    # # Add a colorbar
    # cbar = fig.colorbar(im, ax=ax2)
    # cbar.set_label('Input Current')
    # # Add a title to the plot
    # ax2.set_title('Own Input Spike Heatmap')

    # Set the width and position of the bars
    bar_width = 0.35
    num_bins = len(pytorchOut)
    bar_positions = np.arange(num_bins)

    # Plot the output graph
    ax2.bar(bar_positions - bar_width/2, pytorchOut, color='red', alpha=0.7, width=bar_width, label='Pytorch (batch based)')
    ax2.bar(bar_positions + bar_width/2, ownOut, color='blue', alpha=0.7, width=bar_width, label='Own (event based)')
    ax2.set_xlim(0, num_bins)
    ax2.set_ylim(0, 4)
    ax2.set_xlabel('Time Bins')
    ax2.set_ylabel('Output Spikes')
    ax2.legend()

    disjunctSpikes = pytorchOut != ownOut
    hamming = np.count_nonzero(disjunctSpikes)/len(disjunctSpikes)
    jaccard = np.count_nonzero(disjunctSpikes)/np.count_nonzero(pytorchOut + ownOut)
    print("\nChannel %d\nHamming distance: %f\nJaccard distance: %f\n" %(channel, hamming, jaccard))

    disjunctSpikesAll = pytorchOutAll != ownOutAll
    hammingAll = np.count_nonzero(disjunctSpikesAll)/(len(disjunctSpikesAll)*len(disjunctSpikesAll[0]))
    jaccardAll = np.count_nonzero(disjunctSpikesAll)/np.count_nonzero(pytorchOutAll + ownOutAll)
    print("Layer %s\nHamming distance: %f\nJaccard distance: %f\n" %(layerName, hammingAll, jaccardAll))


    # Add a title to the figure
    #fig.suptitle('Spiking behaviour of a single neuron')
    # Display the figure
    plt.show()

    #plt.close()

    return

def plotThresholdComp(jacDistance, hamDistance, thresholds):
    # Plotting the first line graph
    plt.plot(thresholds, jacDistance, label='Jaccard')

    # Plotting the second line graph
    plt.plot(thresholds, hamDistance, label='Hamming')

    # Adding labels and title
    plt.xlabel('Thresholds')
    plt.ylabel('Distance')
    #plt.title('Two Line Graphs')

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show()