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
    if len(pytorchOutAll.shape) > 2:
        pytorchOutAll = pytorchOutAll[:,:,1,1]
    ownOutAll = np.load(ownArrOutPath)

    ownOutAll, pytorchOutAll = cropLogs(ownOutAll, pytorchOutAll)
    pytorchInSum = np.sum(pytorchIn, axis= (-1,-2))
    pytorchOut = pytorchOutAll[:150, channel]
    ownOut = ownOutAll[:150, channel]

     # Create the figure and subplots
    plt.rcParams.update({'font.size': 18})
    fig, (ax1, ax3, ax2) = plt.subplots(3, 1, figsize=(8, 6))

    # Plot the pytorch input heatmap
    im = ax1.imshow(np.transpose(ownIn[:150]), cmap='viridis', aspect='auto')
    # Set the x and y-axis labels
    #ax3.set_xlabel('Time Bins')
    ax1.set_ylabel('Input Spikes\nper Channel')

    # Add a colorbar
    #cbar = fig.colorbar(im, ax=ax1)
    #cbar.set_label('Input Spikes')
    # Add a title to the plot
    ax1.set_title('Neuron Activity\nLayer ' + layerName + ' Channel ' + str(channel))

    # # Plot the input heatmap of the own implementation
    # im = ax2.imshow(np.transpose(own), cmap='viridis', aspect='auto')
    # # Set the x and y-axis labels
    # #ax3.set_xlabel('Time Bins')
    # ax2.set_ylabel('Weighted Input Spikes\nper Channel')

    # # Add a colorbar
    # cbar = fig.colorbar(im, ax=ax2)
    # cbar.set_label('Input Current')
    # # Add a title to the plot

    # Set the width and position of the bars
    bar_width = 0.35
    num_bins = len(pytorchOut)
    bar_positions = np.arange(num_bins)

    # Plot the output graph
    ax3.plot(bar_positions, ownOut[:,0], color='red', alpha=0.7)
    ax3.axhline(y = 0.95, color = 'gray', linestyle = '--')
    ax3.set_xlim(0, num_bins)
    #ax3.set_ylim(-5, 2)
    #ax3.set_xlabel('Time Bins')
    ax3.set_ylabel('Membrane\nPotential')
    #ax2.set_title('Neuron Activity Level')

    # Plot the output graph
    ax2.bar(bar_positions - bar_width/2, pytorchOut, color='blue', alpha=0.7, width=bar_width, label='Pytorch (batch based)')
    ax2.bar(bar_positions + bar_width/2, ownOut[:,1], color='red', alpha=0.7, width=bar_width, label='Own (event based)')
    ax2.set_xlim(0, num_bins)
    ax2.set_ylim(0, 1.5)
    ax2.set_xlabel('Time Bins')
    ax2.set_ylabel('Output\nSpikes')
    #ax2.legend()
    #ax2.set_title('Output Spike Comparison')


    #disjunctSpikes = pytorchOut != ownOut[:,1]
    matchingSpikes = np.logical_and(ownOut[:,1], pytorchOut)
    jaccard = np.count_nonzero(matchingSpikes)/np.count_nonzero(pytorchOut + ownOut[:,1])
    print("\nChannel %d\nJaccard distance: %f\n" %(channel, jaccard))

    matchingSpikesAll = np.logical_and(ownOutAll[:,:,1], pytorchOutAll)
    jaccardAll = np.count_nonzero(matchingSpikesAll)/np.count_nonzero(pytorchOutAll + ownOutAll[:,:,1])
    print("Layer %s\nJaccard distance: %f\n" %(layerName, jaccardAll))

    fig.set_size_inches(w=15, h=7)
    plt.subplots_adjust(bottom=0.13, right=0.97, left=0.09, top=0.97, hspace=0.42)

    # Add a title to the figure
    #fig.suptitle('Spiking behaviour of a single neuron')
    # Display the figure
    plt.show()

    #plt.close()

    return
