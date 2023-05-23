import matplotlib.pyplot as plt
import numpy as np



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
    bar_width = 0.35
    num_bins = len(neuronOutputSpikes)
    bar_positions = np.arange(num_bins)

    # Plot the output graph
    #ax2.bar(bar_positions - bar_width/2, neuronOutputSpikes, color='red', alpha=0.7, width=bar_width, label='Pytorch (batch based)')
    ax2.bar(bar_positions + bar_width/2, neuronOutputSpikes, color='blue', alpha=0.7, width=bar_width, label='Own (event based)')
    ax2.set_xlim(0, num_bins)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel('Time Bins')
    ax2.set_ylabel('Output Spikes')
    ax2.legend()

    # Add a title to the figure
    fig.suptitle('Spiking behaviour of a single neuron')
    # Display the figure
    plt.show()
