# Event-based Spiking Neural Network Pipeline Simulator

This project implements a Python-based spiking neural network (SNN) simulator with an event-based convolution pipeline.
It was part of my masters thesis [Implementation of an Event-based Pipeline for Convolutional Spiking Neural Networks on Programmable Hardware](https://elib.dlr.de/201511/) and serves as a proof of concept for the VHDL hardware design I developed. It is based on the PyTorch implementation [FireNet](https://github.com/tudelft/event_flow) and uses its model and pre-trained parameters provided under the [MIT license](/LICENSE).

## 📚 Background & Context

Unlike conventional CNNs, SNNs don't update neuron values synchronously for the entire layer at once, but instead perform asynchronous neuron updates only for the neurons at which an incoming spike (event) is detected. Given enough incoming spikes, the neurons potential will then be raised over a specified threshold and it will then spike itself, eventually leading to a chain of spike events propagating through the layers with minimal delay.

For further information on SNNs, please refer to [Spiking Neural Networks and Their Applications: A Review](https://www.mdpi.com/2076-3425/12/7/863).

Since this project was primarily a **development aid for the hardware design**, it is **not optimized for efficiency**. Instead, the code structure and logic were deliberately kept as close to the hardware implementation as possible, even if some of the design choices may seem unconventional from a software engineering perspective.

### 🔍 Project Motivation

The sparse spike computation within SNNs holds great potential for energy efficient implementations, especially when used with event-based data as input (e.g. dynamic vision sensor data). However, this rules out the use of normal convolution with matrix multiplication as this directly contradicts the sparse nature of the input data and neuron updates. Because the time and location of spikes is mostly unpredictable, computing them in software causes significant overhead. Application-specific hardware pipelines can perform those calculations more efficiently and concurrently, which is why a VHDL design for an FPGA was needed. Because hardware description languages only provide rudimentary debugging and slow simulations, this Python simulation was used to design the structure of the hardware pipeline.

## 🚀 Project Overview

### ✅ Key Features

- **FireNet Model Integration:** Uses pre-trained LIFFireNet parameters for accurate reference comparisons.
- **Hardware-Oriented SNN Dynamics:** Designed to mirror FireNet’s behavior in a way that can be implemented as a hardware architecture.
- **Custom Spike Processing Pipeline:** Implements convolutional operations with support for recurrent connections.
- **Event-Based Convolution Processing:** Efficiently updates all affected neurons upon receiving a single spike event.
- **Layer-Wise Execution Metrics:** Logs processing times and neuron activity for performance evaluation.
- **Minimal dependencies:** Uses only `numpy` for calculations and `time` for runtime profiling.

### 📁 Project Structure

```
.
├── datasets/          # Input events in hdf5 format
├── models/            # LIF-FireNet source code
├── pretrained/        # LIF-FireNet pre-trained model
├── config.py          # Configuration parameters (paths, constants)
├── dataloader.py      # Loads model parameters and input spike events
├── main.py            # Initializes LIFFireNet topology
├── neurocore.py       # Convolutional layer that instantiates pipelines
├── pipeline.py        # Implements neuron processing pipeline
├── utils.py           # Helper functions (e.g., spike queues, data structures)
└── README.md          # Project documentation
```

### 💻 Usage

To setup the project on Windows in VS Code follow these steps:

1. Create a virtual environment:  `python -m venv .venv`
2. If execution is restricted, run: `Set-ExecutionPolicy Unrestricted -Scope Process`
3. Activate the environment: `.venv\Scripts\Activate`
4. Install the dependencies: `pip install -r requirements.txt`
5. Run the script: `python main.py`

## 🛠️ Implementation

The network follows the FireNet topology, including recurrent layers.
Pre-trained kernel weights, thresholds, and leak rates were used from the LIFFireNet model.
Input event data was provided by DLR, capturing a falling cup using a Prophesee Metavision® EVK4 sensor.

![FireNet](/img/firenet_architecture.png)

The Neurocore is the central processing module responsible for handling spike-based computations. It processes incoming events layer by layer, ensuring that neuron updates follow the event-driven dynamics of Spiking Neural Networks (SNNs). Spikes are grouped into short discrete time steps to ensure comparability with the LIFFireNet implementation. A layer processes all spikes from one time step before advancing to the next.

### ⚙️ Processing Flow

1. Layer Processing
    - The Neurocore loads parameters for one layer at a time and processes all spikes within a short time step before moving to the next layer.
2. Spike Handling
    - Spikes are stored in a FIFO queue and processed sequentially.
    - When a neuron receives a spike, its local receptive field is updated based on kernel weights.
3. Recurrent Spikes
    - Spikes from recurrent layers are delayed and reintroduced into the pipeline.
    - The model ensures that all input spikes are processed before recurrent spikes for a given time step.
4. Threshold Check
    - Any neuron potential that exceeds its threshold generates a spike into the outgoing queue and gets reset.
5. Neuron Leaks
    - After processing spikes, a leak function is applied to all neurons at once, reducing computational overhead.
6. Neuron State Updates
    - The updated neuron values are written back into the layer array.
    - The next layer then processes the newly generated spikes.


![SW modules](/img/sw_architecture.png)

## 💡 Lessons Learned

This project was my first major Python implementation, having previously only worked with small Jupyter Notebook scripts. Initially, the learning curve was steep, but I quickly realized that many of the programming concepts I had learned in C and C++ translated well into Python. This allowed me to write functional code while simultaneously learning Python-specific best practices.

Looking at the project now, I recognize that I took the goal of mirroring the hardware implementation too far. While aligning the software with hardware behavior was necessary, I could have taken further advantage of object-oriented programming (OOP) to improve code structure and maintainability. For example, instead of reusing and reassigning a single Neurocore object for each layer—mimicking hardware module reuse in VHDL—I could have instantiated separate objects for better clarity.

Lastly, even though this code was meant as a proof of concept rather than a production-ready implementation, a more consistent documentation from the start of the project would have been of great value. A more structured approach to docstrings, type annotations, and modular functions would have made the codebase easier to follow, maintain, and extend.

### 🔧 Potential Improvements

- **Refactor inference function** into smaller, modular functions for better readability and maintainability
- **Optimize data structures** by using dictionaries or classes instead of using individual numpy arrays for managing layer parameters
- **Improve type annotations and docstrings** make the code easier to maintain
- **Improve logging functionality** for reference comparison with LIFFireNet
- **Add visualization tools** to illustrate neuron activity and spike propagation

## 🔑 License

This project is for research and development purposes. If you use or modify it, please cite the original [FireNet](https://github.com/tudelft/event_flow) repository and include the [MIT license](/LICENSE).
