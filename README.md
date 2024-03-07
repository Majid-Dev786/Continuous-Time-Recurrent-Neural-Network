# Continuous Time Recurrent Neural Network (CTRNN)

## Description

This project implements a Continuous Time Recurrent Neural Network (CTRNN), designed to simulate dynamic systems and neural processes over continuous time. 

The CTRNN model is capable of exhibiting complex dynamics such as oscillations, chaos, and fixed-point behaviors, making it a powerful tool for research in computational neuroscience, robotics, and dynamic system analysis. 

The implementation is done in Python, leveraging NumPy for efficient numerical computations and Plotly for dynamic visualizations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Installation

To use this CTRNN model, you need to have Python installed on your system along with the necessary libraries: NumPy and Plotly. 

You can install these dependencies via pip:

```bash
pip install numpy plotly
```

Clone the repository to your local machine:

```bash
git clone https://github.com/Sorena-Dev/Continuous-Time-Recurrent-Neural-Network.git
```

Navigate into the cloned directory:

```bash
cd Continuous-Time-Recurrent-Neural-Network
```

## Usage

To run the simulation and visualize the network's dynamics, execute the script:

```bash
python "Continuous Time Recurrent Neural Network.py"
```

This script initializes a CTRNN with predefined parameters, runs a simulation over a specified duration, and plots the outputs of the neurons over time.

### Real World Scenario

In real-world scenarios, CTRNNs can be applied to simulate biological neural systems, control robotic systems, or model time-varying dynamic systems. 

For example, you can use this model to study the behavior of neural circuits in animals or to develop control systems for robots that require adaptive behaviors over continuous time.

## Features

- **Customizable Network Structure**: Define the size and timestep of the neural network according to your simulation needs.
- **Dynamic State Update**: Utilizes the Euler method for numerical integration to update neuron states over continuous time.
- **Visualization**: Includes functionality to plot the outputs of the neurons, showcasing the dynamic behavior of the network over time.
- **Parameter Adjustment**: Allows for the adjustment of neuron parameters such as biases, time constants (taus), and synaptic weights to explore different network behaviors.


