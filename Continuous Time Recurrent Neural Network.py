# Import necessary libraries
import numpy as np  # For numerical operations
import plotly.graph_objects as go  # For plotting

# Define global variables for the simulation
RUN_DURATION = 250  # Total duration of the simulation
NET_SIZE = 2  # Size of the neural network (number of neurons)
STEP_SIZE = 0.01  # Step size for the Euler integration method

class CTRNN:
    """This class defines a Continuous Time Recurrent Neural Network (CTRNN)."""

    def __init__(self, size, step_size=STEP_SIZE):
        """Initialize the CTRNN with default or specified parameters."""
        self.size = size  # Number of neurons in the network
        self.step_size = step_size  # Integration step size
        # Initialize neuron states, outputs, biases, gains, time constants (taus), and synaptic weights
        self.states = np.zeros(size)  # Neuron states initialized to zero
        self.outputs = np.zeros(size)  # Neuron outputs initialized to zero
        self.biases = np.ones(size)  # Neuron biases initialized to one
        self.gains = np.ones(size)  # Neuron gains initialized to one
        self.taus = np.ones(size)  # Neuron time constants initialized to one
        self.weights = np.random.uniform(-2, 2, (size, size))  # Random synaptic weights between -2 and 2

    def euler_step(self, external_inputs):
        """Perform one step of Euler integration to update the network's state."""
        # Compute the change in states using the CTRNN equations
        dydt = (1 / self.taus) * (-self.states + np.dot(self.weights, self.outputs) + self.biases + external_inputs)
        # Update the states based on the change and step size
        self.states += self.step_size * dydt
        # Update the outputs using the sigmoid activation function
        self.outputs = 1 / (1 + np.exp(-self.states))

    def randomize_outputs(self, lb, ub):
        """Randomize the initial outputs of the neurons within specified bounds."""
        self.outputs = np.random.uniform(lb, ub, self.size)

class NetworkParameters:
    """This class is responsible for setting specific parameters of the CTRNN."""

    def __init__(self, network):
        """Initialize with a reference to the CTRNN whose parameters will be set."""
        self.network = network

    def set_parameters(self):
        """Set predefined parameters for the CTRNN."""
        # Set time constants, biases, and specific synaptic weights for demonstration purposes
        self.network.taus = np.array([1., 1.])
        self.network.biases = np.array([-2.75, -1.75])
        self.network.weights[0, 0] = 4.5
        self.network.weights[0, 1] = 1
        self.network.weights[1, 0] = -1
        self.network.weights[1, 1] = 4.5

class NetworkSimulation:
    """This class simulates the dynamics of the CTRNN over time."""

    def __init__(self, network):
        """Initialize with a reference to the CTRNN to be simulated."""
        self.network = network

    def simulate(self):
        """Run the simulation and return the outputs of the network over time."""
        outputs = []
        # Run the simulation for the specified duration and step size
        for _ in range(int(RUN_DURATION/STEP_SIZE)):
            # Update the network state with zero external input
            self.network.euler_step(np.zeros(NET_SIZE))
            # Store a copy of the outputs at each step
            outputs.append(self.network.outputs.copy())
        return np.asarray(outputs)

class PlotOscillatorOutput:
    """This class handles plotting the outputs of the CTRNN."""

    def __init__(self, outputs):
        """Initialize with the outputs obtained from the simulation."""
        self.outputs = outputs

    def plot(self):
        """Generate and display a plot of the neuron outputs over time."""
        fig = go.Figure()
        # Add traces for each neuron's output
        fig.add_trace(go.Scatter(y=self.outputs[:, 0], mode='lines', name='Neuron 1', line=dict(color='black')))
        fig.add_trace(go.Scatter(y=self.outputs[:, 1], mode='lines', name='Neuron 2', line=dict(color='blue')))
        # Update plot layout with titles and labels
        fig.update_layout(title='Continuous Time Recurrent Neural Network Outputs Over Time', xaxis_title='Time Step', yaxis_title='Neuron Output')
        fig.show()  # Display the plot

# Instantiate and configure the CTRNN
network = CTRNN(size=NET_SIZE, step_size=STEP_SIZE)
params = NetworkParameters(network)
params.set_parameters()
# Randomize the initial outputs of the neurons
network.randomize_outputs(0.1, 0.2)
# Simulate the network and obtain outputs
simulation = NetworkSimulation(network)
outputs = simulation.simulate()
# Plot the neuron outputs over time
plot = PlotOscillatorOutput(outputs)
plot.plot()





