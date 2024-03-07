import numpy as np
import plotly.graph_objects as go

# Global Variables
RUN_DURATION = 250
NET_SIZE = 2
STEP_SIZE = 0.01

class CTRNN:
    def __init__(self, size, step_size=STEP_SIZE):
        self.size = size
        self.step_size = step_size
        self.states = np.zeros(size)
        self.outputs = np.zeros(size)
        self.biases = np.ones(size)
        self.gains = np.ones(size)
        self.taus = np.ones(size)
        self.weights = np.random.uniform(-2, 2, (size, size))

    def euler_step(self, external_inputs):
        dydt = (1 / self.taus) * (-self.states + np.dot(self.weights, self.outputs) + self.biases + external_inputs)
        self.states += self.step_size * dydt
        self.outputs = 1 / (1 + np.exp(-self.states))

    def randomize_outputs(self, lb, ub):
        self.outputs = np.random.uniform(lb, ub, self.size)

class NetworkParameters:
    def __init__(self, network):
        self.network = network

    def set_parameters(self):
        self.network.taus = np.array([1., 1.])
        self.network.biases = np.array([-2.75,-1.75])
        self.network.weights[0,0] = 4.5
        self.network.weights [0,1]= 1
        self.network.weights[1,0] = -1
        self.network.weights[1,1] = 4.5

class NetworkSimulation:
    def __init__(self, network):
        self.network = network

    def simulate(self):
        outputs = []
        for _ in range(int(RUN_DURATION/STEP_SIZE)):
            self.network.euler_step(np.zeros(NET_SIZE))
            outputs.append(self.network.outputs.copy())
        return np.asarray(outputs)

class PlotOscillatorOutput:
    def __init__(self, outputs):
        self.outputs = outputs

    def plot(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.outputs[:, 0], mode='lines', name='Neuron 1', line=dict(color='black')))
        fig.add_trace(go.Scatter(y=self.outputs[:, 1], mode='lines', name='Neuron 2', line=dict(color='blue')))
        fig.update_layout(title='Continuous Time Recurrent Neural Network Outputs Over Time', xaxis_title='Time Step', yaxis_title='Neuron Output')
        fig.show()


network = CTRNN(size=NET_SIZE, step_size=STEP_SIZE)


params = NetworkParameters(network)
params.set_parameters()


network.randomize_outputs(0.1, 0.2)


simulation = NetworkSimulation(network)
outputs = simulation.simulate()


plot = PlotOscillatorOutput(outputs)
plot.plot()
