import nengo
import numpy as np
import matplotlib.pyplot as plt

# Create a Nengo model
model = nengo.Network()
with model:
    # Input node providing a sine wave
    input_node = nengo.Node(lambda t: np.sin(2 * np.pi * t))
    # Ensemble with LIF neurons
    ens = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.LIF())
    # Connect the input node to the ensemble
    nengo.Connection(input_node, ens)
    # Output node to receive the decoded signal
    output_node = nengo.Node(size_in=1)
    # Connect the ensemble to the output node
    nengo.Connection(ens, output_node, synapse=0.1)
    # Probes to record data
    probe_output = nengo.Probe(output_node, synapse=0.01)

# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)

# Plot the results
plt.figure()
plt.plot(sim.trange(), sim.data[probe_output], label="Decoded Output")
plt.title("Encoding and Decoding with LIF Neurons")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.savefig("images/part1/ex3.png")
