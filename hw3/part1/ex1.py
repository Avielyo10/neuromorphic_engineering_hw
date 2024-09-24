import nengo
import numpy as np
import matplotlib.pyplot as plt

# Create a Nengo model
model = nengo.Network()
with model:
    # Input node providing a sine wave
    input_node = nengo.Node(lambda t: np.sin(2 * np.pi * t))
    # Ensemble of 100 neurons representing the input
    ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    # Connect the input node to the ensemble
    nengo.Connection(input_node, ens)
    # Probes to record data
    probe_input = nengo.Probe(input_node)
    probe_output = nengo.Probe(ens, synapse=0.01)

# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)

# Plot the results
plt.figure()
plt.plot(sim.trange(), sim.data[probe_input], label="Input Signal")
plt.plot(sim.trange(), sim.data[probe_output], label="Decoded Output")
plt.title("Encoding and Decoding a Scalar Value")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.savefig("images/part1/ex1.png")
