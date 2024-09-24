import nengo
import numpy as np
import matplotlib.pyplot as plt

# Create a Nengo model
model = nengo.Network(label="Vector Transformation")
with model:
    # Input nodes providing two signals
    input_node1 = nengo.Node(lambda t: np.sin(2 * np.pi * t))
    input_node2 = nengo.Node(lambda t: np.cos(2 * np.pi * t))
    # Ensemble representing both input signals (2D)
    input_ens = nengo.Ensemble(n_neurons=200, dimensions=2)
    # Connect the input nodes to the ensemble
    nengo.Connection(input_node1, input_ens[0])
    nengo.Connection(input_node2, input_ens[1])
    # Ensemble representing the product output
    output_ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    # Apply the transformation (compute the product of inputs)
    nengo.Connection(input_ens, output_ens, function=lambda x: x[0] * x[1])
    # Probes to record data
    probe_input1 = nengo.Probe(input_node1, synapse=None)
    probe_input2 = nengo.Probe(input_node2, synapse=None)
    probe_output = nengo.Probe(output_ens, synapse=0.01)

# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(sim.trange(), sim.data[probe_input1], label="Input Signal 1 (sin)")
plt.plot(sim.trange(), sim.data[probe_input2], label="Input Signal 2 (cos)")
plt.plot(sim.trange(), sim.data[probe_output], label="Product Output")
plt.title("Vector Transformation: Multiplying Two Signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig("images/part2/ex3.png")
