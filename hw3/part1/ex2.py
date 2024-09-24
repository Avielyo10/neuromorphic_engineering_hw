import nengo
import numpy as np
import matplotlib.pyplot as plt

# Create a Nengo model
model = nengo.Network()
with model:
    # Input node providing a 2D signal
    input_node = nengo.Node(lambda t: [np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    # Ensemble representing a 2D vector
    ens = nengo.Ensemble(n_neurons=200, dimensions=2)
    # Connect the input node to the ensemble
    nengo.Connection(input_node, ens)
    # Node to receive the decoded product
    product_node = nengo.Node(size_in=1)
    # Connection decoding the product of the two inputs
    nengo.Connection(ens, product_node, function=lambda x: x[0] * x[1])
    # Probes to record data
    probe_product = nengo.Probe(product_node, synapse=0.01)

# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)

# Plot the results
plt.figure()
plt.plot(sim.trange(), sim.data[probe_product], label="Decoded Product")
plt.title("Decoding a Nonlinear Function of a 2D Input")
plt.xlabel("Time (s)")
plt.ylabel("Product Value")
plt.legend()
plt.savefig("images/part1/ex2.png")
