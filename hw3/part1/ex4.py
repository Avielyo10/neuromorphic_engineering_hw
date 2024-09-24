import nengo
import numpy as np
import matplotlib.pyplot as plt

# Create a Nengo model with 10 neurons
model = nengo.Network()
with model:
    input_node = nengo.Node(lambda t: np.sin(2 * np.pi * t))
    ens = nengo.Ensemble(n_neurons=10, dimensions=1)
    nengo.Connection(input_node, ens)
    probe_input = nengo.Probe(input_node)
    probe_output = nengo.Probe(ens, synapse=0.01)

# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)

# Plot the results
plt.figure()
plt.plot(sim.trange(), sim.data[probe_input], label="Input Signal")
plt.plot(sim.trange(), sim.data[probe_output], label="Decoded Output with 10 Neurons")
plt.title("Representation with 10 Neurons")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.savefig("images/part1/ex4-1.png")

# Update the number of neurons to 100
model = nengo.Network()
with model:
    input_node = nengo.Node(lambda t: np.sin(2 * np.pi * t))
    ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(input_node, ens)
    probe_input = nengo.Probe(input_node)
    probe_output = nengo.Probe(ens, synapse=0.01)
# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)
# Plot the results
plt.figure()
plt.plot(sim.trange(), sim.data[probe_input], label="Input Signal")
plt.plot(sim.trange(), sim.data[probe_output], label="Decoded Output with 100 Neurons")
plt.title("Representation with 100 Neurons")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.savefig("images/part1/ex4-2.png")

# Update the number of neurons to 10,000
model = nengo.Network()
with model:
    input_node = nengo.Node(lambda t: np.sin(2 * np.pi * t))
    ens = nengo.Ensemble(n_neurons=10000, dimensions=1)
    nengo.Connection(input_node, ens)
    probe_input = nengo.Probe(input_node)
    probe_output = nengo.Probe(ens, synapse=0.01)
# Run the simulation (Note: This may be computationally intensive)
with nengo.Simulator(model) as sim:
    sim.run(1.0)
# Plot the results
plt.figure()
plt.plot(sim.trange(), sim.data[probe_input], label="Input Signal")
plt.plot(
    sim.trange(), sim.data[probe_output], label="Decoded Output with 10,000 Neurons"
)
plt.title("Representation with 10,000 Neurons")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.savefig("images/part1/ex4-3.png")
