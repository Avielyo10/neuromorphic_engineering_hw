import nengo
import numpy as np
import matplotlib.pyplot as plt

# Create a Nengo model
model = nengo.Network(label="Nonlinear Transformation")
with model:
    # Input node providing a sine wave
    input_node = nengo.Node(lambda t: np.sin(2 * np.pi * t))
    # Ensemble representing the input signal
    input_ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    # Ensemble representing the squared output
    output_ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    # Connect the input node to the input ensemble
    nengo.Connection(input_node, input_ens)
    # Apply a nonlinear transformation (square the input)
    nengo.Connection(input_ens, output_ens, function=lambda x: x**2)
    # Probes to record data
    probe_input = nengo.Probe(input_node, synapse=None)
    probe_output = nengo.Probe(output_ens, synapse=0.01)

# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(sim.trange(), sim.data[probe_input], label="Input Signal")
plt.plot(sim.trange(), sim.data[probe_output], label="Squared Output")
plt.title("Nonlinear Transformation: Squaring a Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig("images/part2/ex2.png")
