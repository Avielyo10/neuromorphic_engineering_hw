import nengo
import numpy as np
import matplotlib.pyplot as plt

# Create a Nengo model
model = nengo.Network(label="Linear Transformation")
with model:
    # Input node providing a sine wave
    input_node = nengo.Node(lambda t: np.sin(2 * np.pi * t))
    # Ensemble representing the input signal
    input_ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    # Ensemble representing the scaled output
    output_ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    # Connect the input node to the input ensemble
    nengo.Connection(input_node, input_ens)
    # Apply a linear transformation (scale by 2)
    nengo.Connection(input_ens, output_ens, function=lambda x: 2 * x)
    # Probes to record data
    probe_input = nengo.Probe(input_node, synapse=0.01)
    probe_output = nengo.Probe(output_ens, synapse=0.01)
# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)
# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(sim.trange(), sim.data[probe_input], label="Input Signal")
plt.plot(sim.trange(), sim.data[probe_output], label="Scaled Output (x2)")
plt.title("Linear Transformation: Scaling a Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig("images/part2/ex1.png")
