import nengo
import numpy as np
import matplotlib.pyplot as plt

# Create a Nengo model
model = nengo.Network(label="Neural Integrator")
with model:
    # Input node: provides a constant input of 1.0 for the first 0.5 seconds
    def input_function(t):
        return 1.0 if t < 0.5 else 0.0

    input_node = nengo.Node(input_function)
    # Ensemble representing the integrator state (1-dimensional)
    integrator = nengo.Ensemble(
        n_neurons=200,
        dimensions=1,
        neuron_type=nengo.LIF(),
        max_rates=nengo.dists.Uniform(100, 200),  # Firing rates between 100 and 200 Hz
        intercepts=nengo.dists.Uniform(-0.5, 0.5),  # Intercepts between -0.5 and 0.5
    )
    # Define the recurrent connection (feedback) to implement integration
    tau = 0.1  # Time constant for the integrator
    nengo.Connection(
        integrator,
        integrator,
        synapse=tau,
        transform=1.0,  # Identity transform for feedback
    )
    # Connect the input to the integrator with appropriate scaling
    nengo.Connection(
        input_node, integrator, synapse=None, transform=tau  # Scale input by tau
    )
    # Probes to record data
    probe_input = nengo.Probe(input_node, synapse=None)
    probe_integrator = nengo.Probe(integrator, synapse=0.01)

# Run the simulation
with nengo.Simulator(model) as sim:
    sim.run(1.0)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(sim.trange(), sim.data[probe_input], label="Input Signal")
plt.plot(sim.trange(), sim.data[probe_integrator], label="Integrator Output")
plt.title("Neural Integrator Example")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig("images/part3/ex1.png")
