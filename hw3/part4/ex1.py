import nengo
import numpy as np
import matplotlib.pyplot as plt

model = nengo.Network(label="Forward Kinematics")
with model:
    # Define time-varying joint angles (for demonstration)
    theta1_node = nengo.Node(lambda t: np.sin(t))
    theta2_node = nengo.Node(lambda t: np.cos(t))
    theta3_node = nengo.Node(lambda t: np.sin(2 * t))

    # Ensembles representing each joint angle
    theta1 = nengo.Ensemble(n_neurons=100, dimensions=1, label="Theta1")
    theta2 = nengo.Ensemble(n_neurons=100, dimensions=1, label="Theta2")
    theta3 = nengo.Ensemble(n_neurons=100, dimensions=1, label="Theta3")

    # Connect input nodes to ensembles
    nengo.Connection(theta1_node, theta1)
    nengo.Connection(theta2_node, theta2)
    nengo.Connection(theta3_node, theta3)

    # Compute sums of angles
    sum12 = nengo.Ensemble(n_neurons=200, dimensions=1, label="Theta1 + Theta2")
    nengo.Connection(theta1, sum12)
    nengo.Connection(theta2, sum12)

    sum123 = nengo.Ensemble(
        n_neurons=200, dimensions=1, label="Theta1 + Theta2 + Theta3"
    )
    nengo.Connection(sum12, sum123)
    nengo.Connection(theta3, sum123)

    # Compute cosines
    cos_theta1 = nengo.Ensemble(n_neurons=200, dimensions=1, label="cos(Theta1)")
    nengo.Connection(theta1, cos_theta1, function=lambda x: np.cos(x))

    cos_sum12 = nengo.Ensemble(
        n_neurons=200, dimensions=1, label="cos(Theta1 + Theta2)"
    )
    nengo.Connection(sum12, cos_sum12, function=lambda x: np.cos(x))

    cos_sum123 = nengo.Ensemble(
        n_neurons=200, dimensions=1, label="cos(Theta1 + Theta2 + Theta3)"
    )
    nengo.Connection(sum123, cos_sum123, function=lambda x: np.cos(x))

    # Compute sines
    sin_theta1 = nengo.Ensemble(n_neurons=200, dimensions=1, label="sin(Theta1)")
    nengo.Connection(theta1, sin_theta1, function=lambda x: np.sin(x))

    sin_sum12 = nengo.Ensemble(
        n_neurons=200, dimensions=1, label="sin(Theta1 + Theta2)"
    )
    nengo.Connection(sum12, sin_sum12, function=lambda x: np.sin(x))

    sin_sum123 = nengo.Ensemble(
        n_neurons=200, dimensions=1, label="sin(Theta1 + Theta2 + Theta3)"
    )
    nengo.Connection(sum123, sin_sum123, function=lambda x: np.sin(x))

    # Compute x and y coordinates
    x_coord = nengo.Ensemble(n_neurons=200, dimensions=1, label="x")
    nengo.Connection(cos_theta1, x_coord)
    nengo.Connection(cos_sum12, x_coord)
    nengo.Connection(cos_sum123, x_coord)

    y_coord = nengo.Ensemble(n_neurons=200, dimensions=1, label="y")
    nengo.Connection(sin_theta1, y_coord)
    nengo.Connection(sin_sum12, y_coord)
    nengo.Connection(sin_sum123, y_coord)

    # Probes for joint angles
    probe_theta1 = nengo.Probe(theta1, synapse=0.01)
    probe_theta2 = nengo.Probe(theta2, synapse=0.01)
    probe_theta3 = nengo.Probe(theta3, synapse=0.01)

    # Probes for x and y coordinates
    probe_x = nengo.Probe(x_coord, synapse=0.01)
    probe_y = nengo.Probe(y_coord, synapse=0.01)

# Create the simulator and run the model
with nengo.Simulator(model) as sim:
    sim.run(5.0)  # Run for 5 seconds

# Extract data
t = sim.trange()
theta1_data = sim.data[probe_theta1]
theta2_data = sim.data[probe_theta2]
theta3_data = sim.data[probe_theta3]
x_data = sim.data[probe_x]
y_data = sim.data[probe_y]

# Plot joint angles
plt.figure(figsize=(12, 4))
plt.plot(t, theta1_data, label="Theta1")
plt.plot(t, theta2_data, label="Theta2")
plt.plot(t, theta3_data, label="Theta3")
plt.title("Joint Angles Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Angle (radians)")
plt.legend()
plt.grid(True)
plt.savefig("images/part4/joint_angles.png")

# Plot x and y coordinates
plt.figure(figsize=(12, 4))
plt.plot(t, x_data, label="x")
plt.plot(t, y_data, label="y")
plt.title("End Effector Coordinates Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Coordinate Value")
plt.legend()
plt.grid(True)
plt.savefig("images/part4/end_effector_coordinates.png")

# Plot x vs y to visualize the trajectory
plt.figure(figsize=(6, 6))
plt.plot(x_data, y_data)
plt.title("End Effector Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.savefig("images/part4/end_effector_trajectory.png")
