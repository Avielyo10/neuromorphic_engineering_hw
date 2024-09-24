import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def lif_model(Rm: int = 1, Cm: int = 5, I: float = 0.2, vTh: int = -40) -> tuple:
    """
    Returns the membrane potential of a LIF model
    Parameters:
    Rm: int - Membrane Resistance [kOhm], default 1
    Cm: int - Capacitance [uF], default 5
    I: float - Current stimulus [mA], default 0.2
    vTh: int - Spike threshold [mV], default -40

    Returns:
    time: np.array - Time array [mSec]
    Vm: np.array - Membrane potential [V]
    frequency: float - Firing rate [Hz]
    """
    # Model Parameters
    T = 50  # Simulation time          [mSec]
    dt = 0.1  # Simulation time interval [mSec]
    t_init = 0  # Stimulus init time       [V]
    vRest = -70  # Resting potential        [mV]
    tau_ref = 1  # Repreactory Period       [mSec]
    vSpike = 50  # Spike voltage            [mV]

    # Simulation parameters
    time = np.arange(0, T * 1e-3 + dt * 1e-3, dt * 1e-3)  # Time array
    Vm = np.ones(len(time)) * vRest * 1e-3  # Membrane voltage array
    tau_m = Rm * 1e3 * Cm * 1e-6  # Time constant
    spikes = []  # Spikes timings

    # Defining the stimulus
    stim = I * 1e-3 * signal.windows.triang(len(time))  # Triangular stimulation pattern

    # Simulating the LIF model
    for i, t in enumerate(time[:-1]):
        if t > t_init:
            uinf = vRest * 1e-3 + Rm * 1e3 * stim[i]
            Vm[i + 1] = uinf + (Vm[i] - uinf) * np.exp(-dt * 1e-3 / tau_m)
            if Vm[i] >= vTh * 1e-3:  # Spike
                spikes.append(t * 1e3)
                Vm[i] = vSpike * 1e-3
                t_init = t + tau_ref * 1e-3
    frequency = 1 / (np.mean(np.diff(spikes)) * 1e-3) if len(spikes) > 1 else 0
    return time, Vm, frequency


def plot_if_curve():
    Rm_values = [1, 5, 10]
    Cm_values = [1, 5, 10]
    I_values = np.linspace(0, 5, 100)  # Input current range (mA)

    plt.figure()
    for Rm, Cm in zip(Rm_values, Cm_values):
        f_values = [
            frequency
            for _, _, frequency in [lif_model(Rm=Rm, Cm=Cm, I=I) for I in I_values]
        ]
        plt.plot(I_values, f_values, label=f"Tau = {Rm * Cm} ms")

    plt.xlabel("Input Current (mA)")
    plt.ylabel("Firing Rate (Hz)")
    plt.title("I-F Curves for Different Tau Values")
    plt.legend()
    plt.savefig("images/I-F.png")


def plot_vt_curves():
    thresholds = [-60, -20, 20]  # Different thresholds in mV
    current = 0.1  # Input current (mA)
    plt.figure(figsize=(10, 10))
    for vTh in thresholds:
        time, Vm, _ = lif_model(I=current, vTh=vTh)
        plt.subplot(3, 1, thresholds.index(vTh) + 1)
        plt.plot(time * 1e3, Vm * 1e3)
        plt.axhline(y=vTh, color="r", linestyle="--", label=f"Threshold = {vTh} mV")
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Voltage (mV)")
        plt.title(f"V-T Curves for Different Thresholds (I = {current} mA)")
        plt.legend()
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.savefig("images/V-T.png")


if __name__ == "__main__":
    plot_if_curve()
    plot_vt_curves()
