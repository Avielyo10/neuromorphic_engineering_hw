import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class ModelType(Enum):
    REGULAR_SPIKING = "Regular Spiking (RS)"
    CHATTERING = "Chattering (CH)"
    FAST_SPIKING = "Fast Spiking (FS)"
    INTRINSICALLY_BURSTING = "Intrinsically Bursting (IB)"
    LOW_THRESHOLD_SPIKING = "Low-Threshold Spiking (LTS)"
    RESONATOR = "Resonator (RZ)"
    THALAMO_CORTICAL_LEFT = "Thalamo-Cortical Left Side (TC)"
    THALAMO_CORTICAL_RIGHT = "Thalamo-Cortical Right Side (TC)"


class IzhikevichModel:
    def __init__(self, exp_type: ModelType, a, b, c, d, v0=-70, T=200, dt=0.25):
        self.x = 5
        self.y = 140
        self.exp_type = exp_type
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v0 = v0
        self.T = T
        self.dt = dt
        self.time = np.arange(0, T + dt, dt)
        self.stim = self.__get_stimuli()
        self.trace = np.zeros((2, len(self.time)))

    def plot_model(self):
        if self.exp_type == ModelType.RESONATOR:
            self.__plot_resonator()
        else:
            self.__plot_model()

    def __plot(self):
        plt.figure(figsize=(10, 5))
        plt.title("Izhikevich Model: {}".format(self.exp_type.value), fontsize=15)
        plt.ylabel("Membrane Potential (mV)", fontsize=15)
        plt.xlabel("Time (msec)", fontsize=15)
        plt.plot(self.time, self.trace[0], linewidth=2, label="Vm")
        plt.plot(self.time, self.trace[1], linewidth=2, label="Recovery", color="green")
        plt.plot(
            self.time,
            self.stim + self.v0,
            label="Stimuli (Scaled)",
            color="sandybrown",
            linewidth=2,
        )
        plt.legend(loc=1)
        plt.savefig(f"images/{self.exp_type.value}.png")

    def __plot_resonator(self):
        v = self.v0
        u = self.b * v
        for i in range(len(self.stim)):
            v += self.dt * (0.04 * v**2 + self.x * v + self.y - u + self.stim[i])
            u += self.dt * self.a * (self.b * v - u)
            if v > 30:
                v = self.c
                u += self.d
            self.trace[0, i] = v
            self.trace[1, i] = u
        self.__plot()

    def __plot_model(self):
        v = self.v0
        u = self.b * v
        for i in range(len(self.stim)):
            v += self.dt * (0.04 * v**2 + self.x * v + self.y - u + self.stim[i])
            u += self.dt * self.a * (self.b * v - u)
            if v > 30:
                self.trace[0, i] = 30
                v = self.c
                u += self.d
            else:
                self.trace[0, i] = v
                self.trace[1, i] = u
        self.__plot()

    def __get_stimuli(self):
        match self.exp_type:
            case ModelType.RESONATOR:
                return self.__resonator_stim()
            case ModelType.THALAMO_CORTICAL_LEFT:
                return self.__thalamo_cortical_left_stim()
            case ModelType.THALAMO_CORTICAL_RIGHT:
                return self.__thalamo_cortical_right_stim()
            case _:
                return self.__stim_func()

    def __stim_func(self):
        stim = np.zeros(len(self.time))
        stim[20:] = 10
        return stim

    def __resonator_stim(self):
        small_spike = 20
        start_spike_bump = 300
        spike_bump_length = 20
        stim = np.zeros(len(self.time))
        stim[small_spike:] = 0.2
        stim[start_spike_bump : start_spike_bump + spike_bump_length + 1] = 1
        return stim

    def __thalamo_cortical_left_stim(self):
        stim = np.zeros(len(self.time))
        stim[400:] = 15
        return stim

    def __thalamo_cortical_right_stim(self):
        stim = np.zeros(len(self.time))
        stim[:21] = -15
        return stim


experiments = [
    {
        "exp_type": ModelType.REGULAR_SPIKING,
        "a": 0.02,
        "b": 0.2,
        "c": -65,
        "d": 8,
        "v0": -70,
        "T": 200,
    },
    {
        "exp_type": ModelType.CHATTERING,
        "a": 0.02,
        "b": 0.2,
        "c": -50,
        "d": 2,
        "v0": -70,
        "T": 200,
    },
    {
        "exp_type": ModelType.FAST_SPIKING,
        "a": 0.1,
        "b": 0.2,
        "c": -65,
        "d": 2,
        "v0": -70,
        "T": 200,
    },
    {
        "exp_type": ModelType.INTRINSICALLY_BURSTING,
        "a": 0.02,
        "b": 0.2,
        "c": -55,
        "d": 4,
        "v0": -70,
        "T": 200,
    },
    {
        "exp_type": ModelType.LOW_THRESHOLD_SPIKING,
        "a": 0.02,
        "b": 0.25,
        "c": -65,
        "d": 2,
        "v0": -70,
        "T": 200,
    },
    {
        "exp_type": ModelType.RESONATOR,
        "a": 0.1,
        "b": 0.26,
        "c": -65,
        "d": 2,
        "v0": -62,
        "T": 200,
    },
    {
        "exp_type": ModelType.THALAMO_CORTICAL_LEFT,
        "a": 0.02,
        "b": 0.25,
        "c": -65,
        "d": 8,
        "v0": -63,
        "T": 250,
    },
    {
        "exp_type": ModelType.THALAMO_CORTICAL_RIGHT,
        "a": 0.02,
        "b": 0.25,
        "c": -65,
        "d": 0.05,
        "v0": -87,
        "T": 250,
    },
]
# Run the experiments
for exp in experiments:
    model = IzhikevichModel(
        exp_type=exp["exp_type"],
        a=exp["a"],
        b=exp["b"],
        c=exp["c"],
        d=exp["d"],
        v0=exp["v0"],
        T=exp["T"],
    )
    model.plot_model()
