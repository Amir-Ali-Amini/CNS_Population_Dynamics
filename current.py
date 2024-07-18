# =====================================================================================================
# =====================================================================================================
# =====================================================================================================
# ====================================  AMIR ALI AMINI ================================================
# ====================================    610399102    ================================================
# =====================================================================================================
# =====================================================================================================


import pymonntorch as pmt
import torch
import random
import numpy as np


class SteadyCurrent(pmt.Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value", 6)
        self.noise_range = self.parameter("noise_range", 0)
        ng.I_inp = ng.vector(self.value)

    def forward(self, ng):
        ng.I_inp = (
            ng.vector(self.value) + (ng.vector("normal(-0,0.5)")) * self.noise_range
        )
        # pass # no change


class StepCurrent(pmt.Behavior):
    def initialize(self, ng):
        self.t0 = self.parameter("t0", 200, required=True)
        self.noise_range = self.parameter("noise_range", 0)
        self.value = self.parameter("value", 6)
        ng.I_inp = ng.vector(0)

    def forward(self, ng):
        if (ng.network.iteration * ng.network.dt) > self.t0:
            ng.I_inp = (
                ng.vector(self.value) + (ng.vector("normal(-0,0.5)")) * self.noise_range
            )  # increase current at t0
        else:
            ng.I_inp = (
                ng.vector(0) + (ng.vector("normal(-0,0.5)")) * self.noise_range
            )  # increase current at t0


class SinCurrent(pmt.Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value", 6) / 2
        self.noise_range = self.parameter("noise_range", 0)
        self.stretch_variable = self.parameter("stretch_variable", 1)
        ng.I_inp = ng.vector(0)

    def forward(self, ng):
        I = (
            torch.sin(
                ng.vector(ng.network.iteration * ng.network.dt / self.stretch_variable)
            )
            * self.value
            + self.value
        )
        ng.I_inp = I + (ng.vector("normal(-0,0.5)")) * self.noise_range


class UniformCurrent(pmt.Behavior):
    def initialize(self, ng):
        self.max_current = self.parameter("value", 6) * 2
        self.step = self.parameter("step", 0.5)
        self.noise_range = self.parameter("noise_range", 0)
        self.initial_current = self.parameter("initial_current", None)
        ng.I_inp = ng.vector("uniform") * self.max_current
        if self.initial_current != None:
            ng.I_inp = ng.vector(self.initial_current / 100) * self.max_current / 2

    def forward(self, ng):
        I = (ng.vector("uniform") - (ng.I_inp / self.max_current)) * self.step
        ng.I_inp += I + (ng.vector("normal(-0,0.5)")) * self.noise_range


class UniformSingleCurrent(pmt.Behavior):
    def initialize(self, ng):
        self.max_current = self.parameter("value", 6) * 2
        self.step = self.parameter("step", 0.5)
        self.noise_range = self.parameter("noise_range", 0)
        self.initial_current = self.parameter("initial_current", None)
        self.t0 = self.parameter("t0", -1) / ng.network.dt
        self.t1 = self.parameter("t1", -1) / ng.network.dt
        self.c0 = self.parameter("c0", 0) / ng.network.dt
        self.c1 = self.parameter("c1", 0) / ng.network.dt

        ng.I_inp = ng.vector(random.random()) * self.max_current
        if self.t1 == -1:
            self.t1 = self.t0

        if self.initial_current != None:
            ng.I_inp = (ng.vector(self.initial_current) / 100) * self.max_current / 2

    def forward(self, ng):
        if ng.network.iteration * ng.network.dt < self.t0:
            I = 0
            ng.I_inp = ng.vector(self.c0)
        elif ng.network.iteration * ng.network.dt < self.t1:
            I = 0
            ng.I_inp = ng.vector(self.c1)
        elif ng.network.iteration * ng.network.dt == self.t1:
            ng.I_inp = ng.vector(random.random()) * self.max_current
            if self.initial_current != None:
                ng.I_inp = (
                    (ng.vector(self.initial_current) / 100) * self.max_current / 2
                )
            I = 0
        else:
            I = (random.random() - (ng.I_inp / self.max_current)) * self.step
        ng.I_inp += I + (random.random() - 0.5) * self.noise_range


class FlagFunction(pmt.Behavior):
    def initialize(self, ng):
        self.offset = self.parameter("value")
        ng.I_inp = ng.vector(0.0)
        self.steps = self.parameter("step", 1000)
        self.t = 0

    def forward(self, ng):
        if (self.t - self.steps / 2) % self.steps == 0:
            ng.I_inp = ng.vector(1.0)
        elif (self.t - self.steps / 2) % self.steps == 1:
            ng.I_inp = ng.vector(0.0)
        self.t += 1


class IncreasingCurrent(pmt.Behavior):
    def initialize(self, ng):
        self.offset = self.parameter("value")
        self.increasing_value = self.parameter("increasing_value", 0.001)
        ng.I_inp = ng.vector(mode=self.offset)

    def forward(self, ng):
        ng.I_inp += self.increasing_value


class PreDefinedCurrent(pmt.Behavior):
    def initialize(self, ng):
        self.iterations = self.parameter("iterations", 100, required=True) + 1
        self.mean = self.parameter("value", 0.0)
        self.std = self.parameter("std", 1.0)

        ng.I_inp = ng.vector()
        self.makeCurrent(self.mean, self.std, self.iterations)

    def forward(self, ng):
        ng.I_inp = ng.vector(float(self.whole_current[ng.network.iteration]))

    def makeCurrent(self, mean, std, size):
        white_noise = np.random.normal(0, 1, size)

        # Generate cumulative sum to simulate Brownian motion
        brownian_motion = np.cumsum(white_noise)

        # Adjust mean and std
        adjusted_brownian_motion = (
            brownian_motion - np.mean(brownian_motion)
        ) / np.std(brownian_motion)

        # Scale to desired mean and std
        scaled_brownian_noise = adjusted_brownian_motion * std + mean

        self.whole_current = scaled_brownian_noise
