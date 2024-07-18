# =====================================================================================================
# =====================================================================================================
# =====================================================================================================
# ====================================  AMIR ALI AMINI ================================================
# ====================================    610399102    ================================================
# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

import pymonntorch as pmt
import torch


class BaseModel(pmt.Behavior):
    def initialize(self, ng):
        # base LIF parameters :
        self.threshold = self.parameter("threshold", -55, required=True)
        self.u_rest = self.parameter("u_rest", -65)
        self.u_reset = self.parameter("u_reset", -73.42)
        self.R = self.parameter("R", 1.7)
        self.tau_m = self.parameter("tau_m", 10)
        self.refractory_period = self.parameter("refractory_period", 0) / ng.network.dt

        # ELIF extra parameters:
        self.u_rh = self.parameter("u_rh", -60)
        self.delta_t = self.parameter("delta_t", 0)

        # AELIF extra parameters:
        self.a = self.parameter("a", 0)
        self.b = self.parameter("b", 0)
        self.tau_w = self.parameter("tau_w", 0)

        # initializing neuron group variables"
        ng.u = ng.vector("uniform") * (self.threshold - self.u_reset) + self.u_reset
        ng.spike = ng.u > self.threshold
        ng.u[ng.spike] = self.u_reset

        ng.w = ng.vector(0)

        ng.refractory_time = ng.vector(-self.refractory_period - 1)

        # ng.spike_counter = ng.vector(0)

    def forward(self, ng):
        ng.u += (
            ng.network.dt
            * (
                self.F(ng)
                + self.R
                * (ng.I * (ng.refractory_time < ng.network.iteration).byte() - ng.w)
            )
            / self.tau_m
        )
        ng.spike = ng.u > self.threshold
        ng.u[ng.spike] = self.u_reset
        ng.refractory_time[ng.spike] = ng.network.iteration + self.refractory_period

        # ng.spike_counter += ng.spike.byte()

        self.refresh_w(ng)

    # F_u:
    def F(self, ng):
        leakage = ng.u - self.u_rest
        exp_part = self.delta_t * torch.exp((ng.u - self.u_rh) / (self.delta_t or 1))
        result = -leakage + exp_part
        return result

    # update w
    def refresh_w(self, ng):
        if self.a or self.b:
            leakage = ng.u - self.u_rest
            # sigma_part = self.b * self.tau_w  * ng.spike_counter
            sigma_part = self.b * self.tau_w * ng.spike.byte()

            ng.w += ng.network.dt * (self.a * leakage - ng.w + sigma_part) / self.tau_w


class InputModel(pmt.Behavior):  # no current, no behavior
    def initialize(self, ng):
        # base LIF parameters :
        self.threshold = self.parameter("threshold", -55, required=True)
        self.u_rest = self.parameter("u_rest", -65)
        self.u_reset = self.parameter("u_reset", -73.42)
        self.R = self.parameter("R", 1.7)
        self.tau_m = self.parameter("tau_m", 10)
        self.refractory_period = self.parameter("refractory_period", 0) / ng.network.dt

        # ELIF extra parameters:
        self.u_rh = self.parameter("u_rh", -60)
        self.delta_t = self.parameter("delta_t", 0)

        # AELIF extra parameters:
        self.a = self.parameter("a", 0)
        self.b = self.parameter("b", 0)
        self.tau_w = self.parameter("tau_w", 0)

        ng.spike = ng.vector(0) != 0
        # initializing neuron group variables"
        ng.u = ng.vector(0)

        ng.w = ng.vector(0)

        ng.refractory_time = ng.vector(-self.refractory_period - 1)

    def forward(self, ng):
        ng.spike = ng.vector(0) != 0


# LIF Model
def LIF(
    threshold=-55, u_rest=-65, u_reset=-70, R=1.7, tau_m=10, tag="", refractory_period=0
):
    return BaseModel(
        threshold=threshold,
        u_reset=u_reset,
        u_rest=u_rest,
        R=R,
        tau_m=tau_m,
        tag="LIF" + (f"_({tag})" if tag else ""),
        refractory_period=refractory_period,
    )


# ELIF Model
def ELIF(
    threshold=+30,
    u_rest=-65,
    u_reset=-70,
    R=1.7,
    tau_m=10,
    u_rh=-45,
    delta_t=1,
    tag="",
    refractory_period=0,
):
    return BaseModel(
        threshold=threshold,
        u_reset=u_reset,
        u_rest=u_rest,
        R=R,
        tau_m=tau_m,
        u_rh=u_rh,
        delta_t=delta_t,
        tag="ELIF" + (f"_({tag})" if tag else ""),
        refractory_period=refractory_period,
    )


# AELIF Model
def AELIF(
    threshold=+30,
    u_rest=-65,
    u_reset=-70,
    R=1.7,
    tau_m=10,
    u_rh=-45,
    delta_t=1,
    a=1,
    b=1,
    tau_w=10,
    tag="",
    refractory_period=0,
):
    return BaseModel(
        threshold=threshold,
        u_reset=u_reset,
        u_rest=u_rest,
        R=R,
        tau_m=tau_m,
        u_rh=u_rh,
        delta_t=delta_t,
        a=a,
        b=b,
        tau_w=tau_w,
        tag="AELIF" + (f"_({tag})" if tag else ""),
        refractory_period=refractory_period,
    )
