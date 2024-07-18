# =====================================================================================================
# =====================================================================================================
# =====================================================================================================
# ====================================  AMIR ALI AMINI ================================================
# ====================================    610399102    ================================================
# =====================================================================================================
# =====================================================================================================

from pymonntorch import Behavior
import torch


class RandomWeight(Behavior):
    def initialize(self, sg):
        self.j0 = self.parameter("j0", 1)
        self.tau = self.parameter("tau", 10)
        sg.W = sg.matrix(mode="normal(0.5, 0.3)")
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)
        self.N = sg.src.size
        self.C = self.N
        sg.I = sg.dst.vector()

    def forward(self, sg):
        sg.I += -sg.I / self.tau * sg.network.dt + torch.sum(sg.W[sg.src.spike], axis=0)


class FullyConnected(Behavior):
    def initialize(self, sg):
        self.j0 = self.parameter("j0", 10)
        self.tau = self.parameter("tau", 10)
        self.variation = self.parameter("variation", 0)
        self.N = sg.src.size
        self.C = self.N
        sg.W = sg.matrix(
            mode=f"normal({(self.j0 / self.N)},{((self.j0 / self.N )* self.variation )/ 100})"
        )
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)
        sg.I = sg.dst.vector(0)

    def forward(self, sg):
        sg.I += -sg.I / self.tau * sg.network.dt + torch.sum(sg.W[sg.src.spike], axis=0)


class RandomConnectivity(Behavior):
    def initialize(self, sg):
        self.j0 = self.parameter("j0", 10)
        self.tau = self.parameter("tau", 10)
        self.p = self.parameter("p", 20) / 100
        self.variation = self.parameter("variation", 0) / 100
        self.N = sg.src.size
        self.C = self.N * self.p
        base_val = self.j0 / self.C
        sg.W = sg.matrix(mode=f"normal({base_val},{abs(base_val* self.variation) })")
        prob_matrix = torch.rand_like(sg.W) > self.p
        sg.W[prob_matrix] = 0
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)
        sg.I = sg.dst.vector(0)

    def forward(self, sg):
        sg.I += -sg.I / self.tau * sg.network.dt + torch.sum(sg.W[sg.src.spike], axis=0)


class RandomConnectivityFix(Behavior):
    def initialize(self, sg):
        self.j0 = self.parameter("j0", 10)
        self.tau = self.parameter("tau", 10)
        self.C = self.parameter("connection_number", 20)
        self.variation = self.parameter("variation", 0) / 100
        self.N = sg.src.size
        base_val = self.j0 / self.C
        sg.W = sg.matrix(mode=f"normal({base_val},{abs(base_val* self.variation )})")

        prob_matrix = torch.zeros_like(sg.W)
        for i in range(sg.dst.size):
            prm = torch.randperm(sg.src.size)
            prob_matrix[:, i] = prm

        prob_matrix = prob_matrix >= self.C

        sg.W[prob_matrix] = 0
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)
        sg.I = sg.dst.vector(0)

    def forward(self, sg):
        sg.I += (-sg.I / self.tau) * sg.network.dt + torch.sum(
            sg.W[sg.src.spike], axis=0
        )
