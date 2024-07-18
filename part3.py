# =====================================================================================================
# =====================================================================================================
# =====================================================================================================
# ====================================  AMIR ALI AMINI ================================================
# ====================================    610399102    ================================================
# =====================================================================================================
# =====================================================================================================

# %%
import pymonntorch as pmt
import torch
from matplotlib import pyplot as plt

import model as mdl
import syn
import dandrit as dnd
import current as cnt
import activity as act
from getDevice import get_device
from dt import TimeResolution
from plot import plot


def LIF(**arg):
    tag = []
    for key in arg.keys():
        tag += [f"{key}={arg[key]}"]
    return mdl.LIF(**arg, tag="|".join(tag))


def ELIF(**arg):
    tag = []
    for key in arg.keys():
        tag += [f"{key}={arg[key]}"]
    return mdl.ELIF(**arg, tag="|".join(tag))


def AELIF(**arg):
    tag = []
    for key in arg.keys():
        tag += [f"{key}={arg[key]}"]
    return mdl.AELIF(**arg, tag="|".join(tag))


def simulate_two_neuron_group(
    title="",
    model_ex=LIF(tau_m=10),
    model_inh=LIF(tau_m=3, R=10),
    syn_model_ex_ex=syn.RandomConnectivityFix(
        j0=50, connection_number=100, tau=1, variation=30
    ),
    syn_model_ex_inh=syn.RandomConnectivityFix(
        j0=0, connection_number=5, variation=10, tau=10
    ),
    syn_model_inh_ex=syn.RandomConnectivityFix(j0=0, connection_number=30, tau=6),
    current=cnt.UniformSingleCurrent(value=6.3, initial_current=90),
    DEVICE=get_device(force_cpu=True)[0],
    dt=1,
    iteration=100,
    n_size=1000,
    print_plots=True,
    ex_size=0,
    in_size=0,
):
    net = pmt.Network(
        device=DEVICE, dtype=torch.float32, behavior={1: TimeResolution(dt=dt)}
    )

    ng1_ex1 = pmt.NeuronGroup(
        size=ex_size or int(n_size * 0.8),
        net=net,
        tag="ng1_ex",
        behavior={
            2: current,
            4: dnd.InpSyn(),
            5: model_ex,
            6: act.Activity(),
            9: pmt.Recorder(
                variables=["u", "I", "I_inp", "T"], tag="ng1_ex1_rec, ng1_ex1_recorder"
            ),
            10: pmt.EventRecorder("spike", tag="ng1_ex1_evrec"),
        },
    )

    ng2_inh = pmt.NeuronGroup(
        size=in_size or int(n_size * 0.2),
        net=net,
        tag="ng2_inh",
        behavior={
            2: cnt.SteadyCurrent(value=0),
            4: dnd.InpSyn(),
            5: model_inh,
            6: act.Activity(),
            9: pmt.Recorder(
                variables=["u", "I", "I_inp", "T"], tag="ng2_inh_rec, ng2_inh_recorder"
            ),
            10: pmt.EventRecorder("spike", tag="ng2_inh_evrec"),
        },
    )

    sg_ex_ex = pmt.SynapseGroup(
        net=net,
        src=ng1_ex1,
        dst=ng1_ex1,
        tag="ex1_ex1",
        behavior={3: syn_model_ex_ex},
    )
    sg_ex_inh = pmt.SynapseGroup(
        net=net,
        src=ng1_ex1,
        dst=ng2_inh,
        tag="ex1_inh",
        behavior={3: syn_model_ex_inh},
    )
    sg_inh_ex = pmt.SynapseGroup(
        net=net,
        src=ng2_inh,
        dst=ng1_ex1,
        tag="inh_ex1",
        behavior={3: syn_model_inh_ex},
    )

    net.initialize()

    net.simulate_iterations(iteration)

    plot_title = f"## {title or ('two neuron groups with '+ sg_ex_inh[3][0].tag)+' synapse group'}  ##\n"
    plot_title += "\n".join(
        [
            f"I_ex: {ng1_ex1[2][0]},I_inh: {ng2_inh[2][0]}",
            f"dt: {dt}, ng_size: {ng1_ex1.size}-{ng2_inh.size}, itr_num: {iteration}",
            f"mdl1: {ng1_ex1[5][0].tag}, mdl2: {ng2_inh[5][0].tag}",
            f"syn_ex_ex: {sg_ex_ex[3][0].tag}(j0: {sg_ex_ex[3][0].j0}, C: {sg_ex_ex[3][0].C}, tau: {sg_ex_ex[3][0].tau})",
            f"syn_ex_inh: {sg_ex_inh[3][0].tag}(j0: {sg_ex_inh[3][0].j0}, C: {sg_ex_inh[3][0].C}, tau: {sg_ex_inh[3][0].tau})",
            f"syn_inh_ex: {sg_inh_ex[3][0].tag}(j0: {sg_inh_ex[3][0].j0}, C: {sg_inh_ex[3][0].C}, tau: {sg_inh_ex[3][0].tau})",
        ]
    )

    print_plots and plot(
        net, plot_title, [ng1_ex1, ng2_inh], print_sum_activities=True, scaling_factor=4
    )


# simulate_two_neuron_group(
#     iteration=100,
#     # syn_model_ex_ex=syn.RandomConnectivityFix(
#     #     j0=28, connection_number=2, tau=1, variation=0
#     # ),
#     # syn_model_ex_ex=syn.RandomConnectivityFix(
#     #     j0=28, connection_number=10, tau=1, variation=0
#     # ),
#     # syn_model_ex_inh=syn.RandomConnectivityFix(
#     #     j0=20, connection_number=40, tau=1, variation=30
#     # ),
#     # syn_model_inh_ex=syn.RandomConnectivityFix(
#     #     j0=-100, connection_number=14, tau=1, variation=30
#     # ),
#     syn_model_ex_ex=syn.RandomConnectivityFix(
#         j0=1, connection_number=10, tau=1, variation=0
#     ),
#     syn_model_ex_inh=syn.RandomConnectivityFix(
#         j0=10, connection_number=10, tau=1, variation=30
#     ),
#     syn_model_inh_ex=syn.RandomConnectivityFix(
#         j0=-4, connection_number=14, tau=1, variation=30
#     ),
# )
# simulate_one_neuron_group(
#     iteration=200,
#     syn_model_ex_ex=syn.RandomConnectivityFix(
#         j0=50, connection_number=100, tau=1, variation=30
#     ),
# )

# %%


# print(
#     LIF(
#         threshold=-55,
#         u_rest=-65,
#         u_reset=-70,
#         R=1.7,
#         tau_m=10,
#         refractory_period=0,
#     ).tag
# )
# print(
#     LIF(
#         threshold=-40,
#         R=20,
#         tau_m=2,
#     ).tag
# )
# print(
#     AELIF(
#         threshold=+30,
#         u_rest=-65,
#         u_reset=-70,
#         R=1.7,
#         tau_m=10,
#         u_rh=-45,
#         delta_t=1,
#         a=0.4,
#         b=1,
#     ).tag
# )
