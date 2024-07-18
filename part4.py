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
import copy


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


def simulate_three_neuron_group(
    title="",
    model_ex=LIF(tau_m=10),
    model_inh=LIF(tau_m=3, R=10),
    syn_model_ex_ex=syn.RandomConnectivityFix(
        j0=1, connection_number=10, tau=1, variation=40
    ),
    syn_model_ex_inh=syn.RandomConnectivityFix(
        j0=10, connection_number=10, tau=1, variation=30
    ),
    syn_model_inh_ex=syn.RandomConnectivityFix(
        j0=-4, connection_number=14, tau=1, variation=30
    ),
    current=None,
    current_ex1=cnt.UniformSingleCurrent(value=6.3, initial_current=90),
    current_ex2=cnt.UniformSingleCurrent(value=6.3, initial_current=90),
    current_inh=cnt.SteadyCurrent(value=0),
    DEVICE=get_device(force_cpu=True)[0],
    dt=1,
    iteration=100,
    n_size=1000,
    print_plots=True,
    ex_size=0,
    in_size=0,
    diff_current=0.05,
):
    net = pmt.Network(
        device=DEVICE, dtype=torch.float32, behavior={1: TimeResolution(dt=dt)}
    )
    if current is not None:
        current_ex1 = copy.deepcopy(current)
        current_ex1.init_kwargs["value"] *= 1 + diff_current

        current_ex2 = copy.deepcopy(current)
        current_ex2.init_kwargs["value"] *= 1 - diff_current

    ng1_ex1 = pmt.NeuronGroup(
        size=ex_size or int(n_size * 0.8),
        net=net,
        tag="ng1_ex1",
        behavior={
            2: current_ex1,
            4: dnd.InpSyn(),
            5: model_ex,
            6: act.Activity(),
            9: pmt.Recorder(
                variables=["u", "I", "I_inp", "T"], tag="ng1_ex1_rec, ng1_ex1_recorder"
            ),
            10: pmt.EventRecorder("spike", tag="ng1_ex1_evrec"),
        },
    )

    ng2_ex2 = pmt.NeuronGroup(
        size=ex_size or int(n_size * 0.8),
        net=net,
        tag="ng2_ex2",
        behavior={
            2: current_ex2,
            4: dnd.InpSyn(),
            5: model_ex,
            6: act.Activity(),
            9: pmt.Recorder(
                variables=["u", "I", "I_inp", "T"], tag="ng3_ex2_rec, ng3_ex2_recorder"
            ),
            10: pmt.EventRecorder("spike", tag="ng3_ex2_evrec"),
        },
    )

    ng3_inh = pmt.NeuronGroup(
        size=in_size or int(n_size * 0.2),
        net=net,
        tag="ng3_inh",
        behavior={
            2: current_inh,
            4: dnd.InpSyn(),
            5: model_inh,
            6: act.Activity(),
            9: pmt.Recorder(
                variables=["u", "I", "I_inp", "T"], tag="ng3_inh_rec, ng3_inh_recorder"
            ),
            10: pmt.EventRecorder("spike", tag="ng3_inh_evrec"),
        },
    )
    # ex1
    sg_ex1_ex1 = pmt.SynapseGroup(
        net=net,
        src=ng1_ex1,
        dst=ng1_ex1,
        tag="ex1_ex1",
        behavior={3: syn_model_ex_ex},
    )
    sg_ex1_inh = pmt.SynapseGroup(
        net=net,
        src=ng1_ex1,
        dst=ng3_inh,
        tag="ex1_inh",
        behavior={3: syn_model_ex_inh},
    )
    sg_inh_ex1 = pmt.SynapseGroup(
        net=net,
        src=ng3_inh,
        dst=ng1_ex1,
        tag="inh_ex1",
        behavior={3: syn_model_inh_ex},
    )
    # ex2
    sg_ex2_ex2 = pmt.SynapseGroup(
        net=net,
        src=ng2_ex2,
        dst=ng2_ex2,
        tag="ex2_ex2",
        behavior={3: syn_model_ex_ex},
    )
    sg_ex2_inh = pmt.SynapseGroup(
        net=net,
        src=ng2_ex2,
        dst=ng3_inh,
        tag="ex2_inh",
        behavior={3: syn_model_ex_inh},
    )
    sg_inh_ex2 = pmt.SynapseGroup(
        net=net,
        src=ng3_inh,
        dst=ng2_ex2,
        tag="inh_ex2",
        behavior={3: syn_model_inh_ex},
    )

    net.initialize()

    net.simulate_iterations(iteration)

    plot_title = f"## {title or ('two neuron groups with '+ sg_ex2_inh[3][0].tag)+' synapse group'}  ##\n"
    plot_title += "\n".join(
        [
            f"I_ex1: {ng1_ex1[2][0]}",
            f"I_ex2: {ng2_ex2[2][0]}",
            f"I_inh: {ng3_inh[2][0]}",
            f"dt: {dt}, ng_size: {ng1_ex1.size}-{ng3_inh.size}, itr_num: {iteration}",
            f"mdl_ex: {ng1_ex1[5][0].tag}, mdl_inh: {ng3_inh[5][0].tag}",
            f"syn_ex_ex: {sg_ex2_ex2[3][0].tag}(j0: {sg_ex2_ex2[3][0].j0}, C: {sg_ex2_ex2[3][0].C}, tau: {sg_ex2_ex2[3][0].tau})",
            f"syn_ex_inh: {sg_ex2_inh[3][0].tag}(j0: {sg_ex2_inh[3][0].j0}, C: {sg_ex2_inh[3][0].C}, tau: {sg_ex2_inh[3][0].tau})",
            f"syn_inh_ex: {sg_inh_ex2[3][0].tag}(j0: {sg_inh_ex2[3][0].j0}, C: {sg_inh_ex2[3][0].C}, tau: {sg_inh_ex2[3][0].tau})",
            "",
        ]
    )

    print_plots and plot(
        net,
        plot_title.replace("Steady", "Constant"),
        [ng1_ex1, ng2_ex2, ng3_inh],
        print_sum_activities=False,
        scaling_factor=4,
    )


# simulate_three_neuron_group(
#     current=cnt.UniformSingleCurrent(
#         value=6.3, step=0.1, initial_current=100, noise_range=0.05
#     ),
#     syn_model_ex_ex=syn.RandomConnectivity(j0=6, p=6, tau=1, variation=40),
#     syn_model_ex_inh=syn.RandomConnectivity(j0=8, p=1, tau=1, variation=30),
#     syn_model_inh_ex=syn.RandomConnectivity(j0=-5, p=40, tau=1, variation=30),
#     model_inh=LIF(tau_m=3, R=30),
#     iteration=100,
#     diff_current=0.005,
#     dt=0.5,
# )

# simulate_three_neuron_group(
#     current_ex1=cnt.UniformSingleCurrent(
#         value=6.4, step=0.1, initial_current=100, noise_range=0.05
#     ),
#     current_ex2=cnt.UniformSingleCurrent(
#         value=6.35, initial_current=100, step=0.1, noise_range=0.05
#     ),
#     syn_model_ex_ex=syn.RandomConnectivityFix(
#         j0=1, connection_number=10, tau=1, variation=40
#     ),
#     syn_model_ex_inh=syn.RandomConnectivityFix(
#         j0=7, connection_number=10, tau=1, variation=30
#     ),
#     syn_model_inh_ex=syn.RandomConnectivityFix(
#         j0=-4, connection_number=14, tau=1, variation=30
#     ),
# )

# %%
