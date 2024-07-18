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


def simulate_one_neuron_group(
    title="",
    model=mdl.LIF(),
    syn_model=syn.RandomConnectivity(j0=30, connection_number=20, tau=1),
    current=cnt.SteadyCurrent(value=6),
    DEVICE=get_device(force_cpu=True)[0],
    dt=1,
    iteration=100,
    ng_size=2,
    print_plots=True,
):

    net = pmt.Network(
        device=DEVICE, dtype=torch.float32, behavior={1: TimeResolution(dt=dt)}
    )

    ng1 = pmt.NeuronGroup(
        size=ng_size,
        net=net,
        behavior={
            2: current,
            4: dnd.InpSyn(),
            5: model,
            6: act.Activity(),
            9: pmt.Recorder(
                variables=["u", "I", "I_inp", "T"], tag="ng1_rec, ng1_recorder"
            ),
            10: pmt.EventRecorder("spike", tag="ng1_evrec"),
        },
    )

    sg1 = pmt.SynapseGroup(
        net=net, src=ng1, dst=ng1, tag="synapse_group", behavior={3: syn_model}
    )

    net.initialize()

    net.simulate_iterations(iteration)

    plot_title = (
        f"## {title or ('one neuron group with '+ sg1[3][0].tag)} synapse group ##\n"
    )
    plot_title += "\n".join(
        [
            f"current: {ng1[2][0]}",
            f"time resolution: {dt}, ng size: {ng_size}, iteration num: {iteration}",
            f"model: {ng1[5][0].tag}",
            f"syn_model: {sg1[3][0].tag}, j0: {sg1[3][0].j0}, C: {sg1[3][0].C}, tau: {sg1[3][0].tau}",
        ]
    )

    print_plots and plot(net, plot_title, [ng1])
    return net


# simulate_one_neuron_group(syn_model=syn.FullyConnected(j0=10))
# simulate_one_neuron_group(syn_model=syn.FullyConnected(j0=10,variation=10))
# simulate_one_neuron_group(syn_model=syn.RandomConnectivity(j0=10,p=20))
# simulate_one_neuron_group(
#     syn_model=syn.RandomConnectivityFix(j0=10, connection_number=2, tau=5),
# )
# simulate_one_neuron_group(
#     syn_model=syn.RandomConnectivityFix(j0=10, connection_number=2, tau=5),
#     current=cnt.UniformCurrent(value=6),
#     ng_size=2,
# )


# simulate_one_neuron_group(model=mdl.LIF(refractory_period=0))
# %%  two neuron groups


def simulate_two_neuron_groups(
    title="2 homo neuron groups",
    model_ex=mdl.LIF(tau_m=20, tag="tau=20-R=1.7,"),
    model_inh=mdl.LIF(tau_m=9, R=25, tag="tau=9-R=25"),
    syn_model_ex_inh=syn.RandomConnectivityFix(
        j0=50, connection_number=5, variation=10, tau=1
    ),
    current=cnt.UniformCurrent(value=5.5, step=0.2),
    DEVICE=get_device(force_cpu=True)[0],
    dt=1,
    iteration=100,
    n_size=100,
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
        tag="Main_neuron_group",
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
        tag="no_input_neuron_group",
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

    sg_ex_inh = pmt.SynapseGroup(
        net=net,
        src=ng1_ex1,
        dst=ng2_inh,
        tag="ex1_inh",
        behavior={3: syn_model_ex_inh},
    )

    net.initialize()

    net.simulate_iterations(iteration)

    plot_title = f"## {title or ('two neuron groups with '+ sg_ex_inh[3][0].tag)+' synapse group'}  ##\n"
    plot_title += "\n".join(
        [
            f"I_ex: {ng1_ex1[2][0]},I_inh: {ng2_inh[2][0]}",
            f"dt: {dt}, ng_size: {ng1_ex1.size}-{ng2_inh.size}, itr_num: {iteration}",
            f"mdl1: {ng1_ex1[5][0].tag}, mdl2: {ng2_inh[5][0].tag}",
            f"syn_mdl: {sg_ex_inh[3][0].tag}, j0: {sg_ex_inh[3][0].j0}, C: {sg_ex_inh[3][0].C}, tau: {sg_ex_inh[3][0].tau}",
        ]
    )

    print_plots and plot(net, plot_title, [ng1_ex1, ng2_inh])
