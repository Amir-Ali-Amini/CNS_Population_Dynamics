# =====================================================================================================
# =====================================================================================================
# =====================================================================================================
# ====================================  AMIR ALI AMINI ================================================
# ====================================    610399102    ================================================
# =====================================================================================================
# =====================================================================================================

from matplotlib import pyplot as plt
import torch


def memoized_number():
    dic = [0]

    def inside():
        dic[0] += 1
        return f"fig ({dic[0]}):"

    return inside


def plot(
    net,
    title=None,
    ngs=[],
    scaling_factor=3,
    label_font_size=8,
    print_sum_activities=False,
):

    fig_counter = memoized_number()

    n = len(ngs)
    k = n + 1 if n < 3 else n + 2
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(
        k,
        3,
        figsize=(12 * scaling_factor, 3 * k * scaling_factor),
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )
    fig.suptitle(
        title.replace("Steady", "Constant") or "Plot",
        fontsize=(label_font_size + 4) * scaling_factor,
    )

    av_inp_current = [
        torch.sum(ngs[i][9, 0].variables["I_inp"], axis=1) / ngs[i].size
        for i in range(n)
    ]
    av_current = [
        torch.sum(ngs[i][9, 0].variables["I"], axis=1) / ngs[i].size for i in range(n)
    ]
    for i in range(n):
        # Plot 1 - Voltage and Current
        axs[i, 0].plot(ngs[i][9, 0].variables["u"].cpu())
        axs[i, 0].plot(
            av_current[i].cpu() - torch.max(av_current[i]) - 80, linestyle="--"
        )  # Adjust linestyle for current

        axs[i, 0].xaxis.set_tick_params(
            labelsize=(label_font_size - 2) * scaling_factor
        )
        axs[i, 0].yaxis.set_tick_params(
            labelsize=(label_font_size - 2) * scaling_factor
        )
        axs[i, 0].set_xlim(0, ngs[0].network.iteration)
        axs[i, 0].set_xlabel(
            f"{fig_counter()} time {ngs[i].tag}",
            fontsize=label_font_size * scaling_factor,
        )
        # axs[i, 0].legend(['Voltages (top) each line represent one neuron', 'Current (bottom) General overview'], loc='lower right',bbox_to_anchor=(1, 1), fontsize=label_font_size*scaling_factor)
        axs[i, 0].legend(
            [
                "Voltages (top) each line represent one neuron",
                "Current (bottom) General overview",
            ],
            loc="lower right",
            bbox_to_anchor=(1, 1),
            handlelength=0,
            fontsize=label_font_size * scaling_factor,
        )

        # Plot 2 - Current
        axs[i, 1].plot(ngs[i][9, 0].variables["I"].cpu())
        axs[i, 1].plot(
            av_current[i],
            color="#FF00FF",
            label=f"{ngs[i].tag}_final_current",
            linewidth=4,
        )
        axs[i, 1].plot(
            av_current[i],
            color="white",
            label=f"{ngs[i].tag}_final_current",
            linewidth=2,
        )
        axs[i, 1].set_ylabel(f"Current", fontsize=label_font_size * scaling_factor)

        axs[i, 1].xaxis.set_tick_params(
            labelsize=(label_font_size - 2) * scaling_factor
        )
        axs[i, 1].yaxis.set_tick_params(
            labelsize=(label_font_size - 2) * scaling_factor
        )
        axs[i, 1].set_xlim(0, ngs[0].network.iteration)
        axs[i, 1].set_xlabel(
            f"{fig_counter()} time {ngs[i].tag}",
            fontsize=label_font_size * scaling_factor,
        )
        axs[i, 1].legend(
            [
                f"Current each line represent one neuron - 1, {'pink: average' if print_sum_activities else ''}",
            ],
            loc="lower right",
            bbox_to_anchor=(1, 1),
            handlelength=0,
            fontsize=label_font_size * scaling_factor,
        )

        # Plot 3 - activity
        axs[i, 2].plot(ngs[i][9, 0].variables["T"].cpu())
        axs[i, 2].set_ylabel(f"activity", fontsize=label_font_size * scaling_factor)

        axs[i, 2].xaxis.set_tick_params(
            labelsize=(label_font_size - 2) * scaling_factor
        )
        axs[i, 2].yaxis.set_tick_params(
            labelsize=(label_font_size - 2) * scaling_factor
        )
        axs[i, 2].set_xlim(0, ngs[0].network.iteration)
        axs[i, 2].set_xlabel(
            f"{fig_counter()} time {ngs[i].tag}",
            fontsize=label_font_size * scaling_factor,
        )

    # Plot 4 - Spike

    colors = ["blue", "orange", "red", "pink"]
    if n == 3:
        for i in range(n):
            axs[k - 2, i].scatter(
                ngs[i][10, 0].variables["spike"][:, 0].cpu(),
                ngs[i][10, 0].variables["spike"][:, 1].cpu(),
                label=f"{ngs[i].tag}",
            )
            axs[k - 2, i].set_ylabel(
                "spike (neuron number)", fontsize=label_font_size * scaling_factor
            )

            axs[k - 2, i].xaxis.set_tick_params(
                labelsize=(label_font_size - 2) * scaling_factor
            )
            axs[k - 2, i].yaxis.set_tick_params(
                labelsize=(label_font_size - 2) * scaling_factor
            )
            axs[k - 2, i].set_xlim(0, ngs[0].network.iteration)
            axs[k - 2, i].set_xlabel(
                f"{fig_counter()} time {ngs[i].tag}",
                fontsize=label_font_size * scaling_factor,
            )

    for i in range(n):
        axs[k - 1, 0].scatter(
            ngs[i][10, 0].variables["spike"][:, 0].cpu(),
            ngs[i][10, 0].variables["spike"][:, 1].cpu()
            + sum([ng.size for ng in ngs[:i]]),
            color=colors[i],
            label=f"{ngs[i].tag}",
        )
        axs[k - 1, 1].plot(
            av_current[i],
            color=colors[i],
            label=f"{ngs[i].tag}_final_current",
            linewidth=4.0,
        )
        axs[k - 1, 1].plot(
            av_inp_current[i],
            color=colors[-i - 1],
            label=f"{ngs[i].tag}_input_current",
        )
        axs[k - 1, 2].plot(
            ngs[i][9, 0].variables["T"], color=colors[i], label=f"{ngs[i].tag}"
        )

    if print_sum_activities:
        axs[k - 1, 2].plot(
            sum([ngs[i][9, 0].variables["T"] * ngs[i].size for i in range(n)])
            / sum([ngs[i].size for i in range(n)]),
            color="black",
            label=f"overal activity",
            linewidth=4.0,
        )

    axs[k - 1, 0].set_ylabel(
        "spike (neuron number)", fontsize=label_font_size * scaling_factor
    )

    axs[k - 1, 0].xaxis.set_tick_params(
        labelsize=(label_font_size - 2) * scaling_factor
    )
    axs[k - 1, 0].yaxis.set_tick_params(
        labelsize=(label_font_size - 2) * scaling_factor
    )
    axs[k - 1, 0].set_xlim(0, ngs[0].network.iteration)
    axs[k - 1, 0].set_xlabel(
        f"{fig_counter()} time", fontsize=label_font_size * scaling_factor
    )
    axs[k - 1, 0].legend(
        loc="lower right",
        bbox_to_anchor=(1, 1),
        fontsize=label_font_size * scaling_factor,
    )

    axs[k - 1, 1].set_ylabel(
        "average of current", fontsize=label_font_size * scaling_factor
    )

    axs[k - 1, 1].xaxis.set_tick_params(
        labelsize=(label_font_size - 2) * scaling_factor
    )
    axs[k - 1, 1].yaxis.set_tick_params(
        labelsize=(label_font_size - 2) * scaling_factor
    )
    axs[k - 1, 1].set_xlim(0, ngs[0].network.iteration)
    axs[k - 1, 1].set_xlabel(
        f"{fig_counter()} time", fontsize=label_font_size * scaling_factor
    )
    axs[k - 1, 1].legend(
        loc="lower right",
        bbox_to_anchor=(1, 1),
        fontsize=label_font_size * scaling_factor,
    )

    axs[k - 1, 2].set_ylabel("activity", fontsize=label_font_size * scaling_factor)

    axs[k - 1, 2].xaxis.set_tick_params(
        labelsize=(label_font_size - 2) * scaling_factor
    )
    axs[k - 1, 2].yaxis.set_tick_params(
        labelsize=(label_font_size - 2) * scaling_factor
    )
    axs[k - 1, 2].set_xlim(0, ngs[0].network.iteration)
    axs[k - 1, 2].set_xlabel(
        f"{fig_counter()} time", fontsize=label_font_size * scaling_factor
    )
    axs[k - 1, 2].legend(
        loc="lower right",
        bbox_to_anchor=(1, 1),
        fontsize=label_font_size * scaling_factor,
    )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()
