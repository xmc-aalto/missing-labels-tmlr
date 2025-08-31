import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from results import load_data, extract_best_hyperparams, split_by_evaluation, split_by_metric, make_average_and_std
from losses import LossMode, DataMode

sns.set_theme(style="whitegrid", context="paper", palette="colorblind")


def _plot_loss_over_l2(data, loss_type, data_mode="clean-test"):
    data["Setting"] = data["LossVariant"].map({
        "vanilla-clean": "vanilla a)",
        "vanilla-noisy": "vanilla b)",
        "unbiased-noisy": "unbiased c)",
        "bound-noisy": "bound d)",
    }).to_numpy()
    grid = sns.relplot(
        data=data, kind="line",
        x="config/l2_reg", y=f"{data_mode}/{loss_type}", row="config/loss", col="config/normalized",
        hue="Setting",
        facet_kws={'sharey': (loss_type != "loss"), 'sharex': False},
        linewidth=2.0,
        height=2.5,
        ci="sd"
    )

    grid.set(xscale="log")
    grid.set_titles(template="{col_name} - {row_name}")
    grid.set_axis_labels("$L_2$", loss_type)
    return grid


def plot_loss_over_reg(data):
    # plotting the different loss metrics on test data
    for loss_type in ["loss", "P@1", "P@3", "P@5", "R@1", "R@3", "R@5"]:
        grid = _plot_loss_over_l2(data, loss_type)

        if loss_type == "loss":
            for (row_val, col_val), ax in grid.axes_dict.items():
                ax.set_xlim(xmin=1e-7, xmax=1, emit=True)
                if row_val == "sqh" and col_val is False:
                    ax.set_ylim(top=0.15, bottom=0.02, emit=True)
                    ax.set_xlim(xmin=10**(-4), emit=True)
                    ax.set_title("SQH")
                elif row_val == "sqh" and col_val is True:
                    ax.set_ylim(top=0.05, bottom=0.02, emit=True)
                    ax.set_xlim(xmin=10**(-4), emit=True)
                    ax.set_title("SQH Normalized")
                elif row_val == "cce" and col_val is False:
                    ax.set_ylim(top=10, emit=True)
                    ax.set_title("CCE")
                elif row_val == "cce" and col_val is True:
                    ax.set_title("CCE Normalized")
                elif row_val == "bce" and col_val is True:
                    ax.set_ylim(top=0.125, emit=True)
                    ax.set_title("BCE Normalized")
                elif row_val == "bce":
                    ax.set_title("BCE")
        grid.savefig(f"test-{loss_type}-over-reg.png", dpi=300, transparent=False)
        grid.savefig(f"../../plots/over-reg/test-{loss_type}.pgf")
        plt.close(grid.fig)

    grid = _plot_loss_over_l2(data, "loss", "noisy-train")
    grid.savefig(f"train-{loss_type}-over-reg.png", dpi=300, transparent=False)
    grid.savefig(f"../../plots/over-reg/train-{loss_type}.pgf")
    plt.close(grid.fig)


def plot_loss_relations(data, prefix):
    Path(prefix).mkdir(exist_ok=True)

    def col_to_str(col: str):
        replacements = {"clean-test": "CT", "noisy-test": "NT", "clean-train": "CTr", "noisy-train": "NTr"}
        for f, r in replacements.items():
            col = col.replace(f, r)
        return col.replace("/", "-")

    l2_reg = np.array(data["config/l2_reg"].to_list())
    data["l2_log"] = np.log10(l2_reg)

    def plot_rel(x_data: str, y_data: str):
        grid = sns.relplot(
            data=data,
            x=x_data, y=y_data, col="LossDecomposition",
            style="LossVariant", hue="l2_log",
            facet_kws={'sharey': not y_data.endswith("loss"), 'sharex': not x_data.endswith("loss")}
        )
        grid.savefig(f"{prefix}/{col_to_str(x_data)}-{col_to_str(y_data)}.png", dpi=300, transparent=False)
        plt.close(grid.fig)

    plot_rel("clean-test/loss", "clean-train/loss")
    plot_rel("clean-train/loss", "noisy-train/loss")
    plot_rel("clean-test/loss", "noisy-test/loss")
    quantities = ["loss", "P@1", "P@3", "P@5", "R@1", "R@3", "R@5"]
    for first in range(len(quantities)):
        for other in range(len(quantities)):
            if first >= other:
                continue
            plot_rel(f"clean-test/{quantities[first]}", f"clean-test/{quantities[other]}")
            if first != 0 and other != 0:
                x_data = f"clean-test/{quantities[first]}"
                y_data = f"clean-test/{quantities[other]}"
                grid = sns.relplot(
                    data=data,
                    x=x_data, y=y_data, style="LossDecomposition",
                    hue="LossVariant",
                    facet_kws={'sharey': True, 'sharex': True}
                )
                grid.savefig(f"{prefix}/combined-{col_to_str(x_data)}-{col_to_str(y_data)}.png", dpi=300, transparent=False)
                plt.close(grid.fig)


def overfitting_plot(data, other):
    # select only data trained on noisy labels
    data = pd.concat([data, other])
    data = split_by_evaluation(data)

    data = data.loc[(data["config/data"] == "noisy") | data["eval-on"] != "noisy"]
    data["Setting"] = data["LossVariant"].map({
        "vanilla-clean": "vanilla a)",
        "vanilla-noisy": "vanilla b)",
        "unbiased-noisy": "unbiased c)",
        "unbiased-pretrained": "unbiased c')",
        "bound-noisy": "bound d)",
    }).to_numpy()
    data.loc[data["config/pre"] == 10, "Setting"] = "unbiased c')"

    grouped = data.groupby("LossDecomposition")
    for name, group in grouped:
        hue_order = ["vanilla a)", "vanilla b)", "unbiased c)", "bound d)"]
        if name == "cce":
            hue_order = ["vanilla a)", "vanilla b)", "unbiased c)"]
        elif name == "bce-norm":
            hue_order = ["vanilla a)", "vanilla b)", "unbiased c)", "unbiased c')"]
        # TODO I don't know why this is needed!
        group.reset_index(inplace=True)
        grid = sns.relplot(
            data=group, kind="line",
            x="config/l2_reg", y=f"loss",
            hue="Setting", style="eval-on",
            hue_order=hue_order,
            style_order=["clean-test", "noisy-train", "clean-train"],
            facet_kws={'sharey': False, 'sharex': False},
            linewidth=2.0, aspect=1.4, height=3.5, ci="sd"
        )
        grid.set(xscale="log")
        grid.set_titles(template="")
        grid.set_axis_labels("$L_2$", "loss")
        grid.set(xlim=(1e-7, 1))
        if name == "cce-norm":
            grid.set(ylim=(-1.0, None))
        elif name == "bce-norm":
            grid.set(ylim=(-0.05, 0.15), xlim=(1e-6, 1e-0))
        elif name == "cce":
            grid.set(ylim=(None, 11), xlim=(1e-6, 1))
        elif name == "bce":
            grid.set(ylim=(0.0, None), xlim=(1e-7, 1e-0))
        elif name == "sqh" or name == "sqh-norm":
            grid.set(ylim=(-0.2, 0.5))

        grid.savefig(f"overfitting/{name}.png", dpi=300, transparent=False)
        grid.savefig(f"../../plots/overfitting/{name}.pgf", dpi=300, transparent=False)
        plt.close(grid.fig)


def plot_loss_for_paper(data):
    Path("loss").mkdir(exist_ok=True)
    Path("../../plots/loss").mkdir(exist_ok=True)
    grouped = data.groupby("LossDecomposition")
    for name, group in grouped:
        plot = sns.lineplot(
            data=group,
            x="config/l2_reg", y=f"clean-test/loss",
            hue="LossVariant",
            linewidth=1.0, ci="sd"
        )
        plot.set(xscale="log")
        plot.set_xlabel("$L_2$")
        plot.set_ylabel("loss")
        plot.set(xlim=(1e-7, 10))
        if name == "cce-norm":
            plot.set(ylim=(0.0, None))
        elif name == "bce-norm":
            plot.set(ylim=(0, 0.125), xlim=(1e-7, 1e-1))
        elif name == "cce":
            plot.set(ylim=(None, 10), xlim=(1e-5, 1))
        elif name == "bce":
            plot.set(ylim=(0.0, None))
        elif name == "sqh":
            plot.set(ylim=(0.0, 0.3), xlim=(1e-5, 1))
        if name == "sqh-norm":
            pass
            #plot.set(ylim=(0.0, 0.4))

        plt.savefig(f"loss/{name}.png", dpi=300, transparent=False)
        plt.savefig(f"../../plots/loss/{name}.pgf", dpi=300, transparent=False)
        plt.close()


def plot_at_k_for_paper(data, base_type):
    Path("task").mkdir(exist_ok=True)
    Path("../../plots/task").mkdir(exist_ok=True)

    data = split_by_metric(data)
    #data = data[data["LossVariant"] != "vanilla-clean"]

    grouped = data.groupby("LossDecomposition")
    for name, group in grouped:
        plot = sns.lineplot(
            data=group,
            x="config/l2_reg", y=f"clean-test/value",
            style="metric", style_order=[f"{base_type}@1", f"{base_type}@3", f"{base_type}@5"],
            hue="LossVariant",
            linewidth=2.0, ci="sd"
        )
        plot.set(xscale="log")
        plot.set_xlabel("$L_2$")
        plot.set_ylabel(f"{base_type}@k [%]")
        plot.set(xlim=(1e-7, 1))

        plt.savefig(f"task/{name}-{base_type}.png", dpi=300, transparent=False)
        plt.savefig(f"../../plots/task/{name}-{base_type}.pgf", dpi=300, transparent=False)
        plt.close()


def catplot_for_task_losses(data):
    Path("task").mkdir(exist_ok=True)
    Path("../../plots/task").mkdir(exist_ok=True)
    metrics = ["P@1", "P@3", "P@5", "R@1", "R@3", "R@5"]

    data = split_by_metric(data)
    data = data.loc[data["metric"].isin(metrics)]
    metric_as_num = data["metric"].map({"P@1": 0, "P@3": 1, "P@5": 2, "R@1": 3, "R@3": 4, "R@5": 5}).to_numpy()
    loss_as_num = data["config/loss"].map({"bce": -0.25, "sqh": 0.0, "cce": 0.25}).to_numpy()
    metric_as_num = metric_as_num.astype(np.float64) + np.random.random(len(metric_as_num)) * 0.2 - 0.1 + loss_as_num
    data["metric"] = metric_as_num

    # resample data to prevent fixed z-ordering
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    grid = sns.relplot(x="metric", y="clean-test/value", style_order=["bce", "sqh", "cce"],
                       hue="LossVariant",  data=data, col="config/normalized", style="config/loss", kind="scatter",
                       height=8.0 / 3)
    grid.axes[0, 0].set_xticks(range(len(metrics)))
    grid.axes[0, 0].set_xticklabels(metrics)
    grid.despine(left=True)
    plt.savefig(f"task/all.png", dpi=300, transparent=False)
    plt.savefig(f"../../plots/task/all.pgf", dpi=300, transparent=False)


def summary_for_task_losses(data):
    Path("task").mkdir(exist_ok=True)
    Path("../../plots/task").mkdir(exist_ok=True)
    metrics = ["P@1", "P@3", "P@5", "R@1", "R@3", "R@5"]

    data = split_by_metric(data)
    data = data.loc[data["metric"].isin(metrics)]

    grouped = data.groupby(["config/normalized", "metric", "LossVariant", "config/loss"])

    means = grouped["clean-test/value"].mean().reset_index()
    stdev = grouped["clean-test/value"].aggregate(np.std).reset_index()
    print(means)

    metric_as_num = means["metric"].map({"P@1": 0, "P@3": 1, "P@5": 2, "R@1": 3, "R@3": 4, "R@5": 5}).to_numpy()
    loss_as_num = means["config/loss"].map({"bce": 0, "sqh": 1, "cce": 2}).to_numpy()
    metric_as_num = metric_as_num.astype(np.float64) + (loss_as_num - 1) * 0.25
    variant_names = ["Vanilla (a)", "Vanilla (b)", "Unbiased (c)", "Bound (d)"]
    loss_names = ["BCE", "SQH", "CCE"]
    color = means["LossVariant"].map({"vanilla-clean": 0, "vanilla-noisy": 1, "unbiased-noisy": 2, "bound-noisy": 3}).to_numpy()
    marker_styles = ['o', 'x', 's']

    values = means["clean-test/value"].to_numpy()
    errors = stdev["clean-test/value"].to_numpy()
    color_lookup = sns.color_palette()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    for normalized in [False, True]:
        ax = ax2 if normalized else ax1
        for l in range(3):
            for c in range(4):
                label = None
                if not normalized and l == 0:
                    label = variant_names[c]
                elif normalized and c == 0:
                    label = loss_names[l]

                marker = marker_styles[l]
                indices = ((loss_as_num == l) & np.equal(means["config/normalized"].to_numpy(), normalized) & (color == c))
                ax.errorbar(metric_as_num[indices], values[indices], yerr=errors[indices], color=color_lookup[c],
                            marker=marker, linestyle="None", label=label, capsize=2)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.legend()
    plt.savefig(f"../../plots/task/all-custom.pgf", dpi=300, transparent=False)
    plt.savefig(f"task/all-custom.png", dpi=300, transparent=False)
    plt.show()


full_data = load_data(["results-bce.json", "results-sqh.json", "results-cce.json"])
additional_data = load_data(["results-bce-pre.json"])
best_data = extract_best_hyperparams(full_data)
#plot_loss_relations(full_data, "full")
#plot_loss_relations(best_data, "best")
#plot_loss_over_reg(full_data)
overfitting_plot(full_data, additional_data)
#plot_loss_for_paper(full_data)
#plot_at_k_for_paper(full_data, "P")
#plot_at_k_for_paper(full_data, "R")
#catplot_for_task_losses(best_data)
#summary_for_task_losses(best_data)


table_data = make_average_and_std(best_data)
dup_row = table_data[table_data["Setting"] == "U-CCE"].copy()
dup_row["Setting"] = "B-CCE"
dup_row["config/mode"] = LossMode.BOUND
table_data = table_data.append(dup_row)
table_data.sort_values(by=["config/loss", "config/mode", "config/data", "config/normalized"], inplace=True)
table_data.to_csv("data.csv", index=False)
