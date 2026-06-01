import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import Patch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


MODEL_EPISODE_KEY = "eval_metrics_episode_length_episode_length"
BASELINE_EPISODE_KEY = "eval_metrics_episode_length_episode_length:dummy"

BOXPLOT_TAGS = {
    "eval_metrics_average_speed_all": "eval_metrics/average_speed",
    "eval_metrics_average_speed_all:dummy": "eval_metrics/average_speed",
    "eval_metrics_co2_emission_all": "eval_metrics/co2_emission",
    "eval_metrics_co2_emission_all:dummy": "eval_metrics/co2_emission",
    MODEL_EPISODE_KEY: "eval_metrics/episode_length",
    BASELINE_EPISODE_KEY: "eval_metrics/episode_length",
}

ACTION_TAGS = {
    **{f"action_allow_rate_lane_{lane}_{action}": f"action_allow_rate/lane_{lane}"
       for lane in range(5) for action in ("any", "both", "left", "right")}
}


def load_locations(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_tensorboard_data(log_dir, run_name_to_tag, step_start=0, step_end=None):
    event_file = None
    for file_name in os.listdir(log_dir):
        candidate = os.path.join(log_dir, file_name)
        if os.path.isfile(candidate) and file_name.startswith("events"):
            event_file = candidate
            break

    runs_data = {}
    for run_name in os.listdir(log_dir):
        if run_name not in run_name_to_tag:
            continue
        mapped_tag = run_name_to_tag[run_name]
        if mapped_tag is None:
            continue
        run_path = os.path.join(log_dir, run_name)
        if not os.path.isdir(run_path):
            continue
        event_acc = EventAccumulator(run_path)
        event_acc.Reload()
        if mapped_tag not in event_acc.Tags()["scalars"]:
            continue
        steps, values = [], []
        for scalar_event in event_acc.Scalars(mapped_tag):
            if scalar_event.step < step_start:
                continue
            if step_end is not None and scalar_event.step > step_end:
                break
            steps.append(scalar_event.step)
            values.append(scalar_event.value)
        runs_data[run_name] = {"steps": steps, "values": values}

    if event_file is not None:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        scalar_tags = set(event_acc.Tags()["scalars"])
        for run_name, mapped_tag in run_name_to_tag.items():
            if run_name in runs_data:
                continue
            scalar_tag = mapped_tag if mapped_tag is not None else run_name
            if scalar_tag not in scalar_tags:
                continue
            steps, values = [], []
            for scalar_event in event_acc.Scalars(scalar_tag):
                if scalar_event.step < step_start:
                    continue
                if step_end is not None and scalar_event.step > step_end:
                    break
                steps.append(scalar_event.step)
                values.append(scalar_event.value)
            runs_data[run_name] = {"steps": steps, "values": values}

    return runs_data


def _filter_metric_series(run_data, metric_key, min_epi_length=None, episode_key=MODEL_EPISODE_KEY):
    metric_data = run_data[metric_key]
    steps = list(metric_data.get("steps", []))
    values = list(metric_data.get("values", []))
    if min_epi_length is None:
        return steps, values

    episode_data = run_data.get(episode_key)
    if not episode_data:
        return steps, values

    episode_length_by_step = {
        step: value
        for step, value in zip(episode_data.get("steps", []), episode_data.get("values", []))
    }
    filtered_steps, filtered_values = [], []
    for step, value in zip(steps, values):
        episode_length = episode_length_by_step.get(step)
        if episode_length is None:
            continue
        if episode_length >= min_epi_length:
            filtered_steps.append(step)
            filtered_values.append(value)
    return filtered_steps, filtered_values


def _filter_metric_pair(run_data, baseline_key, metric_key, min_epi_length=None,
                        baseline_episode_key=BASELINE_EPISODE_KEY,
                        metric_episode_key=MODEL_EPISODE_KEY):
    baseline_steps, baseline_values = _filter_metric_series(
        run_data, baseline_key, min_epi_length=min_epi_length, episode_key=baseline_episode_key
    )
    metric_steps, metric_values = _filter_metric_series(
        run_data, metric_key, min_epi_length=min_epi_length, episode_key=metric_episode_key
    )
    baseline_by_step = {step: value for step, value in zip(baseline_steps, baseline_values)}
    metric_by_step = {step: value for step, value in zip(metric_steps, metric_values)}
    common_steps = sorted(set(baseline_by_step) & set(metric_by_step))
    return (
        common_steps,
        [baseline_by_step[step] for step in common_steps],
        [metric_by_step[step] for step in common_steps],
    )


def plot_uplift_boxplot(run_data_by_label, output_path, min_epi_length=300, ylim=None):
    matplotlib.rc("font", family="Times New Roman", size=9)
    matplotlib.rcParams["axes.linewidth"] = 0.5
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(run_data_by_label.keys())
    colors = ["blue", "green", "red"]
    box_data = []
    for label in labels:
        run_data = run_data_by_label[label]
        _, baseline_values, metric_values = _filter_metric_pair(
            run_data,
            "eval_metrics_average_speed_all:dummy",
            "eval_metrics_average_speed_all",
            min_epi_length=min_epi_length,
        )
        uplift = [
            (metric_value - baseline_value) / baseline_value * 100.0
            for baseline_value, metric_value in zip(baseline_values, metric_values)
            if baseline_value != 0
        ]
        box_data.append(uplift)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    flierprops = {"marker": "D", "markerfacecolor": "black", "markersize": 2, "markeredgecolor": "black"}
    bplot = ax.boxplot(
        box_data,
        patch_artist=True,
        labels=labels,
        widths=0.6,
        flierprops=flierprops,
        showfliers=True,
    )
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Average speed uplift(%)")
    ax.grid(axis="y", linestyle="dotted", linewidth=1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _read_action_series(log_dir, lane, action):
    subdir = Path(log_dir) / f"action_allow_rate_lane_{lane}_{action}"
    event_acc = EventAccumulator(str(subdir))
    event_acc.Reload()
    series = event_acc.Scalars(f"action_allow_rate/lane_{lane}")
    return [(entry.step, entry.value) for entry in series]


def _nearest_step(target_step, available_steps):
    return min(available_steps, key=lambda step: (abs(step - target_step), step))


def _action_snapshot(log_dir, target_step):
    snapshot = {}
    used_step = None
    for lane in range(5):
        snapshot[lane] = {}
        for action in ("any", "both", "left", "right"):
            series = _read_action_series(log_dir, lane, action)
            steps = [step for step, _ in series]
            nearest = _nearest_step(target_step, steps)
            if used_step is None:
                used_step = nearest
            value = dict(series)[nearest]
            snapshot[lane][action] = value
    return used_step, snapshot


def _apply_edge_lane_special_logic(snapshot, enabled):
    if not enabled:
        return snapshot
    adjusted = {
        lane: values.copy()
        for lane, values in snapshot.items()
    }
    # Lane 0 is the rightmost lane, so only inward (left) changes are feasible.
    if 0 in adjusted:
        adjusted[0]["any"] = adjusted[0]["left"]
        adjusted[0]["both"] = 0.0
        adjusted[0]["right"] = 0.0
    # Lane 4 is the leftmost lane, so only inward (right) changes are feasible.
    if 4 in adjusted:
        adjusted[4]["any"] = adjusted[4]["right"]
        adjusted[4]["both"] = 0.0
        adjusted[4]["left"] = 0.0
    return adjusted


def plot_action_snapshot(log_dir, target_step, output_path, apply_edge_lane_special_logic=False):
    matplotlib.rc("font", family="Times New Roman", size=10)
    matplotlib.rcParams["axes.linewidth"] = 0.5
    output_path.parent.mkdir(parents=True, exist_ok=True)

    used_step, snapshot = _action_snapshot(log_dir, target_step)
    snapshot = _apply_edge_lane_special_logic(snapshot, apply_edge_lane_special_logic)
    colors = ["blue", "green", "red", "skyblue"]
    actions = ["any", "both", "left", "right"]
    labels = ["Left | Right", "Left & Right", "Left only", "Right only"]
    lanes = [0, 1, 2, 3, 4]

    fig, ax = plt.subplots(figsize=(3, 2))
    width = 0.15
    bar_gap = 0.02
    x = np.arange(len(lanes))

    for i, action in enumerate(actions):
        offsets = x - 1.5 * width + i * (width + bar_gap)
        values = [snapshot[lane][action] for lane in lanes]
        ax.bar(offsets, values, width, label=labels[i], color=colors[i])
    ax.set_ylabel("Action Rate")
    ax.set_xticks(x + 0.2 * width)
    ax.set_xticklabels([f"Lane {i}" for i in lanes])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle="dotted", linewidth=1)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return used_step


def parse_args():
    parser = argparse.ArgumentParser(description="Generate eval boxplots and action-rate plots.")
    parser.add_argument(
        "--json",
        default="eval_5_v4_tf_logs.json",
        help="JSON mapping for selected eval/training tf_log directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_vis_new",
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--apply-edge-lane-special-logic",
        action="store_true",
        help="For action plots only, remap lane 0/lane 4 actions to inward-only edge-lane semantics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    locations = load_locations(base_dir / args.json)
    output_dir = base_dir / args.output_dir

    box_010 = {
        "stable flow ": extract_tensorboard_data(locations["tf_log_dir_010_dummy_box"], BOXPLOT_TAGS),
        "lane degrade": extract_tensorboard_data(locations["tf_log_dir_010_ld_box"], BOXPLOT_TAGS),
        "vehicle stop": extract_tensorboard_data(locations["tf_log_dir_010_vs_box"], BOXPLOT_TAGS),
    }
    box_015 = {
        "stable flow ": extract_tensorboard_data(locations["tf_log_dir_015_dummy_box"], BOXPLOT_TAGS),
        "lane degrade": extract_tensorboard_data(locations["tf_log_dir_015_ld_box"], BOXPLOT_TAGS),
        "vehicle stop": extract_tensorboard_data(locations["tf_log_dir_015_vs_box"], BOXPLOT_TAGS),
    }

    plot_uplift_boxplot(box_010, output_dir / "010_uplift_boxplot.pdf", ylim=(-10, 20))
    plot_uplift_boxplot(box_015, output_dir / "015_uplift_boxplot.pdf", ylim=(-10, 20))

    used_010_100 = plot_action_snapshot(
        locations["tf_log_dir_010_dummy_action_train"],
        100000,
        output_dir / "stable_flow_010_action_allow_rate_100k.pdf",
        apply_edge_lane_special_logic=args.apply_edge_lane_special_logic,
    )
    used_010_200 = plot_action_snapshot(
        locations["tf_log_dir_010_dummy_action_train"],
        200000,
        output_dir / "stable_flow_010_action_allow_rate_200k.pdf",
        apply_edge_lane_special_logic=args.apply_edge_lane_special_logic,
    )
    used_015_100 = plot_action_snapshot(
        locations["tf_log_dir_015_dummy_action_train"],
        100000,
        output_dir / "stable_flow_015_action_allow_rate_100k.pdf",
        apply_edge_lane_special_logic=args.apply_edge_lane_special_logic,
    )
    used_015_200 = plot_action_snapshot(
        locations["tf_log_dir_015_dummy_action_train"],
        200000,
        output_dir / "stable_flow_015_action_allow_rate_200k.pdf",
        apply_edge_lane_special_logic=args.apply_edge_lane_special_logic,
    )

    print("Action-plot checkpoint mapping:")
    print(f"  010 target 100k -> used step {used_010_100}")
    print(f"  010 target 200k -> used step {used_010_200}")
    print(f"  015 target 100k -> used step {used_015_100}")
    print(f"  015 target 200k -> used step {used_015_200}")


if __name__ == "__main__":
    main()
