import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_NAME_TO_TAG = {
    "eval_metrics_average_speed_all": "eval_metrics/average_speed",
    "eval_metrics_average_speed_all:dummy": "eval_metrics/average_speed",
    "eval_metrics_co2_emission_all": "eval_metrics/co2_emission",
    "eval_metrics_co2_emission_all:dummy": "eval_metrics/co2_emission",
    "eval_metrics_delay_all": "eval_metrics/delay",
    "eval_metrics_delay_all:dummy": "eval_metrics/delay",
    "eval_metrics_flow_all": "eval_metrics/flow",
    "eval_metrics_flow_all:dummy": "eval_metrics/flow",
    "eval_metrics_lanechange_count_all": "eval_metrics/lanechange_count",
    "eval_metrics_lanechange_count_all:dummy": "eval_metrics/lanechange_count",
    "eval_metrics_total_time_all": "eval_metrics/total_time",
    "eval_metrics_total_time_all:dummy": "eval_metrics/total_time",
    "eval_metrics_travel_time_all": "eval_metrics/travel_time",
    "eval_metrics_travel_time_all:dummy": "eval_metrics/travel_time",
    "eval_metrics_rewards_avg": "eval_metrics/rewards",
    "eval_metrics_rewards_avg:dummy": "eval_metrics/rewards",
    "training_episode_agent_reward_avg": "training_episode_agent_reward",
    "training_episode_agent_reward_50pt": "training_episode_agent_reward",
    "training_episode_agent_reward_90pt": "training_episode_agent_reward",
    "training_episode_reward_avg": "training_episode_reward",
    "training_episode_reward_50pt": "training_episode_reward",
    "training_episode_reward_90pt": "training_episode_reward",
    "training_episode_reward_spi_avg": "training_episode_reward_spi",
    "training_episode_reward_los_avg": "training_episode_reward_los",
    "training_episode_global_reward_avg": "training_episode_global_reward",
    "training_episode_global_reward_50pt": "training_episode_global_reward",
    "training_episode_global_reward_90pt": "training_episode_global_reward",
    "q_values": None,
    "episode_metrics_lanechange_count_all": "episode_metrics/lanechange_count",
    "episode_length": None,
    "eval_metrics_episode_length_episode_length": "eval_metrics/episode_length",
    "eval_metrics_episode_length_episode_length:dummy": "eval_metrics/episode_length:dummy",
}


TRAIN_FIG_SPECS = [
    {
        "demand": "010",
        "metric_key": "training_episode_reward_avg",
        "ylim": (16.0, 17.2),
        "xlim": (2000, 250000),
        "save_name": "010_reward.pdf",
    },
    {
        "demand": "015",
        "metric_key": "training_episode_reward_avg",
        "ylim": (15.8, 16.8),
        "xlim": (2000, 250000),
        "save_name": "015_reward.pdf",
    },
    {
        "demand": "045",
        "metric_key": "training_episode_reward_avg",
        "ylim": (9.6, 10.6),
        "xlim": (2000, 250000),
        "save_name": "045_reward.pdf",
    },
    {
        "demand": "010",
        "metric_key": "training_episode_reward_spi_avg",
        "ylim": (14.6, 16.2),
        "xlim": (2000, 250000),
        "save_name": "010_spi_reward.pdf",
    },
    {
        "demand": "015",
        "metric_key": "training_episode_reward_spi_avg",
        "ylim": (15.0, 16.2),
        "xlim": (2000, 250000),
        "save_name": "015_spi_reward.pdf",
    },
    {
        "demand": "045",
        "metric_key": "training_episode_reward_spi_avg",
        "ylim": (6.8, 7.8),
        "xlim": (2000, 250000),
        "save_name": "045_spi_reward.pdf",
    },
    {
        "demand": "010",
        "metric_key": "training_episode_reward_los_avg",
        "ylim": (17.4, 18.4),
        "xlim": (2000, 250000),
        "save_name": "010_los_reward.pdf",
    },
    {
        "demand": "015",
        "metric_key": "training_episode_reward_los_avg",
        "ylim": (16.6, 17.4),
        "xlim": (2000, 250000),
        "save_name": "015_los_reward.pdf",
    },
    {
        "demand": "045",
        "metric_key": "training_episode_reward_los_avg",
        "ylim": (12.4, 13.4),
        "xlim": (2000, 250000),
        "save_name": "045_los_reward.pdf",
    },
]


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

        steps = []
        values = []
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

            steps = []
            values = []
            for scalar_event in event_acc.Scalars(scalar_tag):
                if scalar_event.step < step_start:
                    continue
                if step_end is not None and scalar_event.step > step_end:
                    break
                steps.append(scalar_event.step)
                values.append(scalar_event.value)
            runs_data[run_name] = {"steps": steps, "values": values}

    return runs_data


def build_data_dict(locations):
    data_dict = {}
    for demand in ("010", "015", "030", "045"):
        for prefix, short_name in (
            ("dummy", "dummy"),
            ("lane_degrade", "ld"),
            ("vehicle_stop", "vs"),
        ):
            key = f"tf_log_dir_{demand}_{prefix}_dqn"
            if key not in locations:
                continue
            data_dict[f"{short_name}_{demand}_data"] = extract_tensorboard_data(
                locations[key], RUN_NAME_TO_TAG
            )
    return data_dict


def moving_average(data, window_size, carry_nonpositive=True):
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return data
    ma_data = np.zeros_like(data)
    if carry_nonpositive:
        filtered = np.zeros_like(data)
        filtered[0] = data[0]
        for i in range(1, len(data)):
            filtered[i] = data[i] if data[i] > 0.0 else filtered[i - 1]
    else:
        filtered = data.copy()

    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        ma_data[i] = np.mean(filtered[start : i + 1])
    return ma_data


def exponential_moving_average(data, alpha, carry_nonpositive=True):
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return data
    ema_data = np.zeros_like(data)
    ema_data[0] = data[0]
    for i in range(1, len(data)):
        current = data[i]
        if carry_nonpositive and current <= 0.0:
            ema_data[i] = ema_data[i - 1]
        else:
            ema_data[i] = alpha * current + (1 - alpha) * ema_data[i - 1]
    return ema_data


def _resolve_metric_and_episode_data(all_data, key_spec):
    if isinstance(key_spec, (tuple, list)):
        if len(key_spec) == 2:
            run_name, metric_key = key_spec
            episode_key = "episode_length"
        elif len(key_spec) == 3:
            run_name, metric_key, episode_key = key_spec
        else:
            raise ValueError(f"Invalid key_spec tuple length: {len(key_spec)}")
        run_data = all_data[run_name]
        metric_data = run_data[metric_key]
        episode_data = run_data.get(episode_key)
        return metric_data, episode_data

    if isinstance(key_spec, str):
        metric_data = all_data[key_spec]
        episode_data = all_data.get("episode_length")
        return metric_data, episode_data

    raise TypeError(f"Unsupported key_spec type: {type(key_spec)}")


def _filter_series_by_episode_length(metric_data, episode_data, min_epi_length=None):
    steps = list(metric_data.get("steps", []))
    values = list(metric_data.get("values", []))
    if min_epi_length is None or not episode_data:
        return steps, values

    epi_len_by_step = {
        step: value
        for step, value in zip(episode_data.get("steps", []), episode_data.get("values", []))
    }
    filtered_steps = []
    filtered_values = []
    for step, value in zip(steps, values):
        epi_len = epi_len_by_step.get(step)
        if epi_len is None:
            continue
        if epi_len >= min_epi_length:
            filtered_steps.append(step)
            filtered_values.append(value)
    return filtered_steps, filtered_values


def plot_data(data, keys, labels, colors, save_path, kargs, window_size=10):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    font = {"family": "Times New Roman", "size": 9}
    font.update(kargs.get("font", {}))
    matplotlib.rc("font", **font)
    matplotlib.rcParams["axes.linewidth"] = 0.5

    fig, ax = plt.subplots(figsize=kargs.get("figsize", (2.5, 2.5)))
    min_epi_length = kargs.get("min_epi_length")
    carry_nonpositive = kargs.get("carry_nonpositive", True)

    for key_spec, label, color in zip(keys, labels, colors):
        metric_data, episode_data = _resolve_metric_and_episode_data(data, key_spec)
        steps, values = _filter_series_by_episode_length(
            metric_data, episode_data, min_epi_length
        )
        if not values:
            continue

        smooth_values = moving_average(
            values, window_size, carry_nonpositive=carry_nonpositive
        )
        if kargs.get("ema_alpha") is not None:
            smooth_values = exponential_moving_average(
                values, kargs["ema_alpha"], carry_nonpositive=carry_nonpositive
            )

        plt.plot(steps, smooth_values, color=color, alpha=1, label=label, linewidth=1)
        alpha = kargs.get("alpha", 0.2)
        if alpha > 0:
            plt.plot(steps, values, color=color, alpha=alpha, linewidth=0.6)

    plt.xlabel(kargs.get("xlabel", "Learning steps"))
    plt.ylabel(kargs.get("ylabel", "Agent Avg Reward"))
    if kargs.get("xlim"):
        plt.xlim(kargs["xlim"])
    if kargs.get("ylim"):
        plt.ylim(kargs["ylim"])
    if kargs.get("y_space"):
        ax.yaxis.set_major_locator(ticker.MultipleLocator(kargs["y_space"]))
    ax.xaxis.set_major_locator(MultipleLocator(50000))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value / 1000)}k"))
    plt.grid(axis="y", linestyle="dotted", linewidth=0.5)
    if kargs.get("legend", True):
        plt.legend(fontsize=kargs.get("legend_fontsize", 8))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_training_reward_plots(data_dict, output_dir, demands=None, min_episode_length=300):
    demand_filter = set(demands) if demands else None
    colors = ["blue", "red", "green"]
    labels = ["Stable Flow", "Lane Degrade", "Vehicle Stop"]

    for spec in TRAIN_FIG_SPECS:
        demand = spec["demand"]
        if demand_filter and demand not in demand_filter:
            continue
        keys = [
            (f"dummy_{demand}_data", spec["metric_key"]),
            (f"ld_{demand}_data", spec["metric_key"]),
            (f"vs_{demand}_data", spec["metric_key"]),
        ]
        kargs = {
            "xlim": spec["xlim"],
            "xlabel": "Learning steps",
            "ylabel": "Agent Avg Reward",
            "alpha": 0.03,
            "figsize": (2.5, 2.5),
            "font": {"family": "Times New Roman", "size": 9},
            "ylim": spec["ylim"],
            "y_space": 0.2,
            "ema_alpha": 0.01 if spec["metric_key"] in ["training_episode_reward_avg", "training_episode_reward_spi_avg",
                                                        "training_episode_reward_los_avg"] else None,
            "min_epi_length": min_episode_length,
        }
        plot_data(
            data_dict,
            keys,
            labels,
            colors,
            output_dir / spec["save_name"],
            kargs,
            window_size=60,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training plots from tf_logs.")
    parser.add_argument(
        "--json",
        default="train_5_v3_tf_logs.json",
        help="JSON file mapping logical names to tf_log directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="train_vis_new",
        help="Directory for generated PDF plots.",
    )
    parser.add_argument(
        "--demands",
        nargs="*",
        default=["010", "015", "045"],
        help="Demand levels to plot, e.g. 010 015 045.",
    )
    parser.add_argument(
        "--min-episode-length",
        type=float,
        default=300,
        help="Filter out training points whose episode length is below this threshold.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    locations = load_locations(base_dir / args.json)
    data_dict = build_data_dict(locations)
    generate_training_reward_plots(
        data_dict,
        base_dir / args.output_dir,
        args.demands,
        min_episode_length=args.min_episode_length,
    )


if __name__ == "__main__":
    main()
