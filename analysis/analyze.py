import pandas as pd
import matplotlib.pyplot as plt


def load_trials():
    return pd.read_csv(
        "results/summary_trials.csv"
    )


def aggregate_trials(df):
    aggregated = (
        df.groupby("concurrency")
        .agg(
            mean_throughput=(
                "throughput",
                "mean",
            ),
            std_throughput=(
                "throughput",
                "std",
            ),
            mean_completion_time=(
                "mean_completion_time",
                "mean",
            ),
            std_completion_time=(
                "mean_completion_time",
                "std",
            ),
            mean_queue_time=(
                "mean_queue_time",
                "mean",
            ),
            mean_execution_time=(
                "mean_execution_time",
                "mean",
            ),
            mean_peak_memory_mb=(
                "mean_peak_memory_mb",
                "mean",
            ),
            max_peak_memory_mb=(
                "max_peak_memory_mb",
                "max",
            ),
        )
        .reset_index()
    )

    return aggregated


def plot_throughput(df):
    plt.figure()

    plt.errorbar(
        df["concurrency"],
        df["mean_throughput"],
        yerr=df["std_throughput"],
        marker="o",
        capsize=4,
    )

    plt.xlabel("Concurrency")
    plt.ylabel(
        "Throughput (tasks/second)"
    )
    plt.title(
        "Throughput vs Concurrency"
    )

    plt.tight_layout()

    plt.savefig(
        "results/"
        "throughput_vs_concurrency.png",
        dpi=200,
    )

    plt.close()


def plot_completion_time(df):
    plt.figure()

    plt.errorbar(
        df["concurrency"],
        df["mean_completion_time"],
        yerr=df["std_completion_time"],
        marker="o",
        capsize=4,
    )

    plt.xlabel("Concurrency")
    plt.ylabel(
        "Mean Completion Time (seconds)"
    )
    plt.title(
        "Completion Time vs Concurrency"
    )

    plt.tight_layout()

    plt.savefig(
        "results/"
        "completion_time_vs_concurrency.png",
        dpi=200,
    )

    plt.close()


def plot_queue_time(df):
    plt.figure()

    plt.plot(
        df["concurrency"],
        df["mean_queue_time"],
        marker="o",
    )

    plt.xlabel("Concurrency")
    plt.ylabel(
        "Mean Queue Time (seconds)"
    )
    plt.title(
        "Queue Time vs Concurrency"
    )

    plt.tight_layout()

    plt.savefig(
        "results/"
        "queue_time_vs_concurrency.png",
        dpi=200,
    )

    plt.close()


def plot_execution_time(df):
    plt.figure()

    plt.plot(
        df["concurrency"],
        df["mean_execution_time"],
        marker="o",
    )

    plt.xlabel("Concurrency")
    plt.ylabel(
        "Mean Execution Time (seconds)"
    )
    plt.title(
        "Execution Time vs Concurrency"
    )

    plt.tight_layout()

    plt.savefig(
        "results/"
        "execution_time_vs_concurrency.png",
        dpi=200,
    )

    plt.close()


def plot_memory(df):
    plt.figure()

    plt.plot(
        df["concurrency"],
        df["mean_peak_memory_mb"],
        marker="o",
        label="Mean peak memory",
    )

    plt.plot(
        df["concurrency"],
        df["max_peak_memory_mb"],
        marker="o",
        label="Max peak memory",
    )

    plt.xlabel("Concurrency")
    plt.ylabel("Memory Usage (MB)")
    plt.title(
        "Memory Usage vs Concurrency"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/"
        "memory_vs_concurrency.png",
        dpi=200,
    )

    plt.close()


def main():
    trials = load_trials()

    aggregated = aggregate_trials(
        trials
    )

    aggregated.to_csv(
        "results/"
        "summary_aggregated.csv",
        index=False,
    )

    print(
        "\nAggregated results:"
    )

    print(
        aggregated.to_string(
            index=False
        )
    )

    plot_throughput(
        aggregated
    )

    plot_completion_time(
        aggregated
    )

    plot_queue_time(
        aggregated
    )

    plot_execution_time(
        aggregated
    )

    plot_memory(
        aggregated
    )

    print(
        "\nSaved aggregated results "
        "to results/"
        "summary_aggregated.csv"
    )

    print(
        "Saved updated plots "
        "with repeated-trial results."
    )


if __name__ == "__main__":
    main()