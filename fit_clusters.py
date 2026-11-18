"""Fit k-means on class skill ratings, save an elbow plot, and label k=3 clusters."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

SKILL_COLUMNS = [
    "skill_programming",
    "skill_data_stats",
    "skill_frontend_design",
    "skill_business_pitch",
]

DATA_PATH = Path(__file__).resolve().parent / "class_dataset.csv"
PLOT_PATH = Path(__file__).resolve().parent / "elbow.png"
K_VALUES = range(1, 7)
CHOSEN_K = 3


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[SKILL_COLUMNS]
    n_students = len(X)
    print(f"n_students={n_students}")

    inertias = []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, n_init=10, random_state=0)
        model.fit(X)
        inertias.append(model.inertia_)
        print(f"k={k} inertia={model.inertia_}")

    fig, ax = plt.subplots()
    ax.plot(list(K_VALUES), inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("inertia")
    ax.set_title("K-means elbow")
    ax.set_xticks(list(K_VALUES))
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    chosen = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=0)
    df["cluster"] = chosen.fit_predict(X)
    df.to_csv(DATA_PATH, index=False)

    for label in range(CHOSEN_K):
        members = df[df["cluster"] == label]
        centroid = members[SKILL_COLUMNS].mean()
        averages = ", ".join(f"{value:.2f}" for value in centroid)
        print(f"cluster={label} n={len(members)} centroid={averages}")


if __name__ == "__main__":
    main()
