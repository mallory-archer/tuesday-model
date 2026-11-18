"""Assign students to groups by dealing clusters round-robin."""

import csv
from collections import defaultdict
from pathlib import Path

from cluster_model import predict_cluster

SKILL_COLUMNS = (
    "skill_programming",
    "skill_data_stats",
    "skill_frontend_design",
    "skill_business_pitch",
)
DATA_PATH = Path(__file__).resolve().parent / "class_dataset.csv"


def assign_groups(students, n_groups=8):
    """Label each student with a cluster and a group number.

    Clusters are ordered from most students to fewest, and ties go to the
    smaller cluster number. Within a cluster, students are sorted by id.
    Those lists are concatenated and dealt to groups 1..n_groups, and the
    deal continues across clusters instead of restarting.
    """
    by_cluster = defaultdict(list)
    for student in students:
        skills = tuple(student[column] for column in SKILL_COLUMNS)
        cluster = predict_cluster(skills)
        by_cluster[cluster].append({"id": student["id"], "cluster": cluster})

    cluster_order = sorted(by_cluster, key=lambda cluster: (-len(by_cluster[cluster]), cluster))
    deal = []
    for cluster in cluster_order:
        members = sorted(by_cluster[cluster], key=lambda student: student["id"])
        deal.extend(members)

    assigned_by_id = {}
    for index, student in enumerate(deal):
        assigned_by_id[student["id"]] = {
            "id": student["id"],
            "cluster": student["cluster"],
            "group": (index % n_groups) + 1,
        }
    return [assigned_by_id[student["id"]] for student in students]


def load_students(path):
    students = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            student = {"id": row["netid"]}
            for column in SKILL_COLUMNS:
                student[column] = int(row[column])
            students.append(student)
    return students


def main():
    n_groups = 8
    assignments = assign_groups(load_students(DATA_PATH), n_groups=n_groups)
    clusters = sorted({row["cluster"] for row in assignments})
    missing = 0
    for group in range(1, n_groups + 1):
        members = [row for row in assignments if row["group"] == group]
        counts = {cluster: 0 for cluster in clusters}
        for row in members:
            counts[row["cluster"]] += 1
        if any(count == 0 for count in counts.values()):
            missing += 1
        breakdown = " ".join(f"cluster_{cluster}={counts[cluster]}" for cluster in clusters)
        print(f"group={group} n={len(members)} {breakdown}")
    print(f"groups_missing_a_cluster={missing}")


if __name__ == "__main__":
    main()
