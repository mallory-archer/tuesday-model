"""Nearest-centroid labels for the k=3 skill fit.

Centroids are the k-means centers for skill_programming, skill_data_stats,
skill_frontend_design, and skill_business_pitch, in that order.
"""

CENTROIDS = (
    (1.7857142857142854, 2.5, 2.428571428571429, 4.285714285714286),
    (3.4166666666666665, 3.1666666666666665, 2.166666666666667, 2.4166666666666665),
    (3.875, 4.0, 4.25, 4.625),
)


def predict_cluster(skills):
    """Return the label of the centroid nearest to the four skill ratings."""
    best_label = 0
    best_distance = None
    for label, centroid in enumerate(CENTROIDS):
        distance = 0.0
        for value, center in zip(skills, centroid):
            diff = value - center
            distance += diff * diff
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_label = label
    return best_label
