import numpy as np


def cluster(points, radius,eps=0.1):
    centroids = []
    while len(points) != 0:
        current_point = get_random_point(points)
        neighbors = get_neighbors(current_point, radius, points)
        centroid = get_centroid(neighbors)
        while np.linalg.norm(current_point - centroid) > eps:
            current_point = centroid
            neighbors = get_neighbors(current_point, radius, points)
            centroid = get_centroid(neighbors)
        points = remove_points(neighbors, points)
        centroids.append(current_point)
    return centroids


def get_neighbors(p, radius, points):
    neighbors = []
    for point in points:
        if np.linalg.norm(p - point) < radius:
            neighbors.append(point)
    return np.array(neighbors)


def get_centroid(points):
    return np.array([np.mean(points[:, 0]), np.mean(points[:, 1])])


def get_random_point(points):
    random_index = np.random.choice(len(points), 1)[0]
    return points[random_index]


def remove_points(subset, points):
    points = []
    for p in points:
        if p not in subset:
            points.append(p)
    return points
    

    
# # Создаем искусственные данные для демонстрации
np.random.seed(0)
data = np.concatenate([np.random.randn(50, 2) + [2, 2],
                        np.random.randn(50, 2) + [-2, -2],
                        np.random.randn(50, 2) + [2, -2]])
r = 6                       
c = cluster(data,r)

for i in c:
    print(f"Clusters centers: {i}")
