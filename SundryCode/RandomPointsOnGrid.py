import numpy as np
from scipy.spatial import distance

# Grid size
grid_size = 10

# Number of random points
num_random_points = 36

# Number of squares (10x10 grid)
num_squares = grid_size * grid_size

# Create all square centers on the grid (10x10 squares)
square_centers = np.array([(i + 0.5, j + 0.5) for i in range(grid_size) for j in range(grid_size)])

overall_average_min_distance = 0

loopNo=10000

for _ in range(loopNo):
            
    # Generate 16 random points within the grid
    random_points = np.random.uniform(0, grid_size, (num_random_points, 2))

    # Calculate distances from each square center to each random point
    distances = distance.cdist(square_centers, random_points)

    # For each square center, find the minimum distance to the random points
    min_distances = np.min(distances, axis=1)

    # Calculate the average of these minimum distances
    average_min_distance = np.mean(min_distances)

    overall_average_min_distance += average_min_distance

    print(f"Average distance between square centers and the closest random point: {average_min_distance:.2f}")

print(f"Overall average distance: {overall_average_min_distance / loopNo:.3f}")