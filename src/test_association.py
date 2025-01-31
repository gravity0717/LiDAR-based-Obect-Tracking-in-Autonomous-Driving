import numpy as np
import matplotlib.pyplot as plt
from src.association import Association

def generate_mock_data(frame_num, cluster_count, noise_level=0.5):
    """
    Mock data generator: cluster trajectories with noise
    :param frame_num: Number of frames to simulate
    :param cluster_count: Number of clusters per frame
    :param noise_level: Magnitude of Gaussian noise
    :return: List of cluster centroids for each frame
    """
    clusters = []
    base_positions = np.random.rand(cluster_count, 2) * 10  # Base positions
    velocity = np.random.rand(cluster_count, 2) * 0.5  # Constant velocity
    for _ in range(frame_num):
        base_positions += velocity
        noisy_positions = base_positions + np.random.randn(cluster_count, 2) * noise_level
        clusters.append(noisy_positions)
    return clusters

def visualize_tracking(data, association, save_path=None):
    """
    Visualizes the tracking results using matplotlib.
    :param data: List of input cluster centroids for each frame
    :param association: Association object containing tracking results
    :param save_path: File path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # Assign unique colors to each track ID
    track_colors = {}

    for frame_idx, centroids in enumerate(data):
        plt.clf()  # Clear the plot for the current frame

        # Update clusters in the association
        association.update_clusters(centroids)

        # Plot input centroids for the current frame
        plt.scatter(centroids[:, 0], centroids[:, 1], c='gray', label='Clusters', s=50, alpha=0.5)

        # Plot each tracklet
        for track_id, track_data in association.tracklets.items():
            tracker_position = track_data['tracker'].ekf.x[:2]

            # Assign a consistent color to each track ID
            if track_id not in track_colors:
                track_colors[track_id] = np.random.rand(3)  # Random color
            color = track_colors[track_id]

            # Plot the tracker position
            plt.scatter(tracker_position[0], tracker_position[1], c=[color], label=f'Track {track_id}')
            plt.text(tracker_position[0], tracker_position[1], f'{track_id}', color=color)

        plt.title(f"Frame {frame_idx + 1}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.xlim(0, 15)
        plt.ylim(0, 15)
        plt.legend(loc='upper right')

        # Pause to create an animation effect
        plt.pause(0.5)

    if save_path:
        plt.savefig(save_path)

    plt.show()

def test_association_with_visualization():
    # Parameters
    frame_count = 10
    cluster_count = 5
    noise_level = 0.2
    max_age = 2

    # Generate mock data
    data = generate_mock_data(frame_count, cluster_count, noise_level)
    # Initialize Association
    association = Association()
    association.max_age = max_age

    # Visualize tracking
    visualize_tracking(data, association)

if __name__ == "__main__":
    test_association_with_visualization()
