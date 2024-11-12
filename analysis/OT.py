import glob
import os
import json
import re
import shutil

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean
from scipy.sparse import dok_matrix
from sklearn.manifold import MDS
import random


# Load JSON file
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}, skipping this file.")
    except FileNotFoundError:
        print(f"File not found: {file_path}, skipping this file.")
    except Exception as e:
        print(f"Unexpected error reading file {file_path}: {e}, skipping this file.")
    return None


def extract_frames(data, default_frame=50, length=10):
    half_len = length // 2
    time_steps = []
    frames = []

    if "min_dist_frame" in data:
        frame = int(data["min_dist_frame"])
    else:
        frame = default_frame
    for i in range(frame - half_len, frame + half_len + 1):
        time_step = f"{i}"
        time_steps.append(time_step)

    for time_step in time_steps:
        details = data.get(time_step)
        if not isinstance(details, dict) or not details.get("player"):
            continue

        frame_data = {'npcs': []}
        player_position = details["player"]["transform"]["location"]
        player_velocity = details["player"]["velocity"]
        player_data = {
            'position': np.array([player_position['x'], player_position['y'], player_position['z']]),
            'velocity': np.array([player_velocity['x'], player_velocity['y'], player_velocity['z']]),
            'type': 'player'
        }
        frame_data['npcs'].append(player_data)

        if "NPC" in details and isinstance(details["NPC"], list):
            for npc in details["NPC"]:
                npc_position = npc["transform"]["location"]
                npc_velocity = npc["velocity"]
                npc_data = {
                    'position': np.array([npc_position['x'], npc_position['y'], npc_position['z']]),
                    'velocity': np.array([npc_velocity['x'], npc_velocity['y'], npc_velocity['z']]),
                    'type': 'npc'
                }
                frame_data['npcs'].append(npc_data)

        frames.append(frame_data)
    return frames


# Calculate the state distance between two nodes (vehicles)
def node_distance(node1, node2):
    pos_dist = euclidean(node1['position'], node2['position'])
    vel_dist = euclidean(node1['velocity'], node2['velocity'])
    return pos_dist + vel_dist


# Compute optimal transport distance without a threshold mechanism
def compute_ot_distance_pruned(npcs1, npcs2, k=5):
    num_nodes1, num_nodes2 = len(npcs1), len(npcs2)
    cost_matrix = dok_matrix((num_nodes1, num_nodes2))

    for i in range(num_nodes1):
        distances = [(j, node_distance(npcs1[i], npcs2[j])) for j in range(num_nodes2)]
        distances = sorted(distances, key=lambda x: x[1])[:k]
        for j, dist in distances:
            cost_matrix[i, j] = dist

    cost_matrix_dense = cost_matrix.toarray()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_dense)
    total_cost = cost_matrix_dense[row_ind, col_ind].sum()

    return total_cost


# Graph-DTW for alignment
def graph_dtw(sequence1, sequence2, k=5):
    len1, len2 = len(sequence1), len(sequence2)
    dtw_matrix = np.full((len1 + 1, len2 + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = compute_ot_distance_pruned(sequence1[i - 1]['npcs'], sequence2[j - 1]['npcs'], k=k)
            min_path = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
            dtw_matrix[i, j] = cost + min_path

    return dtw_matrix[len1, len2]


# Sampling function to reduce the number of frames

def calculate_similarity_matrix(base_folder_path):
    file_paths = []
    data_list = []
    files_to_remove = []

    for root, dirs, files in os.walk(base_folder_path):
        dirs.sort()
        files.sort()
        for filename in files:
            if filename.endswith('.json') and 'time_record' in root:
                file_path = os.path.join(root, filename)
                data = load_json(file_path)
                if data is None:
                    continue
                # Extract frames and check against min_frame_count
                extracted_frames = extract_frames(data)
                data_list.append(extracted_frames)
                file_paths.append(file_path)

    num_files = len(file_paths)
    similarity_matrix = np.zeros((num_files, num_files))

    for i in range(num_files):
        for j in range(i, num_files):
            if i == j:
                similarity_matrix[i, j] = 0
            else:
                print(f"Calculating similarity between file {i + 1}/{num_files} and file {j + 1}/{num_files}")
                try:
                    # ?????????????
                    if np.array_equal(data_list[i], data_list[j]):
                        print(f"File {file_paths[i]} and File {file_paths[j]} have identical data frames.")

                        # ????????? 'save' ?????????
                        if 'save' in file_paths[i]:
                            print(f"Deleting file {file_paths[i]} because it contains 'save' in its path.")
                            files_to_remove.append(file_paths[i])  # ???? 'save' ???
                        else:
                            print(f"Deleting file {file_paths[j]} because it contains 'save' in its path.")
                            files_to_remove.append(file_paths[j])  # ???? 'save' ???
                        continue  # ????????????????????

                    # ?? DTW ???
                    score = graph_dtw(data_list[i], data_list[j])
                    if score == np.inf:
                        score = 10000
                    similarity_matrix[i, j] = similarity_matrix[j, i] = score

                    print(f"Similarity score between file {i} and file {j}: {score}")
                except Exception as e:
                    print(f"Error calculating similarity between file {i} and file {j}: {e}")
                    similarity_matrix[i, j] = similarity_matrix[j, i] = np.inf

    large_constant = np.max(similarity_matrix[np.isfinite(similarity_matrix)]) * 1.1
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0, posinf=large_constant)

    np.save('similarity_matrix.npy', similarity_matrix)

    # ??????
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    return similarity_matrix  # Return both similarity matrix and file paths


def find_nearest_points(points, num_points=10):
    # Calculate Euclidean distance of each point from the origin (0,0)
    distances_to_origin = np.linalg.norm(points, axis=1)

    # Get the indices of the nearest points
    nearest_indices = np.argsort(distances_to_origin)[:num_points]

    # Get the nearest points and their distances
    nearest_points = points[nearest_indices]
    nearest_distances = distances_to_origin[nearest_indices]

    return nearest_points, nearest_distances, nearest_indices

# Ensure `find_nearest_points` function is defined or imported here

def plot_similarity_matrix(similarity_matrix):
    file_paths = []
    output_folder = "json_out"
    for root, dirs, files in os.walk('../data'):
        dirs.sort()
        files.sort()
        for filename in files:
            if filename.endswith('.json') and 'time_record' in root:
                file_paths.append(os.path.join(root, filename))

    # MDS transformation
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    points = mds.fit_transform(similarity_matrix)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    cluster_center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - cluster_center, axis=1)
    normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

    plt.figure(figsize=(10, 8))
    tmp_points, save_points, new_points = [], [], []
    tmp_normalized_distances, save_normalized_distances, new_normalized_distances = [], [], []

    tmp_to_new_mapping = {}

    for i, file_path in enumerate(file_paths):
        if 'tmp' in file_path:
            tmp_points.append(i)
            # tmp_normalized_distances.append(normalized_distances[i])
            corresponding_new_file = file_path.replace("/tmp/", "/new/").replace(".json", "_modified.json")
            for j, new_file in enumerate(file_paths):
                if new_file == corresponding_new_file:
                    tmp_to_new_mapping[i] = j
                    break
        elif 'save' in file_path:
            save_points.append(i)
            # save_normalized_distances.append(normalized_distances[i])
        elif 'new' in file_path:
            new_points.append(i)
            # new_normalized_distances.append(normalized_distances[i])

    distance_matrix_A = similarity_matrix[np.ix_(tmp_points, tmp_points)]
    points_A = mds.fit_transform(distance_matrix_A)
    cluster_center_A = np.mean(points_A, axis=0)
    distances_A = np.linalg.norm(points_A - cluster_center_A, axis=1)
    normalized_distances_A = (distances_A - np.min(distances_A)) / (np.max(distances_A) - np.min(distances_A))

    distance_matrix_B = similarity_matrix[np.ix_(new_points, new_points)]
    points_B = mds.fit_transform(distance_matrix_B)
    cluster_center_B = np.mean(points_B, axis=0)
    distances_B = np.linalg.norm(points_B - cluster_center_B, axis=1)
    normalized_distances_B = (distances_B - np.min(distances_B)) / (np.max(distances_B) - np.min(distances_B))

    # tmp_points = np.array(tmp_points, dtype=int)
    # save_points = np.array(save_points, dtype=int)
    # new_points = np.array(new_points, dtype=int)
    #
    # distance_increase = sorted(
    #     [(new_idx, normalized_distances[new_idx] - normalized_distances[tmp_idx])
    #      for tmp_idx, new_idx in tmp_to_new_mapping.items() if
    #      normalized_distances[new_idx] > normalized_distances[tmp_idx]],
    #     key=lambda x: x[1], reverse=True
    # )

    # for tmp_idx, new_idx in tmp_to_new_mapping.items():
    #     plt.arrow(
    #         points[tmp_idx, 0], points[tmp_idx, 1],
    #         points[new_idx, 0] - points[tmp_idx, 0],
    #         points[new_idx, 1] - points[tmp_idx, 1],
    #         edgecolor='gray', length_includes_head=True, head_width=0.02, head_length=0.05
    #     )

    # for idx, _ in distance_increase:
    #     file_name = os.path.basename(file_paths[idx])
    #     plt.text(points[idx, 0], points[idx, 1], file_name, color='red', fontsize=10, ha='right', va='bottom')
    #     plt.scatter(points[idx, 0], points[idx, 1], c='none', edgecolor='blue', s=120, linewidth=1.5)

    # plt.scatter(points[points_A, 0], points[points_A, 1], c=tmp_normalized_distances, cmap='coolwarm', s=100,
    #             edgecolor='red', linewidth=1, label='TMP Files')
    # plt.scatter(points[new_points, 0], points[new_points, 1], c=new_normalized_distances, cmap='coolwarm', s=100,
    #             edgecolor='black', linewidth=1, label='New Files')

    # plt.scatter(points[points_B, 0], points[points_B, 1], c=save_normalized_distances, cmap='coolwarm', s=100,
    #             label='Save Files')
    # plt.scatter(points_A[:, 0], points_A[:, 1], c=normalized_distances_A, cmap='coolwarm', s=100,
    #              label='Original')
    #
    # plt.scatter(points_B[:, 0], points_B[:, 1], c=normalized_distances_B, cmap='coolwarm', s=100,
    #              label='After LLM Mutation')

    distances_from_origin_A = np.linalg.norm(points_A, axis=1)
    distances_from_origin_B = np.linalg.norm(points_B, axis=1)

    # ??????
    for (xA, yA), (xB, yB), dA, dB in zip(points_A, points_B, distances_from_origin_A, distances_from_origin_B):
        # if dB > dA:
        direction_A = np.array([xA, yA]) / dA
        direction_B = np.array([xB, yB]) / dB

        offset_A = direction_A * (dA + 0.5)  # 0.5
        offset_B = direction_B * (dB + 0.5)  # 0.5

        plt.arrow(offset_A[0], offset_A[1], offset_B[0] - offset_A[0], offset_B[1] - offset_A[1],
                  color='black', length_includes_head=True, head_width=100, head_length=100, width=0.1)

    min_len = min(len(points_A), len(points_B))
    points_A = points_A[:min_len]
    points_B = points_B[:min_len]

    plt.scatter(points_A[:, 0], points_A[:, 1], c='grey', s=100, label='Original')
    plt.scatter(points_B[:, 0], points_B[:, 1], c='red', s=100, label='After LLM Mutation')
    plt.yticks(size=30)
    plt.xticks(size=30)

    distances_from_origin_A = np.linalg.norm(points_A, axis=1)
    distances_from_origin_B = np.linalg.norm(points_B, axis=1)
    further_points_count = np.sum(distances_from_origin_B > distances_from_origin_A)
    total_points_count = len(points_A)

    further_points_ratio = further_points_count / total_points_count

    mean_distance_A = np.mean(distances_from_origin_A)
    mean_distance_B = np.mean(distances_from_origin_B)

    print("further_points_ratio", further_points_ratio)
    print("distance_from_origin_A :", mean_distance_A)
    print("distance_from_origin_B :", mean_distance_B)
    print("distance_increase % :", mean_distance_B / mean_distance_A)

    plt.legend(fontsize=18)
    plt.savefig("ot.pdf", dpi=300)
    plt.show()


    # for idx, dist in distance_increase:
    #     print(f"{os.path.basename(file_paths[idx])}: Distance increase = {dist:.4f}")


# Main function
def main(folder_path):
    try:
        similarity_matrix = np.load('similarity_matrix.npy')
    except FileNotFoundError:
        similarity_matrix = calculate_similarity_matrix(folder_path)
    plot_similarity_matrix(similarity_matrix)
    # test()


# Execute main function
if __name__ == "__main__":
    folder_path = '../data'  # Replace with the path to your JSON files folder
    main(folder_path)
