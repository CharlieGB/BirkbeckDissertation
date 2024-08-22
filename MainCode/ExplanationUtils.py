import numpy as np

def ExtractExplanation(test_results, exp_thresh, bShuffle_Exp, excludeDuplicatesWithDifferentAction=False, excludeWhereDQNWithin=None):
    weights = np.array(test_results['weights'])
    actions = np.array(test_results['actions'])
    memories = np.array(test_results['memories'])
    values = np.array(test_results['values'])
    observations = np.array(test_results['observations'])
    dqn = np.array(test_results['DQN_q'])


    thresholded_actions = actions[weights > exp_thresh]
    thresholded_memories = memories[weights > exp_thresh, :]
    thresholded_values = values[weights > exp_thresh, :]
    thresholded_weights = weights[weights > exp_thresh]
    thresholded_observations = observations[weights > exp_thresh]
    thresholded_dqn = dqn[weights > exp_thresh]


    memories_dict = {}
    for i in range(thresholded_weights.shape[0]):
        if excludeWhereDQNWithin is not None:
            qValues = thresholded_weights[i] * thresholded_values[i] + (1 - thresholded_weights[i]) * thresholded_dqn[i]
            maxQ = np.max(qValues)
            action = np.argmax(thresholded_values[i])
            threshold = abs(excludeWhereDQNWithin * maxQ)
            if np.argmax(thresholded_dqn[i]) == action and abs(np.max(thresholded_dqn[i]) - maxQ) < threshold:
                continue

        m = tuple(thresholded_memories[i, :])

        if (m in memories_dict):
            if (not excludeDuplicatesWithDifferentAction and memories_dict[m]['action'] != thresholded_actions[i]):
                if memories_dict[m]['weight'] < thresholded_weights[i]:
                    # Replace the existing memory with the new one
                    # Create a new memory with the existing memory's observation
                    m2 = tuple(memories_dict[m]['observation'])
                    memories_dict[m2] = {}
                    memories_dict[m2]['weight'] = memories_dict[m]['weight']
                    memories_dict[m2]['ind'] = memories_dict[m]['ind']
                    memories_dict[m2]['action'] = memories_dict[m]['action']
                    memories_dict[m2]['observation'] = memories_dict[m]['observation']
                    thresholded_memories[memories_dict[m]['ind'], :] = memories_dict[m]['observation']
                    memories_dict[m]['weight'] = thresholded_weights[i]
                    memories_dict[m]['ind'] = i
                    memories_dict[m]['action'] = thresholded_actions[i]
                else:
                    m2 = tuple(thresholded_observations[i, :])
                    memories_dict[m2] = {}
                    memories_dict[m2]['weight'] = thresholded_weights[i]
                    memories_dict[m2]['ind'] = i
                    thresholded_memories[i, :] = thresholded_observations[i, :]
                    memories_dict[m2]['action'] = thresholded_actions[i]
                    memories_dict[m2]['observation'] = thresholded_observations[i, :]
            else:
                if memories_dict[m]['weight'] < thresholded_weights[i]:
                    memories_dict[m]['weight'] = thresholded_weights[i]
                    memories_dict[m]['ind'] = i
        else:
            memories_dict[m] = {}
            memories_dict[m]['weight'] = thresholded_weights[i]
            memories_dict[m]['ind'] = i
            memories_dict[m]['action'] = thresholded_actions[i]
            memories_dict[m]['observation'] = thresholded_observations[i]
    final_inds = []
    for key, item in memories_dict.items():
        final_inds.append(item['ind'])

    explanation_length = len(final_inds)

    if bShuffle_Exp:
        final_inds = np.random.choice(weights.shape[0], explanation_length, replace=False).tolist()
        exp_weights = weights[final_inds]
        exp_actions = actions[final_inds]
        exp_memories = memories[final_inds, :]
        exp_values = values[final_inds, :]
        exp_observations = observations[final_inds, :]
    else:
        exp_weights = thresholded_weights[final_inds]
        exp_actions = thresholded_actions[final_inds]
        exp_memories = thresholded_memories[final_inds, :]
        exp_values = thresholded_values[final_inds, :]
        exp_observations = thresholded_observations[final_inds, :]

    return exp_actions, exp_memories, exp_values, exp_weights, exp_observations

# Experimental, not used
def ExtractExplanation2(test_results, exp_thresh, bShuffle_Exp, exp_thresh2=None, identifyTopTier=False):
    weights = np.array(test_results['weights'])
    actions = np.array(test_results['actions'])
    memories = np.array(test_results['memories'])
    values = np.array(test_results['values'])

    # # Keep highest weighting where there are 2 values for the same location
    # memories_dict = {}
    # for i in range(weights.shape[0]):
    #     m = tuple(memories[i, :])

    #     if (m in memories_dict):
    #         if (memories_dict[m]['weight'] < weights[i]):
    #             memories_dict[m]['weight'] = weights[i] # + memories_dict[m]['weight']/10
    #             memories_dict[m]['ind'] = i
    #     else:
    #         memories_dict[m] = {}
    #         memories_dict[m]['weight'] = weights[i]
    #         memories_dict[m]['ind'] = i

    # final_inds = []

    # for key, item in memories_dict.items():
    #     final_inds.append(item['ind'])

    # weights = weights[final_inds]
    # actions = actions[final_inds]
    # memories = memories[final_inds, :]
    # values = values[final_inds, :]


#    exp_thresh = useJenksToFindBreak(test_results['weights'])

    if identifyTopTier:
        weightsCopy = np.sort(np.copy(weights))[::-1]
    
        ratios = weightsCopy[:-1] / weightsCopy[1:]
        threshold = np.mean(ratios) - np.std(ratios)
        # cutoff_index = np.argmax(ratios < threshold)

        cutoff_index = len(weightsCopy) - 1
        meanRatio = np.mean(ratios)
        medianRatio = np.median(ratios)
        for i in range(1, len(ratios) - 1):
            if ratios[i] > ratios[i - 1] and (ratios[i] > meanRatio and ratios[i] > medianRatio): # or i > 0.5 * len(ratios)):
                cutoff_index = i
                break
        exp_thresh = weightsCopy[cutoff_index]

    # thresholded_actions = np.array([])
    # thresholded_memories = np.array([])
    # thresholded_values = np.array([])
    # thresholded_weights = np.array([])
    thresholded_actions = []
    thresholded_memories = []
    thresholded_values = []
    thresholded_weights = []

    if exp_thresh2 is None:
        thresholded_actions = actions[weights > exp_thresh]
        thresholded_memories = memories[weights > exp_thresh, :]
        thresholded_values = values[weights > exp_thresh, :]
        thresholded_weights = weights[weights > exp_thresh]
    else:
        steps = len(actions)
        threshold_increment = (exp_thresh2 - exp_thresh) / (steps - 1)
        threshold = exp_thresh
        for i in range(0, steps):
            if weights[i] > threshold:
                # thresholded_actions = np.append(thresholded_actions, actions[i])
                # thresholded_memories = np.append(thresholded_memories, np.array(memories[i, :]))
                # thresholded_values = np.append(thresholded_values, np.array(values[i, :]))
                # thresholded_weights = np.append(thresholded_weights, weights[i])
                thresholded_actions.append(actions[i])
                thresholded_memories.append(memories[i, :])
                thresholded_values.append(values[i, :])
                thresholded_weights.append(weights[i])
            threshold += threshold_increment
        thresholded_weights = np.array(thresholded_weights)
        thresholded_actions = np.array(thresholded_actions)
        thresholded_memories = np.array(thresholded_memories)
        thresholded_values = np.array(thresholded_values)

    memories_dict = {}
    for i in range(thresholded_weights.shape[0]):
        m = tuple(thresholded_memories[i, :])

        if (m in memories_dict):
            if (memories_dict[m]['weight'] < thresholded_weights[i]):
                memories_dict[m]['weight'] = thresholded_weights[i]
                memories_dict[m]['ind'] = i
        else:
            memories_dict[m] = {}
            memories_dict[m]['weight'] = thresholded_weights[i]
            memories_dict[m]['ind'] = i
    final_inds = []
    for key, item in memories_dict.items():
        final_inds.append(item['ind'])

    explanation_length = len(final_inds)

    if bShuffle_Exp:
        final_inds = np.random.choice(weights.shape[0], explanation_length, replace=False).tolist()
        exp_weights = weights[final_inds]
        exp_actions = actions[final_inds]
        exp_memories = memories[final_inds, :]
        exp_values = values[final_inds, :]
    else:
        exp_weights = thresholded_weights[final_inds]
        exp_actions = thresholded_actions[final_inds]
        exp_memories = thresholded_memories[final_inds, :]
        exp_values = thresholded_values[final_inds, :]

    return exp_actions, exp_memories, exp_values, exp_weights


import jenkspy
import pickle
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
import os

def useJenksToFindBreak(weights):
    # Calculate natural breaks
    weights = np.sort(weights)
    breaks = jenkspy.jenks_breaks(weights, 5)

    print(f"Natural Breaks: {breaks}")

    # Categorize data based on breaks
    lower_group = [x for x in weights if x <= breaks[1]]
    medium_group = [x for x in weights if breaks[1] < x <= breaks[2]]
    medium_group2 = [x for x in weights if breaks[2] < x <= breaks[3]]
    medium_group3 = [x for x in weights if breaks[3] < x <= breaks[4]]
    higher_group = [x for x in weights if x > breaks[4]]

    print(f"Lower group: {lower_group}")
    print(f"Medium group: {medium_group}")
    print(f"Medium group2: {medium_group2}")
    print(f"Medium group3: {medium_group3}")
    print(f"Higher group: {higher_group}")

    exp_thresh = breaks[2]
    print(f"Jenks {exp_thresh}")
    return exp_thresh

def useKMeansToFindBreak(weights):
        # Apply KMeans
    weights = np.sort(weights)
    weights = weights.reshape(-1, 1)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(weights)
    # Get cluster centers
    centers = kmeans.cluster_centers_.flatten()

    labels = kmeans.labels_
    #centers = kmeans.cluster_centers_

    # Group data points based on labels
    groups = {}
    for label in np.unique(labels):
        groups[label] = weights[labels == label].flatten().tolist()

    # Create a list of clusters with their centers
    clusters_with_centers = [(label, centers[label], groups[label]) for label in groups]

    # Sort clusters by their centers
    sorted_clusters_by_centers = sorted(clusters_with_centers, key=lambda x: x[1])


    # Extract cluster data
    lower_group = weights[labels == sorted_clusters_by_centers[0][0]].flatten().tolist()
    medium_group = weights[labels == sorted_clusters_by_centers[1][0]].flatten().tolist()
    medium_group2 = weights[labels == sorted_clusters_by_centers[2][0]].flatten().tolist()
    medium_group3 = weights[labels == sorted_clusters_by_centers[3][0]].flatten().tolist()
    higher_group = weights[labels == sorted_clusters_by_centers[4][0]].flatten().tolist()

    print(f"Lower group: {lower_group}")
    print(f"Medium group: {medium_group}")
    print(f"Medium group 2: {medium_group2}")
    print(f"Medium group 3: {medium_group3}")
    print(f"Higher group: {higher_group}")

    exp_thresh = max(medium_group)
    print(f"K Means {exp_thresh}")
    return exp_thresh

def useHierarchicalClsutering(weights):
    # Perform hierarchical clustering
    weights = np.sort(weights)
    Z = linkage(weights.reshape(-1, 1), method='ward')
    clusters = fcluster(Z, 5, criterion='maxclust')

    # Group data points based on cluster labels
    groups = {}
    for label in np.unique(clusters):
        groups[label] = weights[clusters == label].flatten().tolist()

    # Order Clusters by Mean Value
    clusters_with_means = [(label, np.mean(groups[label]), groups[label]) for label in groups]
    sorted_clusters_by_means = sorted(clusters_with_means, key=lambda x: x[1])

    print("Sorted Clusters by Mean Value:")
    for label, mean, group in sorted_clusters_by_means:
        print(f"Cluster {label}: Mean = {mean}, Points = {group}")




    # Extract cluster data
    lower_group = weights[clusters == sorted_clusters_by_means[0][0]].tolist()
    lower_group2 = weights[clusters == sorted_clusters_by_means[1][0]].tolist()
    higher_group = weights[clusters == sorted_clusters_by_means[2][0]].tolist()
    higher_group2 = weights[clusters == sorted_clusters_by_means[3][0]].tolist()
    higher_group3 = weights[clusters == sorted_clusters_by_means[4][0]].tolist()

    print(f"Lower group: {lower_group}")
    print(f"Lower group2: {lower_group2}")
    print(f"Higher group: {higher_group}")
    print(f"Higher group2: {higher_group2}")
    print(f"Higher group3: {higher_group3}")
    exp_thresh = max(lower_group2)
    print(f"Hierarchical {exp_thresh}")
    return exp_thresh


def returnExplanations(explanationFile, removeDuplicateWeights=False):
    with open(explanationFile, 'rb') as f:
        results = pickle.load(f)
    weights = np.array(results['weights'])
    actions = np.array(results['actions'])
    memories = np.array(results['memories'])
    values = np.array(results['values'])

    if removeDuplicateWeights:

        # Keep highest weighting where there are 2 values for the same location
        memories_dict = {}
        for i in range(weights.shape[0]):
            m = tuple(memories[i, :])

            if (m in memories_dict):
                if (memories_dict[m]['weight'] < weights[i]):
                    memories_dict[m]['weight'] = weights[i] #+ memories_dict[m]['weight']/10
                    memories_dict[m]['ind'] = i
            else:
                memories_dict[m] = {}
                memories_dict[m]['weight'] = weights[i]
                memories_dict[m]['ind'] = i

        final_inds = []

        for key, item in memories_dict.items():
            final_inds.append(item['ind'])

        weights = weights[final_inds]
        actions = actions[final_inds]
        memories = memories[final_inds, :]
        values = values[final_inds, :]

    results['weights'] = weights
    results['actions'] = actions
    results['memories'] = memories
    results['values'] = values


    return results


def groupSignificantExplanations(explanationFile):
    with open(explanationFile, 'rb') as f:
        results = pickle.load(f)

    weights = np.array(results['weights'])
    actions = np.array(results['actions'])
    memories = np.array(results['memories'])
    values = np.array(results['values'])

    # Keep highest weighting where there are 2 values for the same location
    memories_dict = {}
    for i in range(weights.shape[0]):
        m = tuple(memories[i, :])

        if (m in memories_dict):
            if (memories_dict[m]['weight'] < weights[i]):
                memories_dict[m]['weight'] = weights[i]
                memories_dict[m]['ind'] = i
        else:
            memories_dict[m] = {}
            memories_dict[m]['weight'] = thresholded_weights[i]
            memories_dict[m]['ind'] = i

    final_inds = []

    for item in memories_dict.items():
        final_inds.append(item['ind'])

    weights = weights[final_inds]
    actions = actions[final_inds]
    memories = memories[final_inds, :]
    values = values[final_inds, :]








    # Calculate natural breaks
    breaks = jenkspy.jenks_breaks(weights, 2)

    print(f"Natural Breaks: {breaks}")

    # Categorize data based on breaks
    lower_group = [x for x in weights if x <= breaks[1]]
    medium_group = [x for x in weights if x > breaks[1]]
    #higher_group = [x for x in weights if x > breaks[2]]

    print(f"Lower group: {lower_group}")
    print(f"Medium group: {medium_group}")
    #print(f"Higher group: {higher_group}")

    exp_thresh = breaks[1]
    print(f"Jenks {exp_thresh}")



    # Apply KMeans
    weights2 = weights.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(weights2)
    labels = kmeans.labels_
    #centers = kmeans.cluster_centers_

    # Extract cluster data
    lower_group = weights2[labels == 0].flatten().tolist()
    medium_group = weights2[labels == 1].flatten().tolist()
#    higher_group = weights[labels == 2].flatten().tolist()

    print(f"Lower group: {lower_group}")
    print(f"Medium group: {medium_group}")
    #print(f"Higher group: {higher_group}")

    exp_thresh = max(lower_group)
    print(f"K Means {exp_thresh}")



    # Perform hierarchical clustering
    Z = linkage(weights.reshape(-1, 1), method='ward')
    clusters = fcluster(Z, 2, criterion='maxclust')

    # Extract cluster data
    lower_group = weights[clusters == 1].tolist()
    higher_group = weights[clusters == 2].tolist()

    print(f"Lower group: {lower_group}")
    print(f"Higher group: {higher_group}")
    exp_thresh = max(lower_group)
    print(f"Hierarchical {exp_thresh}")

    thresholded_actions = actions[weights > exp_thresh]
    thresholded_memories = memories[weights > exp_thresh, :]
    thresholded_values = values[weights > exp_thresh, :]
    thresholded_weights = weights[weights > exp_thresh]

    memories_dict = {}
    for i in range(thresholded_weights.shape[0]):
        m = tuple(thresholded_memories[i, :])

        if (m in memories_dict):
            if (memories_dict[m]['weight'] < thresholded_weights[i]):
                memories_dict[m]['weight'] = thresholded_weights[i]
                memories_dict[m]['ind'] = i
        else:
            memories_dict[m] = {}
            memories_dict[m]['weight'] = thresholded_weights[i]
            memories_dict[m]['ind'] = i
    final_inds = []
    for key, item in memories_dict.items():
        final_inds.append(item['ind'])

    explanation_length = len(final_inds)

    bShuffle_Exp = False
    if bShuffle_Exp:
        final_inds = np.random.choice(weights.shape[0], explanation_length, replace=False).tolist()
        exp_weights = weights[final_inds]
        exp_actions = actions[final_inds]
        exp_memories = memories[final_inds, :]
        exp_values = values[final_inds, :]
    else:
        exp_weights = thresholded_weights[final_inds]
        exp_actions = thresholded_actions[final_inds]
        exp_memories = thresholded_memories[final_inds, :]
        exp_values = thresholded_values[final_inds, :]

    return exp_actions, exp_memories, exp_values, exp_weights

import os
import Plotters
import matplotlib.pyplot as plt



if __name__ == "__main__":
    plt.ioff()
    plt.interactive(True)
    plt.ion()
    observations = []

    explanationFile = '/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-08-22 11-41-02 91/Explanation_1_TestTrial_1.pkl'

    folder = os.path.dirname(explanationFile)
    mazeFile = os.path.join(folder, 'Maze.npy')
    maze = np.load(mazeFile)
    results = returnExplanations(explanationFile)
    if 'observations' not in results:
        # Specify the shape of the array, e.g., (3, 2) for a 3x2 array
        shape = (len(results['memories']), 2)
        # Create the array filled with -1
        results['observations'] = np.full(shape, -1)
    actions, memories, values, weights, observations = ExtractExplanation(results, 0.6, False, excludeWhereDQNWithin=.01)
    Plotters.ShowExplanation(maze, memories, actions, weights, observations, "Threshold .6")

    # Get the directory name of the existing file
    directory = os.path.dirname(explanationFile)
    # Name of the new file to be saved in the same directory
    explFile = 'threshold75.png'
    # Create the full path for the new file
    explPath = os.path.join(directory, explFile)
    Plotters.PlotExplanation(maze, memories, actions, weights, observations, explPath)

    actions, memories, values, weights, observations = ExtractExplanation(results, 0.4, False)
    Plotters.ShowExplanation(maze, memories, actions, weights, "Threshold .4")
    threshold = useJenksToFindBreak(results['weights'])
    print(f"Jenks: {threshold}")
    actions, memories, values, weights, observations = ExtractExplanation(results, threshold, False)
    Plotters.ShowExplanation(maze, memories, actions, weights, observations, "Jenks")
    threshold = useKMeansToFindBreak(results['weights'])
    print(f"KMeans: {threshold}")
    actions, memories, values, weights, observations = ExtractExplanation(results, threshold, False)
    Plotters.ShowExplanation(maze, memories, actions, weights, observations, "KMeans")
    threshold = useHierarchicalClsutering(results['weights'])
    print(f"Hierarchical: {threshold}")
    actions, memories, values, weights, observations = ExtractExplanation(results, threshold, False)
    Plotters.ShowExplanation(maze, memories, actions, weights, observations, "Hierarchical")
    Plotters.ShowDQN(maze, results['actions'], results['DQN_q'], "")
    plt.show(block=True)

