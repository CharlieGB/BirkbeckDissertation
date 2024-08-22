import os
import pickle
import numpy as np
import pandas as pd


def ParseIntoDataframes(dirs_to_compare):

    data_frames = []

    for model in dirs_to_compare:
        folders = os.listdir('GridWorld/Results/' + model)
        data_frames.append(ParseDataFrame(folders, model))

    return data_frames

def ParseDataFrame(folders, dir):

    results_dict = {'dir': [], 'rewards': [], 'lengths': [], 'chosen_explanation': [], 'maze': [],
                    'som_memories': [], 'som_actions': []}

    for folder in folders:

        if (folder == '.DS_Store'):
            pass
        else:
            results_dict['dir'].append(folder)

            with open('GridWorld/Results/' + dir + '/' + folder + '/Results.pkl', 'rb') as handle:
                dict = pickle.load(handle)

            results_dict['rewards'].append(dict['rewards'])
            results_dict['lengths'].append(dict['lengths'])

            with open('GridWorld/Results/' + dir + '/' + folder + '/som_contents.pkl', 'rb') as handle:
                som_contents = pickle.load(handle)

            results_dict['som_memories'].append(som_contents['w'].copy())
            results_dict['som_actions'].append(som_contents['actions'].copy())

            with open('GridWorld/Results/' + dir + '/' + folder + '/explanation.pkl', 'rb') as handle:
                explanations = pickle.load(handle)

            for key, value_dict in enumerate(explanations):
                key = 'Explanation_' + str(key)

                if (key not in results_dict):
                    results_dict[key] = []

                results_dict[key].append(value_dict)

            file = open('GridWorld/Results/' + dir + '/' + folder + '/Settings.txt', 'r')
            settings = file.readlines()
            file.close()

            chosen_explanation = -1

            for setting in settings:
                vals = setting.split(': ')

                if (vals[0] == 'chosen_explanation'):
                    chosen_explanation = float(vals[1])

                else:

                    if (vals[0] not in results_dict):
                        results_dict[vals[0]] = []

                    try:
                        results_dict[vals[0]].append(float(vals[1]))
                    except:
                        results_dict[vals[0]].append(vals[1])

            results_dict['maze'].append(np.load('GridWorld/Results/' + dir + '/' + folder + '/Maze.npy'))
            results_dict['chosen_explanation'].append(chosen_explanation)

    df = pd.DataFrame.from_dict(results_dict)

    return df