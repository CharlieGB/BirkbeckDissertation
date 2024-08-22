import random
import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.dqn.policies import (
    CnnPolicy,
    DQNPolicy,
    MlpPolicy,
    MultiInputPolicy,
    QNetwork,
)
from stable_baselines3.common.utils import get_schedule_fn


import optuna
from stable_baselines3.common.evaluation import evaluate_policy


import torch as th
import torch.optim as optim
from CTDL import CTDL
from CTDLPolicy import CTDLPolicy
import EventsForModel
import pickle
import os
import fnmatch
import csv
from datetime import datetime
from Parameters import maze_params, agent_params
import Utilities
import statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


import Maze

def getRandomSeed():
    seedRange=(0, 2**32-1)
    return random.randint(*seedRange)

def getModel(modelName, env, directory, somSize = 4, wDecay = 1, modelSeed = None, numpySeed = None):
    agent_params['SOM_size'] = somSize
    agent_params['w_decay'] = wDecay
    agent_params['agent_type'] = modelName

    if modelSeed is None:
        modelSeed = getRandomSeed()
    agent_params['model_random_seed'] = modelSeed
    if numpySeed is None:
        numpySeed = getRandomSeed()
        np.random.seed(numpySeed)  # Needs updating to new random generator code
    agent_params['numpy_random_seed'] = numpySeed


    if modelName == "A2C":
        model = A2C(
            "MlpPolicy", env, verbose=1, device="cuda"
        )  # , tensorboard_log=log_dir)
    elif modelName == "DQNVanilla":
        model = DQN(policy="MlpPolicy", env=env, verbose=0)
    elif modelName == "DQN":
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=50000,
            # learning_starts=100,
            batch_size=64,
            # tau=1.0,
            # gamma=0.99,
            # train_freq=16,
            gradient_steps=4,
            target_update_interval=250,  # this really steps up learning
            exploration_fraction=0.7,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            # max_grad_norm=10,
#            tensorboard_log=log_dir,
            # policy_kwargs=dict(normalize_images=False),
            policy_kwargs=dict(net_arch=[256, 256, 4]),  # big improvment
            verbose=1,
            # device='cuda',
        )
    elif modelName == "DQNCTDL":
        # Create an instance of the custom policy
        # learning_rate = get_schedule_fn(0.00025) # Unused, but for future reference
        # policy = DQNPolicy(env.observation_space, env.action_space, learning_rate)

        # # Define a custom RMSprop optimizer with parameters
        # optimizer = optim.RMSprop(
        #     params=policy.parameters(), lr=0.00025,
        #     eps=0.01, momentum=0.95,
        # )
        # lr=1e-2,
        # alpha=0.99,
        # eps=1e-8,
        # weight_decay=0,
        # momentum=0,

        optimizer_kwargs = {
#            'lr': 1e-3,         # Learning rate / Use the parameter sent to DQN directly
#            'alpha': 0.99,      # Smoothing constant
            'eps': 0.01,        # Term added to the denominator to improve numerical stability
#            'weight_decay': 0,  # Weight decay (L2 penalty)
            'momentum': 0.95       # Momentum factor
        }

        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.00025,
            buffer_size=100000,
            # learning_starts=100,
            batch_size=32,
            #tau=1,
            #gamma=0.95,
            # train_freq=8,
            gradient_steps=4,
            target_update_interval=250,  # this really steps up learning
            exploration_fraction=0.4,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            # max_grad_norm=10,
            # tensorboard_log=log_dir,
            # policy_kwargs=dict(normalize_images=False),
            policy_kwargs=dict(
                net_arch=[128, 128],
#                optimizer_class=optim.RMSprop,
#                optimizer_kwargs=optimizer_kwargs,
                ),
            # device='cuda',
            # optimal settings = {'model_settings': {'learning_rate': 0.00025, 'buffer_size': 100000, 'learning_starts': 100, 'batch_size': 128, 'tau': 0.075, 'gamma': 0.94, 'train_freq': TrainFreq(frequency=8, unit=), 'gradient_steps': 8, 'target_update_interval': 250, 'exploration_fraction': 0.35, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.1, 'policy_kwargs': {'net_arch': [128, 128]}}}
            #seed=modelSeed,
        )
    elif modelName == "PPO":
        model = PPO(
            "MlpPolicy", env, verbose=1, device="cuda"
        )  # , tensorboard_log=log_dir)
    elif modelName == "CTDL":
        model = CTDL(
            policy=CTDLPolicy,
            ignore_som=False,
            env=env,
            learning_rate=0.00025,
            buffer_size=100000,
            # learning_starts=100,
            batch_size=32,
            # tau=0.075,
            # gamma=0.94,
            # train_freq=8,
            # gradient_steps=8,
            target_update_interval=250,  # this really steps up learning
            exploration_fraction=0.4,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            # max_grad_norm=10,
#            tensorboard_log=log_dir,
            # policy_kwargs=dict(normalize_images=False),
            policy_kwargs=dict(net_arch=[128, 128]),
#            verbose=1,
            # device='cuda',
            directory=directory,
            seed=modelSeed,
        )
    else:
        print(f"Unknown model name: {modelName}")
        quit()

    print(f"Optimizer values: f{model.policy.optimizer.defaults}")
    return model

def getEnvironment(mazeNumber=1, envSeed_action=None,  envSeed_observation=None):
    maze_params['maze_num'] = mazeNumber
    env = gym.make("maze-v0")
    if envSeed_action is None:
        envSeed_action = getRandomSeed()
    if envSeed_observation is None:
        envSeed_observation = getRandomSeed()
    agent_params['env_random_seed_action'] = envSeed_action
    agent_params['env_random_seed_observation'] = envSeed_observation

    env.action_space.seed(envSeed_action)
    env.observation_space.seed(envSeed_observation)

    return env

def testModel(model, env, directory, trial=1, deterministic=True):
    explanationLength = 0
    obs = env.reset()[0]
    model.set_env(env)
    terminated = False
    testTrial = 1
    obs = env.reset()[0]
    terminated = False
    while not terminated:
        initial_obs = obs
        action, _ = model.predict(
            observation=obs, deterministic=deterministic
        )  # Turn on deterministic, so predict always returns the same behavior
        obs, reward, terminated, _, _ = env.step(action)
        if isinstance(model, CTDL):
            model.test_rewards.append(reward)
            model.test_actions.append(action)
            model.test_weights.append(model.w)
            model.test_memories.append(
                model.SOM.SOM_layer.units["w"][model.best_unit]
            )
            model.test_observations.append(initial_obs)
            model.test_values.append(model.QValues[model.best_unit, :])
            model.test_DQN_q_values.append(model.DQN_q_values[0])

        if terminated:
            if isinstance(model, CTDL):
                model.directory = directory
                explanationLength = model.RecordTestResults(env, trial, testTrial)
                model.SaveTestResults()

    saveEnvResults(env, directory, "test_results", explanationLength, trial)
    

def trainModel(model, iterations, directory, iters=1, progressModel=[]):

    # Where to store trained model and logs
    model_dir = directory + "models"
    log_dir = directory + "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = model.get_env().envs[0]
    env.unwrapped.SaveMaze(directory)
    Utilities.RecordSettings(directory, maze_params, agent_params)


    modelName = model.__class__.__name__

    startTime = datetime.now()

    if isinstance(model, CTDL):
        model = model.learn(total_timesteps=iterations, reset_num_timesteps=False, progress_model=progressModel)  # , callback=events) # train
    else:
        model = model.learn(total_timesteps=iterations, reset_num_timesteps=False)

    endTime = datetime.now()
    difference = endTime - startTime
    print(f"Time taken {difference}")

    model.save(f"{model_dir}/{modelName}_{iterations}_{iters}") 


    if isinstance(model, CTDL):
        model.PlotValueFunction()
        model.PlotResults()
        model.saveSOM()

    saveEnvResults(env, directory, "training_results", 0, iters)
    saveModelSettings(model, directory)

    return model

def saveModelSettings(model, directory):

    # Extract the model's hyperparameters
    params = model.get_parameters()

    # Get additional important attributes
    settings = {
        "learning_rate": model.learning_rate,
        "buffer_size": model.buffer_size,
        "learning_starts": model.learning_starts,
        "batch_size": model.batch_size,
        "tau": model.tau,
        "gamma": model.gamma,
        "train_freq": model.train_freq,
        "gradient_steps": model.gradient_steps,
        #"n_episodes_rollout": model.n_episodes_rollout,
        "target_update_interval": model.target_update_interval,
        "exploration_fraction": model.exploration_fraction,
        "exploration_initial_eps": model.exploration_initial_eps,
        "exploration_final_eps": model.exploration_final_eps,
        "policy_kwargs": model.policy_kwargs,
    }

    # Combine both settings and params if needed
    full_settings = {"model_settings": settings}

    # Save to a text file
    with open(directory + "dqn_settings.pkl", "wb") as f:
        pickle.dump(full_settings, f)

def saveEnvResults(env, directory, filename, explanationLength=0, iteration = 1):
    gymEnvironment = env.unwrapped

    csvFile = directory + filename + str(iteration) + '.csv'
    # gymEnvironment.results["Model"] = modelName
    gymEnvironment.results["Steps"] = round(sum(gymEnvironment.results["lengths"]))
    gymEnvironment.results["Count"] = len(gymEnvironment.results["lengths"])

    threshold = 0.0
    # Use boolean indexing to create a boolean array where the condition is met
    lengthArray = np.array(gymEnvironment.results["rewards"])
    boolean_array = lengthArray >= threshold
    # Sum the boolean array to count the number of True values
    countIdealEpisodes = round(np.sum(boolean_array))
    gymEnvironment.results["Ideal episodes"] = countIdealEpisodes

    gymEnvironment.results["Length mean/sd"] = [
        round(float(statistics.mean(gymEnvironment.results["lengths"])),1),
        round(float(statistics.pstdev(gymEnvironment.results["lengths"])),1),
    ]
    gymEnvironment.results["Reward mean/sd"] = [
        round(float(statistics.mean(gymEnvironment.results["rewards"])),2),
        round(float(statistics.pstdev(gymEnvironment.results["rewards"])),2),
    ]
    gymEnvironment.results['lengths'] = [round(value) for value in gymEnvironment.results['lengths']]
    gymEnvironment.results['rewards'] = [round(value, 2) for value in gymEnvironment.results['rewards']]
    gymEnvironment.results['Explanation len'] = explanationLength


    # Open the file in write mode
    with open(csvFile, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header (keys of the dictionary)
        writer.writerow(gymEnvironment.results.keys())
        
        # Write the rows (values of the dictionary)
        for i in range(len(gymEnvironment.results['lengths'])):
            #rowData = [gymEnvironment.results['rewards'][i], gymEnvironment.results['lengths'][i]]
            rowData = []
            for j in gymEnvironment.results.keys():
                if isinstance(gymEnvironment.results[j], list):
                    if i < len(gymEnvironment.results[j]):
                        rowData.append(gymEnvironment.results[j][i])
                elif i < 2:
                    if i == 0:
                        rowData.append(gymEnvironment.results[j])
                    else:
                        rowData.append("")
            writer.writerow(rowData)
            #writer.writerows(zip(*gymEnvironment.results.values()))

    print(f"CSV data has been written to {csvFile}")

    with open(directory + filename + str(iteration) + ".pkl", "wb") as handle:
        pickle.dump(gymEnvironment.results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    saveToMainLog(env, gymEnvironment.results, directory)
    graphResults(gymEnvironment.results, directory, iteration)
    gymEnvironment.results = {'rewards': [], 'lengths': []}



def graphResults(resultsCTDL, directory, iteration = 1):
    if len(resultsCTDL['lengths']) < 10:
        return
    resultsCTDL['Cumulative'] = np.cumsum(resultsCTDL['lengths'])
    resultsCTDL['Episodes'] = np.arange(1, len(resultsCTDL['lengths'])+1)

    # Create the first plot
    plt.plot(resultsCTDL['Cumulative'], resultsCTDL['Episodes'], marker='o', linestyle='-', color='g', linewidth=1, markersize=1, label='CTDL Cumulative Episodes')

    # Add legend (color key) with customized location and appearance
    plt.legend(loc='best', fontsize='small', frameon=True, fancybox=True, shadow=True)

    # Add title and labels
    plt.title("Cumulative completed episodes by step count")
    plt.xlabel("Number of steps")
    plt.ylabel("Completed episodes")
    
    # Format the x-axis and y-axis to include commas
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

    plt.savefig(os.path.join(directory, 'Graph Episodes ' + str(iteration) + '.png'))
    plt.figure()


    resultsCTDL['Cumulative reward'] = np.cumsum(resultsCTDL['rewards'])


    # Create the second plot
    plt.plot(resultsCTDL['Cumulative'], resultsCTDL['Cumulative reward'], marker='x', linestyle='--', color='r', linewidth=1, markersize=1, label='CTDL')

    # Add legend (color key) with customized location and appearance
    plt.legend(loc='best', fontsize='small', frameon=True, fancybox=True, shadow=True)

    # Add title and labels
    plt.title("Cumulative rewards by step count")
    plt.xlabel("Number of steps")
    plt.ylabel("Cumulative rewards")
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

    plt.savefig(os.path.join(directory, 'Graph Rewards ' + str(iteration) + '.png'))
    plt.figure()


def saveToMainLog(env, results, directory):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    # Define the file name
    filename = os.path.join(log_dir, "main_log.csv")
    dataRow = {
        "Maze": maze_params['maze_num'],
#        "Agent": env.unwrapped.agentType,
#        "Model": results["Model"],
        "Steps": results["Steps"],
        "Count": results["Count"],
        "Ideal episodes": results["Ideal episodes"],
        "Length mean": results["Length mean/sd"][0],
        "Length sd": results["Length mean/sd"][1],
        "Reward mean": results["Reward mean/sd"][0],
        "Reward sd": results["Reward mean/sd"][1],
        "Folder": directory,
    }

    dataRow.update(agent_params)
    dataRow.update(maze_params)
    dataRow.update({"Explanation len": results["Explanation len"]})

    # Check if the file already exists
    file_exists = os.path.isfile(filename)

    # Open the file in append mode if it exists, otherwise write mode
    with open(filename, mode='a' if file_exists else 'w', newline='') as csvfile:
        # Define the fieldnames (header)
        fieldnames = dataRow.keys()
        
        # Create a DictWriter object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header only if the file doesn't already exist
        if not file_exists:
            writer.writeheader()
        
        # Write the data
        writer.writerow(dataRow)




def loadModel(model, env, fileName):
    model = model.load(fileName) # Save a trained model every TIMESTEPS
    model.set_env(env)
    return model

def loadSOMOnly(model, env, modelPath):
    SOMpkl = modelPath + "_SOM.pkl"
    model.loadSOMBin(SOMpkl) # Save a trained model every TIMESTEPS
    model.set_env(env)
    return model


def list_files(directory, file_spec):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, file_spec):
            matching_files.append(os.path.join(root, filename))
    return matching_files

def loadTestInterimModels(mazeNo, directory):
    env = getEnvironment(mazeNo)
    model = getModel("CTDL", env, directory)
    modelFiles = list_files(directory, "CTDL_ProgressModel_*.zip")
    for modelFile in modelFiles:
        # Find the index of the last '_' character
        last_underscore_index = modelFile.rfind('_')
        # Find the index of the first '.' character after the last '_'
        first_dot_index = modelFile.find('.', last_underscore_index)
        uniqueid = modelFile[last_underscore_index + 1:first_dot_index]
        newDir = os.path.join(directory, "Interim " + uniqueid + "/")
        os.makedirs(newDir, exist_ok=True)
        # newDir = CreateResultsDirectory(" Interim " + uniqueid)
        env = getEnvironment(mazeNo)
        model = loadModel(model, env, modelFile[:-4])
        testModel(model, env, newDir)



def printNetwork(model):
    import torch.nn as nn
    pytorch_model = model.policy
    state_dict = pytorch_model.state_dict()


    # Query the number of input and output neurons
    for name, layer in pytorch_model.named_modules():
        if isinstance(layer, nn.Linear):
            print(f"Layer: {name}, Input Neurons: {layer.in_features}, Output Neurons: {layer.out_features}")
    for layer_name, weights in state_dict.items():
        print(f"Layer: {layer_name}")
        print(weights)
    

    # Print out the weights and biases of the first layer
    first_layer_weights = state_dict['q_net.q_net.0.weight']
    first_layer_biases = state_dict['q_net.q_net.0.bias']

    print("First layer weights:")
    print(first_layer_weights)

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    plt.ioff()
    plt.interactive(True)

    # Convert weights to numpy for easier manipulation
    weights_numpy = first_layer_weights.detach().numpy()

    # Plotting the weights of the first layer
    plt.imshow(weights_numpy, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("First Layer Weights")
    plt.xlabel("Neurons")
    plt.ylabel("Inputs")
    plt.show()
    print("and that's it folks")


    # Convert weights to numpy for easier manipulation
    weights_numpy = first_layer_weights.numpy()
    biases_numpy = first_layer_biases.numpy()

    # Determine the size of the network
    num_input_features = weights_numpy.shape[1]
    num_output_neurons = weights_numpy.shape[0]

    # Adjust the figure size based on the network size
    fig_width = max(8, num_input_features // 2)
    fig_height = max(6, num_output_neurons // 2)

    # Plotting weights and biases
    plt.figure(figsize=(fig_width, fig_height))

    # Plotting weights
    plt.subplot(1, 2, 1)
    plt.imshow(weights_numpy, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("First Layer Weights")
    plt.xlabel("Input Features")
    plt.ylabel("Output Neurons")

    # Plotting biases
    plt.subplot(1, 2, 2)
    plt.bar(range(len(biases_numpy)), biases_numpy, color='orange')
    plt.title("First Layer Biases")
    plt.xlabel("Output Neurons")
    plt.ylabel("Bias Value")

    plt.tight_layout()
    plt.show()

    print("and that's it again folks")



def CreateResultsDirectory(uniqueId=None):
    today = datetime.now()
    # Format the date as a string, e.g., "2024-07-13"
    dirAdd = ""
    if uniqueId is not None:
        dirAdd = f" {uniqueId}"
    else:
        dirAdd = f" {today.microsecond // 10000}"
    date_str = today.strftime("%Y-%m-%d %H-%M-%S") + dirAdd
    # Define the directory name based on the formatted date
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = dir_path + "/Results/"
    # Create the directory
    os.makedirs(results_dir, exist_ok=True)
    results_dir += date_str + "/"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def testDQN():
    for i in range(1, 4):
        directory = CreateResultsDirectory()
        env = getEnvironment(1)
        model = getModel("DQNCTDL", env, directory)
        model = trainModel(model, 100000, directory)
        #env = getEnvironment(6)
        #model =loadModel(model, env, '/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-07-24 21-41-00/models/DQN_200000')
        testModel(model, env, directory)

def testDQNVanilla():
    directory = CreateResultsDirectory()
    env = getEnvironment(1)
    model = getModel("DQNVanilla", env, directory)
    model = trainModel(model, 100000, directory)
    #env = getEnvironment(6)
    #model =loadModel(model, env, '/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-07-24 21-41-00/models/DQN_200000')
    testModel(model, env, directory)

def testDQNVanillaLoad():
    directory = CreateResultsDirectory()
    env = getEnvironment(2)
    model = getModel("DQNVanilla", env, directory)
    model = loadModel(model, env, 'C:/Users/charl/Documents/Dissertation/Torch23Python311/DissertationWorkingCode/Results/2024-07-30 18-40-08/models/DQN_600000_1')
    testModel(model, env, directory)

def testDQNLoad():
    directory = CreateResultsDirectory()
    env = getEnvironment(2)
    model = getModel("DQNCTDL", env, directory)
    #env = getEnvironment(6)
    model =loadModel(model, env, 'C:/Users/charl/Documents/Dissertation/Torch23Python311/DissertationWorkingCode/Results/2024-07-30 17-26-48/models/DQN_600000_1')
    #trainModel(model, 20000, directory)
#    trainModel(model, 150000, directory)
    testModel(model, env, directory)

def testCTDL():
    for mazeNo in (1, 2, 3, 4, 5):
        for i in range(3):
            directory = CreateResultsDirectory()
            env = getEnvironment(mazeNo)
            model = getModel("CTDL", env, directory, 4)
            model = trainModel(model, 100000, directory, progressModel=[10000, 20000, 30000, 40000, 50000, 60000])
            testModel(model, env, directory)

def testCTDL2():
    for mazeNo in (2, 1):
        for i in range(10):
            directory = CreateResultsDirectory()
            env = getEnvironment(mazeNo)
            model = getModel("CTDL", env, directory, 4)
            model = trainModel(model, 100000, directory)
            #env = getEnvironment(6)
            testModel(model, env, directory)

def quicktestCTDL(mazeNo = 1, steps = 20000, somSize = 4):
    directory = CreateResultsDirectory()
    env = getEnvironment(mazeNo)
    model = getModel("CTDL", env, directory, somSize)
    model = trainModel(model, steps, directory)
    testModel(model, env, directory)

def testCTDLLoad(envNo, modelPath):
    directory = CreateResultsDirectory()
    env = getEnvironment(envNo)
    model = getModel("CTDL", env, directory)
    model = loadModel(model, env, modelPath)
    #model = loadModel(model, env, '/Users/charles/Development/Dissertation/DissertationWorkingCode/models/CTDL_150000')
    #env = getEnvironment(6)
    testModel(model, env, directory)

def getSOMModel(env, directory, modelPath, SOMSize=4):
    model = getModel("CTDL", env, directory, SOMSize)
    model = loadSOMOnly(model, env, modelPath)
    return model

def testSOMModel(envNo, modelPath, SOMSize=4, trainSteps=10000):
    directory = CreateResultsDirectory()
    env = getEnvironment(envNo)
    model = getSOMModel(env, directory, modelPath, SOMSize)
    if trainSteps > 0:
        model = trainModel(model, trainSteps, directory)
    testModel(model, env, model.directory)
    
def testSOMSize():
    for mazeno in range(4, 5):
        for repeat in range(3): # temporarily for maze 4, only need 4 loops
            for somSize in (3, 4, 5, 6):
                trainingsize = 100000
                directory = CreateResultsDirectory()
                env = getEnvironment(mazeno)
                model = getModel("CTDL", env, directory, somSize)
                model = trainModel(model, trainingsize, directory)
                testModel(model, env, directory)

def testWDecay():
    for mazeNo in (3, 4, 5):
         for i in range(3):
            for wDecay in (0.5, 0.7, 1, 1.5, 2, 10, 4, 20):
                directory = CreateResultsDirectory()
                env = getEnvironment(mazeNo)
                model = getModel("CTDL", env, directory, 4, wDecay=wDecay)
                model = trainModel(model, 100000, directory)
                testModel(model, env, directory)

def testExistingModel(modelFile):
        directory = CreateResultsDirectory()
        env = getEnvironment(1)
        model = getModel("CTDL", env, directory)
        model = loadModel(model, env, modelFile)
        testModel(model, env, directory)

def testDefaultModel():
        directory = CreateResultsDirectory()
        env = getEnvironment(1)
        model = getModel("CTDL", env, directory)
        trainModel(model, 150000, directory)
        testModel(model, env, directory)

def testDifferentStages():
        directory = CreateResultsDirectory()
        env = getEnvironment(1)
        model = getModel("CTDL", env, directory, somSize=4)
        model = trainModel(model, 10000, directory)
        testModel(model, env, directory)
        model = trainModel(model, 20000, directory, 2)
        testModel(model, env, directory, 2)
        model = trainModel(model, 30000, directory, 3)
        testModel(model, env, directory, 3)
        model = trainModel(model, 40000, directory, 4)
        testModel(model, env, directory, 4)

def progressModel():
        directory = CreateResultsDirectory()
        env = getEnvironment(1)
        model = getModel("CTDL", env, directory, somSize=4)
        model = trainModel(model, 100000, directory, progressModel=[50000,80000,90000])
        testModel(model, env, directory)

def testProgressModel():
    directory = CreateResultsDirectory()
    env = getEnvironment(1)
    model = getModel("CTDL", env, directory)
    model = loadModel(model, env, '/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-07-29 08-58-37/CTDL_ProgressModel_50000')
    testModel(model, env, directory, 2)
    model = loadModel(model, env, '/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-07-29 08-58-37/CTDL_ProgressModel_80000')
    testModel(model, env, directory, 3)
    model = loadModel(model, env, '/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-07-29 08-58-37/CTDL_ProgressModel_90000')
    testModel(model, env, directory, 4)


def graphSOMSizes():
    with open('/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-08-21 14-56-34 66 - chosen 3/training_results1.pkl', 'rb') as f:
        results9 = pickle.load(f)
    results9['Cumulative'] = np.cumsum(results9['lengths'])
    results9['Episodes'] = np.arange(1, len(results9['lengths'])+1)
#    with open('/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-07-31 17-30-16/training_results1.pkl', 'rb') as f:
    with open('/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-08-21 15-00-40 17 0 chosen 4/training_results1.pkl', 'rb') as f:
        results16 = pickle.load(f)
    results16['Cumulative'] = np.cumsum(results16['lengths'])
    results16['Episodes'] = np.arange(1, len(results16['lengths'])+1)


#    with open('/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-07-31 18-37-24/training_results1.pkl', 'rb') as f: # Maze 5
    with open('/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-08-21 14-47-30 26 chosen 5/training_results1.pkl', 'rb') as f:
        results25 = pickle.load(f)
    results25['Cumulative'] = np.cumsum(results25['lengths'])
    results25['Episodes'] = np.arange(1, len(results25['lengths'])+1)

    with open('/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-08-21 14-51-59 41 - chosen 6/training_results1.pkl', 'rb') as f:
        results36 = pickle.load(f)
    results36['Cumulative'] = np.cumsum(results36['lengths'])
    results36['Episodes'] = np.arange(1, len(results36['lengths'])+1)


    # Create the first plot
    plt.plot(results9['Cumulative'], results9['Episodes'], marker='o', linestyle='-', color='g', linewidth=1, markersize=1, label='SOM 9')


    # Create the first plot
    plt.plot(results16['Cumulative'], results16['Episodes'], marker='o', linestyle='-', color='b', linewidth=1, markersize=1, label='SOM 16')

    # Create the second plot
    plt.plot(results25['Cumulative'], results25['Episodes'], marker='o', linestyle='-', color='r', linewidth=1, markersize=1, label='SOM 25')

    # Create the second plot
    plt.plot(results36['Cumulative'], results36['Episodes'], marker='o', linestyle='-', color='y', linewidth=1, markersize=1, label='SOM 36')

    #plt.plot(results['Cumulative'], results['Episodes'], marker='o')
    # Add legend (color key) with customized location and appearance
    plt.legend(loc='best', fontsize='small', frameon=True, fancybox=True, shadow=True)
    # Add title and labels
    plt.title("Cumulative completed episodes by step count")
    plt.xlabel("Number of steps")
    plt.ylabel("Completed episodes")
    
    # Format the x-axis and y-axis to include commas
    # Format the x-axis and y-axis to include commas
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

    # fig, ax = plt.subplots()
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

    plt.show()
    plt.savefig('progress2.png')
    plt.figure()


def graphCountVsCompleted():
    with open('/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-08-08 20-17-34 38 DQN Original/training_results1.pkl', 'rb') as f:
        resultsDQNOriginal = pickle.load(f)
    resultsDQNOriginal['Cumulative'] = np.cumsum(resultsDQNOriginal['lengths'])
    resultsDQNOriginal['Episodes'] = np.arange(1, len(resultsDQNOriginal['lengths'])+1)
#    with open('/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-07-31 17-30-16/training_results1.pkl', 'rb') as f:
    with open('/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-08-08 20-33-21 50 DQN Modified/training_results1.pkl', 'rb') as f:
        resultsDQN = pickle.load(f)
    resultsDQN['Cumulative'] = np.cumsum(resultsDQN['lengths'])
    resultsDQN['Episodes'] = np.arange(1, len(resultsDQN['lengths'])+1)


#    with open('/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-07-31 18-37-24/training_results1.pkl', 'rb') as f: # Maze 5
    with open('/Users/charles/Development/Dissertation/DissertationWorkingCode/Results/2024-08-08 20-45-40 43 CTDL/training_results1.pkl', 'rb') as f:
        resultsCTDL = pickle.load(f)
    resultsCTDL['Cumulative'] = np.cumsum(resultsCTDL['lengths'])
    resultsCTDL['Episodes'] = np.arange(1, len(resultsCTDL['lengths'])+1)


    # Create the first plot
    plt.plot(resultsDQNOriginal['Cumulative'], resultsDQNOriginal['Episodes'], marker='o', linestyle='-', color='g', linewidth=1, markersize=1, label='DQN Original')


    # Create the first plot
    plt.plot(resultsDQN['Cumulative'], resultsDQN['Episodes'], marker='o', linestyle='-', color='b', linewidth=1, markersize=1, label='DQN SB3')

    # Create the second plot
    plt.plot(resultsCTDL['Cumulative'], resultsCTDL['Episodes'], marker='x', linestyle='--', color='r', linewidth=1, markersize=1, label='CTDL')

    #plt.plot(results['Cumulative'], results['Episodes'], marker='o')
    # Add legend (color key) with customized location and appearance
    plt.legend(loc='best', fontsize='small', frameon=True, fancybox=True, shadow=True)
    # Add title and labels
    plt.title("Cumulative completed episodes by step count")
    plt.xlabel("Number of steps")
    plt.ylabel("Completed episodes")
    
    # Format the x-axis and y-axis to include commas
    # Format the x-axis and y-axis to include commas
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

    # fig, ax = plt.subplots()
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

    plt.show()
    plt.savefig('progress2.png')
    plt.figure()


    resultsDQNOriginal['Cumulative reward'] = np.cumsum(resultsDQNOriginal['rewards'])

    resultsDQN['Cumulative reward'] = np.cumsum(resultsDQN['rewards'])


    resultsCTDL['Cumulative reward'] = np.cumsum(resultsCTDL['rewards'])

    # Create the first plot
    plt.plot(resultsDQNOriginal['Cumulative'], resultsDQNOriginal['Cumulative reward'], marker='o', linestyle='-', color='g', linewidth=1, markersize=1, label='DQN Original')


    # Create the first plot
    plt.plot(resultsDQN['Cumulative'], resultsDQN['Cumulative reward'], marker='o', linestyle='-', color='b', linewidth=1, markersize=1, label='DQN SB3')

    # Create the second plot
    plt.plot(resultsCTDL['Cumulative'], resultsCTDL['Cumulative reward'], marker='x', linestyle='--', color='r', linewidth=1, markersize=1, label='CTDL')

    #plt.plot(results['Cumulative'], results['Episodes'], marker='o')
    # Add legend (color key) with customized location and appearance
    plt.legend(loc='best', fontsize='small', frameon=True, fancybox=True, shadow=True)
    # Add title and labels
    plt.title("Cumulative rewards by step count")
    plt.xlabel("Number of steps")
    plt.ylabel("Cumulative rewards")
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    plt.show()
    os.system("pause")
    plt.savefig('progress3.png')


def objective(trial):
    env = getEnvironment(1)

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    buffer_size = trial.suggest_int('buffer_size', 10000, 100000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    gamma = trial.suggest_uniform('gamma', 0.9, 0.99)
    target_update_interval = trial.suggest_int('target_update_interval', 100, 10000)
    train_freq = trial.suggest_categorical('train_freq', [1, 4, 8, 16])
    gradient_steps = trial.suggest_categorical('gradient_steps', [1, 4, 8, 16])
    tau = trial.suggest_uniform('tau', 0.01, 0.1)
    exploration_fraction = trial.suggest_uniform('exploration_fraction', 0.1, 0.5)
        # Network architecture: number of layers and number of neurons in each layer
    net_arch = trial.suggest_categorical('net_arch', [[64, 64], [128, 128], [256, 256, 128], [512, 512]])
    policy_kwargs = dict(net_arch=net_arch)

    model = DQN('MlpPolicy', env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                gamma=gamma,
                target_update_interval=target_update_interval,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                tau=tau,
                verbose=0,
                exploration_fraction=exploration_fraction,
                # exploration_fraction=0.3,
                # exploration_initial_eps=1.0,
                # exploration_final_eps=0.1,
                policy_kwargs=policy_kwargs
    )
    
    model.learn(total_timesteps=200000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)
    return mean_reward

# Define the callback function to write trial results to a CSV file
def csv_callback(study, trial):

    file_name = 'optuna_trial_results.csv'
    header = ['trial_number'] + list(trial.params.keys()) + ['value']
    
    # Check if file exists to write the header only once
    try:
        with open(file_name, 'r') as f:
            pass
    except FileNotFoundError:
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [trial.number] + list(trial.params.values()) + [trial.value]
        writer.writerow(row)

def plotSOMContents(env, file):
    #directory = CreateResultsDirectory()
    directory = ''
    env = getEnvironment(env)
    model = getModel("CTDL", env, directory)
    model = loadModel(model, env, file)
    model.PlotSOMContents()

def optimise():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, callbacks=[csv_callback])

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')   

def tryAllSOMParameters():
    for somSigmaConst in (.001, .1):
        for somAlpha in (1, .01):
            for somSigma in (1, .1):
                for QAlpha in (.1, .9):
                    for TD_Decay in (100, 1):
                        directory = CreateResultsDirectory()
                        env = getEnvironment(5)
                        agent_params['SOM_alpha'] = somAlpha
                        agent_params['SOM_sigma'] = somSigma
                        agent_params['SOM_sigma_const'] = somSigmaConst
                        agent_params['TD_Decay'] = TD_Decay
                        agent_params['Q_alpha'] = QAlpha
                        model = getModel("CTDL", env, directory, 4)
                        model = trainModel(model, 100000, directory)
                        testModel(model, env, directory)

if __name__ == "__main__":
    quicktestCTDL()
