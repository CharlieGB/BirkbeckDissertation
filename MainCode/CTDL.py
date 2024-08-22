import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
import pickle
import csv

import os

from Plotters import *
import statistics
from CTDLPolicy import CTDLPolicy

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from stable_baselines3.dqn import DQN


from Parameters import agent_params, maze_params
from SOM import SOM
from ExplanationUtils import ExtractExplanation



SelfCTDL = TypeVar("SelfCTDL", bound="CTDL")


class CTDL(DQN):
    """
    CTDL - Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[CTDLPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        ignore_som: bool = False,
        directory: str = None,
        cuda: bool = False,
    ) -> None:
        policy: CTDLPolicy
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        # Initialize the SOM and related variables
        self.SOM = SOM("SOM", maze_params['width'], maze_params['height'], 2,  agent_params['SOM_size'],
                       agent_params['SOM_alpha'], agent_params['SOM_sigma'],
                       agent_params['SOM_sigma_const'])
        self.Q_alpha = agent_params['Q_alpha']
        self.QValues = np.zeros((agent_params['SOM_size'] * agent_params['SOM_size'], 4))
        self.update_mask = np.ones((self.SOM.SOM_layer.num_units))
        self.weighting_decay = agent_params['w_decay']
        self.discount_factor = 0.99
        self.TD_decay = agent_params['TD_decay']
        if hasattr(self, 'policy'):
            self.policy.ctdl = self
        self.ignoreSom = ignore_som
        self.gymEnvironment = env

        if directory is not None:
            self.recordData = True
            self.directory = directory

        self.maze_width = maze_params['width']
        self.maze_height = maze_params['height']
        self.maze_random_seed = maze_params['random_seed']
        self.maze_type = maze_params['type']

        self.explanation_length = agent_params['exp_length']

        self.weighting_decay = agent_params['w_decay']
        self.TD_decay = agent_params['TD_decay']

        self.discount_factor = 0.99
        self.epsilon = 0
        self.final_epsilon = .9 #1
        self.num_epsilon_trials = agent_params['e_trials']
        self.epsilon_increment = self.final_epsilon / self.num_epsilon_trials

        self.c = 10000
        self.ci = 0

        self.results = {'rewards': [], 'lengths': []}
        self.trial_reward = 0
        self.trial_length = 0
        self.plot_num = 0

        self.prev_state = None
        self.prev_action = None
        self.prev_Qvalue = None
        self.bStart_learning = False

        self.ti = 0
        self.w = 0
        self.best_unit = None
        self.DQN_q_values = None
        self.SOM_q_values = None

        self.eta_counter = 0
        self.trial_num = 0

        self.test_actions = []
        self.test_weights = []
        self.test_memories = []
        self.test_observations = []
        self.test_values = []
        self.test_results = []
        self.test_rewards = []
        self.test_DQN_q_values = []
        self.test_SOM_q_values = []

        self.bLoad_exp = agent_params['bLoad_Exp']
        self.bShuffle_Exp = agent_params['bShuffle_Exp']
        self.exp_thresh = agent_params['exp_thresh']
        self.chosen_units = np.array([])

        self.bSOM = agent_params['bSOM']

        self.lastBufferPos = 0
        self.cuda = th.cuda.is_available()

        self.stepCount = 0
        self.trainStepCount = 0





    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        # Get the most recent entry index
        endPos = self.replay_buffer.pos
        if self.replay_buffer.pos < self.lastBufferPos:
            endPos = self.replay_buffer.pos + self.replay_buffer.buffer_size

        for bufferPos in range(max(0, self.lastBufferPos), endPos): # - batch_size
            self.trainStepCount += 1
            recent_index = bufferPos % self.replay_buffer.buffer_size
            # Retrieve the most recent entry
            recent_state = self.replay_buffer.observations[recent_index]
            recent_action = self.replay_buffer.actions[recent_index]
            recent_reward = self.replay_buffer.rewards[recent_index]
            recent_next_state = self.replay_buffer.next_observations[recent_index]
            recent_done = self.replay_buffer.dones[recent_index]
            # print(f"Training: {recent_state}, {recent_action}, {recent_reward}, {recent_next_state}, {recent_done}")
            # replay_data = self.replay_buffer.sample(1, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                if self.cuda:
                    next_q_values = self.q_net_target(th.tensor(recent_next_state).cuda()) # to do
                else:
                    next_q_values = self.q_net_target(th.tensor(recent_next_state))    
                
                # Alter them based on the SOM
                next_q_values = self.GetQValues(recent_next_state, next_q_values.cpu().numpy())


                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = th.tensor(next_q_values).max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = th.tensor(recent_reward) + (1 - th.tensor(recent_done)) * self.gamma * next_q_values



            # Get current Q-values estimates
            if self.cuda:
                current_q_values = self.q_net(th.tensor(recent_state).cuda())
            else:
                current_q_values = self.q_net(th.tensor(recent_state))

            # Retrieve the q-values for the actions from the replay buffer
            if self.cuda:
                current_q_values = th.gather(current_q_values, dim=1, index=th.tensor(recent_action).cuda())
            else:
                current_q_values = th.gather(current_q_values, dim=1, index=th.tensor(recent_action))

            # Compute Huber loss (less sensitive to outliers)
            if self.cuda:
                loss = F.smooth_l1_loss(current_q_values, target_q_values.cuda())
            else:
                loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()



            
            # Update the SOM with the result (this will also update the q graph)
            self.UpdateSOM(recent_state, recent_action, target_q_values.cpu().numpy())

            # Save interim version of model if requested in the progresssModel array of step counts
            if self.progressModel is not None:
                if (self.trainStepCount) in self.progressModel:
                    self.save(f"{self.directory}/CTDL_ProgressModel_{self.trainStepCount}")
                    

        self.lastBufferPos = self.replay_buffer.pos

        # Increase update counter
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def UpdateSOM(self, observation, action, target):
        if self.cuda:
            delta = np.exp(np.abs(target -
                            np.squeeze(self.q_net(
                                th.tensor(np.expand_dims(observation, axis=0)).cuda()))[action].detach().cpu().numpy()) / self.TD_decay) - 1
        else:
            delta = np.exp(np.abs(target -
                            np.squeeze(self.q_net(
                                th.tensor(np.expand_dims(observation, axis=0))).detach().cpu().numpy())[action]) / self.TD_decay) - 1


        delta = np.clip(delta, 0, 1)

        prev_best_unit = self.SOM.GetOutput(observation)
        self.SOM.Update(observation, prev_best_unit, delta, self.update_mask)

        w = self.GetWeighting(prev_best_unit, observation)
        self.QValues[prev_best_unit, action] += self.Q_alpha * w * (
            target - self.QValues[prev_best_unit, action]) * self.update_mask[prev_best_unit]

        self.Replay()
        self.SOM.RecordLocationCounts()


    def Replay(self):
        losses = []

        # Randomly sample from the SOM for replay
        units = np.random.randint(0, self.SOM.SOM_layer.num_units, 32)
        actions = np.random.randint(0, 4, 32) #.astype(np.int64)

        states = self.SOM.SOM_layer.units['w'][units, :]
        target_q_values = self.QValues[units, actions]

        # Get current Q-values estimates
        if self.cuda:
            current_q_values = self.q_net(th.tensor(states).cuda())
        else:
            current_q_values = self.q_net(th.tensor(states))

        # Retrieve the q-values for the actions from the replay buffer
        indexTensor = th.tensor(actions).long()
        if self.cuda:
            current_q_values = th.gather(current_q_values, dim=1, index=indexTensor.unsqueeze(1).cuda())
        else:
            current_q_values = th.gather(current_q_values, dim=1, index=indexTensor.unsqueeze(1))

        # Compute Huber loss (less sensitive to outliers)
        if self.cuda:
            loss = F.smooth_l1_loss(current_q_values, th.tensor(target_q_values).unsqueeze(1).cuda())
        else:
            loss = F.smooth_l1_loss(current_q_values, th.tensor(target_q_values).unsqueeze(1))
        losses.append(loss.item())

        # Optimize the policy
        self.policy.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self.logger.record("train/loss", np.mean(losses))


    def GetWeighting(self, best_unit, state):

        diff = np.sum(np.square(self.SOM.SOM_layer.units['w'][best_unit, :] - state))
        w = np.exp(-diff / self.weighting_decay)

        return w



    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)

        return action, state


    def learn(
        self: SelfCTDL,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "CTDL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        progress_model = None
    ) -> SelfCTDL:
        self.progressModel = progress_model
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def GetQValues(self, state, q_graph_values):

        best_unit = self.SOM.GetOutput(state)
        som_action_values = self.QValues[best_unit, :]
        w = self.GetWeighting(best_unit, state)
        q_values = (w * som_action_values) + ((1 - w) * q_graph_values)
        self.w = w
        self.best_unit = best_unit
        self.DQN_q_values = q_graph_values
        self.SOM_q_values = som_action_values

        return q_values

    def _on_step(self) -> None:
        super()._on_step()
        self.stepCount += 1
    


    def save(self, path):
        # Call the parent save method to save the model
        super().save(path)
        
        self.saveSOMBin(path + '_som.pkl')
        # with open(custom_info_path, 'w') as f:
        #     json.dump(self.custom_info, f)


    def saveSOM(self, custom_info_path=""):

        if custom_info_path == "":
            custom_info_path = self.directory + '/som_contents.pkl'

        actions_dict = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

        action_inds = []

        for i in range(self.QValues.shape[0]):
            if np.all(self.QValues[i, :] == 0):
                action_inds.append(np.random.randint(4))
            else:
                action_inds.append(np.argmax(self.QValues[i, :]))

        som_actions = np.array([actions_dict[a] for a in action_inds])

        som_contents = {
            "xy": self.SOM.SOM_layer.units["xy"],
            "w": self.SOM.SOM_layer.units["w"],
            "values": self.QValues,
            "actions": som_actions,
        }

        with open(custom_info_path, 'wb') as f:
            pickle.dump(som_contents, f, pickle.HIGHEST_PROTOCOL)

        return

    def saveSOMBin(self, custom_info_path=""):

        if custom_info_path == "":
            custom_info_path = self.directory + '/som_contents_bin.pkl'

        som_contents = {
            "xy": self.SOM.SOM_layer.units["xy"],
            "w": self.SOM.SOM_layer.units["w"],
            "values": self.QValues,
            "etas":self.SOM.SOM_layer.units["etas"],
            "errors":self.SOM.SOM_layer.units["errors"],
        }

        with open(custom_info_path, 'wb') as f:
            pickle.dump(som_contents, f, pickle.HIGHEST_PROTOCOL)

        return

    def loadSOMBin(self, custom_info_path=""):

        if custom_info_path == "":
            custom_info_path = self.directory + '/som_contents_bin.pkl'

        with open(custom_info_path, 'rb') as f:
            som_contents = pickle.load(f)

        self.SOM.SOM_layer.units["xy"] = som_contents["xy"]
        self.SOM.SOM_layer.units["w"] = som_contents["w"]
        self.QValues = som_contents["values"]
        self.SOM.SOM_layer.units["etas"] = som_contents["etas"]
        self.SOM.SOM_layer.units["errors"] = som_contents["errors"]

        return


    def load(cls, path, env=None, custom_objects=None, **kwargs):
        # Load the model using the parent load method
        model = super(CTDL, cls).load(path, env=env, custom_objects=custom_objects, **kwargs)
        model.policy.ctdl = model
        model.reset_params()
        
        model.loadSOMBin(path + '_som.pkl')
        return model
    
    def RecordTestResults(self, maze, trial, test_trial):

        self.test_rewards = np.array(self.test_rewards)
        self.test_weights = np.array(self.test_weights)
        self.test_memories = np.array(self.test_memories)
        self.test_values = np.array(self.test_values)
        self.test_DQN_q_values = np.array(self.test_DQN_q_values)
        self.test_SOM_q_values = np.array(self.test_SOM_q_values)

        actions_dict = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.test_actions = np.array([actions_dict[a] for a in np.array(self.test_actions)])

        results = {'rewards': self.test_rewards,
                   'actions': self.test_actions,
                   'weights': self.test_weights,
                   'memories': self.test_memories,
                   'observations': self.test_observations,
                   'values': self.test_values,
                   'DQN_q': self.test_DQN_q_values}

        self.test_results.append(results)

        actions, memories, values, weights, observations = ExtractExplanation(results, self.exp_thresh, self.bShuffle_Exp)
        PlotExplanation(maze.unwrapped.GetMaze(), memories, actions, weights, observations,
                        self.directory + '/LearningTrial_' + str(trial) + '_TestTrial_' + str(test_trial))

        max_SOM_qvalues = [np.max(arr) for arr in self.QValues]
        # Find the highest and lowest values from the list of max values
        highest_value = max(max_SOM_qvalues)
        lowest_value = min(max_SOM_qvalues)

        minQ, maxQ = self.PlotDQNContents(fileId='_' + str(trial), minVal=lowest_value, maxVal=highest_value)
        ShowDQN(maze.unwrapped.GetMaze(), self.test_actions, self.test_DQN_q_values, save_name=self.directory + '/DQNExplanation_' + str(trial) + '_TestTrial_' + str(test_trial), minVal=minQ, maxVal=maxQ)
        self.PlotSOMContents(fileId='_' + str(trial), minVal=minQ, maxVal=maxQ)
        with open(self.directory + '/Explanation_' + str(trial) + '_TestTrial_' + str(test_trial) + '.pkl', 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        with open(self.directory + '/Explanation_' + str(trial) + '_TestTrial_' + str(test_trial) + '.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results.keys())
            writer.writeheader()
            writer.writerows([dict(zip(results, t)) for t in zip(*results.values())])
        extractedExplanations = {
                   'actions': actions,
                   'weights': weights,
                   'memories': memories,
                   'values': values}
        with open(self.directory + '/Extracted_Explanation_' + str(trial) + '_TestTrial_' + str(test_trial) + '.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=extractedExplanations.keys())
            writer.writeheader()
            writer.writerows([dict(zip(extractedExplanations, t)) for t in zip(*extractedExplanations.values())])


        self.test_rewards = []
        self.test_actions = []
        self.test_weights = []
        self.test_memories = []
        self.test_observations = []
        self.test_values = []
        self.test_DQN_q_values = []

        return len(actions)




    def PlotSOMContents(self, fileId="", minVal=None, maxVal=None):
        actions_dict = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        memories = []
        actions = []
        weights = []
        for i in range(self.SOM.SOM_layer.num_units):
            max_value = np.max(self.QValues[i, :])
            # if Q Values have been initialised i.e. not zero
            if not np.all(self.QValues[i, :] == 0):
                action = np.argmax(self.QValues[i, :])
                actions.append(actions_dict[action])
                weights.append(max_value)
                memories.append(self.SOM.SOM_layer.units['w'][i])
        if len(actions) > 0:
            PlotAllExplanations(self.gymEnvironment.unwrapped.GetMaze(), memories, actions, weights, self.directory + 'SOMContents' + fileId, "Q-Value", minVal=minVal, maxVal=maxVal)

    def PlotDQNContents(self, fileId="", minVal=None, maxVal=None):
        actions_dict = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        memories = []
        actions = []
        weights = []
        maze = self.gymEnvironment.unwrapped.GetMaze()

        for row in range(maze.shape[0]):
            for col in range(maze.shape[1]):
                observation = th.tensor([row, col])
                observation_tensor = observation.float().unsqueeze(0)
                if self.cuda:
                    q_graph_values = self.q_net(observation_tensor.cuda()).detach().cpu().numpy()
                else:
                    q_graph_values = self.q_net(observation_tensor).detach().cpu().numpy()
                vals = q_graph_values

                max_value = np.max(vals)
                action = np.argmax(vals)
                actions.append(actions_dict[action])
                weights.append(max_value)
                memories.append([row, col])
        minQ, maxQ = PlotAllExplanations(self.gymEnvironment.unwrapped.GetMaze(), memories, actions, weights, self.directory + 'DQNContents' + fileId, "Q-Value", minVal=minVal, maxVal=maxVal)
        return minQ, maxQ

    def SaveTestResults(self):

        with open(self.directory + '/explanation.pkl', 'wb') as f:
            pickle.dump(self.test_results, f, pickle.HIGHEST_PROTOCOL)

        return
    

    def PlotValueFunction(self):

        self.plot_num += 1
        up_value_function = np.zeros((self.maze_height, self.maze_width))
        down_value_function = np.zeros((self.maze_height, self.maze_width))
        left_value_function = np.zeros((self.maze_height, self.maze_width))
        right_value_function = np.zeros((self.maze_height, self.maze_width))

        for row in range(self.maze_height):
            for col in range(self.maze_width):
                #
                # get q values
                observation = th.tensor([row, col])
                observation_tensor = observation.float().unsqueeze(0)
                if self.cuda:
                    q_graph_values = self.q_net(observation_tensor.cuda()).detach().cpu().numpy()
                else:
                    q_graph_values = self.q_net(observation_tensor).detach().cpu().numpy()

                #q_graph_values = np.squeeze(np.array(self.q_graph.GetActionValues(np.array([[row, col]]))))

                if(self.bSOM):
                    vals = self.GetQValues([row, col], q_graph_values)
                else:
                    vals = q_graph_values

                up_value_function[row, col] = vals[0][0]
                down_value_function[row, col] = vals[0][1]
                left_value_function[row, col] = vals[0][2]
                right_value_function[row, col] = vals[0][3]

        fig, axes = plt.subplots(2, 2)

        vmin = np.amin([up_value_function, down_value_function, left_value_function, right_value_function])
        vmax = np.amax([up_value_function, down_value_function, left_value_function, right_value_function])

        im = axes[0, 0].imshow(up_value_function, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Up Value Function')

        im = axes[0, 1].imshow(down_value_function, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Down Value Function')

        im = axes[1, 0].imshow(left_value_function, cmap='hot', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Left Value Function')

        im = axes[1, 1].imshow(right_value_function, cmap='hot', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title('Right Value Function')

        for axis in axes.ravel():
            axis.set_xticklabels([])
            axis.set_xticks([])
            axis.set_yticklabels([])
            axis.set_yticks([])

        fig.colorbar(im, ax=axes.ravel().tolist())
        plt.savefig(self.directory + 'ValueFunction' + str(self.plot_num) + '.png')
        plt.close()

        if(self.bSOM):
            self.SOM.PlotResults(self.plot_num)
            self.PlotEtaHeatMap()


        return

    def PlotEtaHeatMap(self):
        # Plot eta heat map
        self.eta_counter += 1
        eta_map = np.zeros((self.maze_height, self.maze_width))
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                best_unit = self.SOM.GetOutput([y, x])
                eta_map[y, x] = self.GetWeighting(best_unit, [y, x])
        plt.figure()
        plt.imshow(eta_map, cmap='hot')
        plt.colorbar()
        plt.savefig(self.directory + '/EtaHeatMap' + str(self.eta_counter) + '.png')
        plt.close()


        return

    def PlotResults(self):

        plt.figure()
        plt.plot(self.gymEnvironment.results['rewards'])
        found_goal = np.where(np.array(self.gymEnvironment.results['rewards']) > 0)
        if(found_goal):
            for loc in found_goal[0]:
                plt.axvline(x=loc, color='g')
        plt.savefig(self.directory + 'AgentTrialRewards.pdf')
        plt.close()

        with open(self.directory + 'Results.pkl', 'wb') as handle:
            self.gymEnvironment.results['Count']=len(self.gymEnvironment.results['lengths'])
            self.gymEnvironment.results['Length mean/sd']=[statistics.mean(self.gymEnvironment.results['lengths']), statistics.pstdev(self.gymEnvironment.results['lengths'])]
            self.gymEnvironment.results['Reward mean/sd']=[statistics.mean(self.gymEnvironment.results['rewards']), statistics.pstdev(self.gymEnvironment.results['rewards'])]

            pickle.dump(self.gymEnvironment.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        np.save(self.directory + 'LocationCounts', self.SOM.location_counts)

        return

    def reset_params(self):
        self.explanation_length = agent_params['exp_length']

        self.weighting_decay = agent_params['w_decay']
        self.TD_decay = agent_params['TD_decay']
        self.num_epsilon_trials = agent_params['e_trials']
        self.bLoad_exp = agent_params['bLoad_Exp']
        self.bShuffle_Exp = agent_params['bShuffle_Exp']
        self.exp_thresh = agent_params['exp_thresh']
        self.bSOM = agent_params['bSOM']


    def load_explanation(self, explanation_file):
        with open(explanation_file, 'rb') as handle:
            explanations = pickle.load(handle)

        actions, memories, values, weights, observations = ExtractExplanation(explanations, self.exp_thresh, self.bShuffle_Exp)

        chosen_units = np.random.choice(self.SOM.SOM_layer.num_units, actions.shape[0], replace=False)

        for i, unit in enumerate(chosen_units.tolist()):
            self.SOM.SOM_layer.units['w'][unit, :] = memories[i, :]
            self.QValues[unit, :] = values[i, :]

        self.chosen_units = chosen_units
        actions_dict = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.explanation_actions = [actions_dict[a] for a in actions.tolist()]
        self.update_mask[self.chosen_units] = 0


        return
