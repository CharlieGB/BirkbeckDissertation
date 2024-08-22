from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN
from SOM import *
from Parameters import *






class TriggerEvents(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.SOM = SOM("SOM", maze_params['width'], maze_params['height'], 2,  agent_params['SOM_size'],
                       agent_params['SOM_alpha'], agent_params['SOM_sigma'],
                       agent_params['SOM_sigma_const'])
        self.Q_alpha = agent_params['Q_alpha']
        self.QValues = np.zeros((agent_params['SOM_size'] * agent_params['SOM_size'], 4))
        self.update_mask = np.ones((self.SOM.SOM_layer.num_units))

        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        
        # Ensure the replay buffer is populated
        if False and replay_buffer.size() > 0:
            # Access the replay buffer from the model
            replay_buffer = self.model.replay_buffer
            # Sample a batch from the replay buffer
            observations, actions, next_observations, dones, rewards = replay_buffer.sample(1)

            # Get Q-values from the policy
            q_values = self.model.q_net(observations).detach().cpu().numpy()
            #print(f"Q-values: {q_values}")

            lastObservation = observations.detach().cpu().numpy()
            nextObservation = next_observations.detach().cpu().numpy()
            action = actions.detach().cpu().numpy()[0][0]
            reward = rewards.detach().cpu().numpy()[0][0]
            done = dones.detach().cpu().numpy()[0][0]
            self.update_mask = np.ones((self.SOM.SOM_layer.num_units))
            self.Q_alpha = agent_params['Q_alpha']
            prev_best_unit = self.SOM.GetOutput(lastObservation)
            self.SOM.Update(lastObservation, prev_best_unit, self.Q_alpha, self.update_mask)
            print(f"From the model ------------------------------------ last: {lastObservation}, next {nextObservation}, action {action}, reward {reward}, done {done}")

            # Log the Q-values
            #print(f"Q-values: {q_values}")
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass