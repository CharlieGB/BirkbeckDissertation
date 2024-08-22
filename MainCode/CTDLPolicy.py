from typing import Any, Dict, List, Optional, Type

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
#from CTDL import CTDL
import numpy as np


class CTDLPolicy(DQNPolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.ctdl = None

    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        #qValues = self.q_net(self.q_net.extract_features(obs, self.q_net.features_extractor))
        #action = qValues.argmax(dim=1).reshape(-1)

        
        q_graph_values = self.q_net(self.q_net.extract_features(obs, self.q_net.features_extractor))
        q_graph_values = q_graph_values.cpu().numpy()
        #q_graph_values = np.squeeze(np.array(self.q_net(th.tensor(np.expand_dims(obs, axis=0))).detach().cpu().numpy()))
        q_values = self.ctdl.GetQValues(obs.cpu().numpy(), q_graph_values)
        action = th.tensor(np.argmax(q_values))


        # if not self.ctdl.ignoreSom:
        #     # adjust predicted action with influence from SOM
        #     state = obs.numpy()
        #     best_unit = self.ctdl.SOM.GetOutput(state)
        #     som_action_values = self.ctdl.QValues[best_unit, :]
        #     w = self.ctdl.GetWeighting(best_unit, state)
        #     q_values = (w * som_action_values) + ((1 - w) * qValues.numpy())
        #     self.ctdl.w = w
        #     self.ctdl.best_unit = best_unit
        #     actionVal = np.argmax(q_values)
        #     action = th.tensor(actionVal)

        return action
    