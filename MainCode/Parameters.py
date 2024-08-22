from Enums import MazeType, AgentType


maze_params = {'type': MazeType.human,
               'maze_num': 1,
               'width': 10,
               'height': 10,
               'num_rewards': 1,
               'num_trials': 1000,
               'random_seed': 0,
               'max_steps': 1000,
               'num_repeats': 20,
               'print_freq': 1000,
               'explanation_freq': 200,
               'num_test_trials': 10,
               'num_hazards': 5,
               }

agent_params = {'agent_type': AgentType.CTDL,
                'bSOM': True,
                'bLoad_Exp': False,
                'bShuffle_Exp': False,
                'exp_length': 20,
                'exp_thresh': 0.5,
                'SOM_alpha': 1,             # .01 Learning rate for updating the weights of the SOM
                'SOM_sigma': 1,             # .1 Standard deviation of the SOM neighbourhood function σ
                'SOM_sigma_const': .001,    # .1 Constant for denominator in SOM neighbourhood function
                'Q_alpha': .1,              # .9 Learning rate for updating the Q values of the SOM
                'w_decay': 1,               # 10 Temperature for calculating η
                                                # weighting parameter η ∈ {0, 1}, which is used to calculate a weighted average of the action values from the SOM
                                                # and the DNN. If the best matching unit is close to the current state then a larger weighting will be applied to
                                                #  the Q value produced by the SOM. A free parameter τη acts as a temperature parameter to scale the euclidean 
                                                # distance between βu and st when calculating the weighted average.
                                                # Lower values to lower weight of SOM contribuiton

                'TD_decay': 100,            # 1 Temperature for calculating δ
                                                # the TD error is used to create an exponentially increasing value
                                                # δ ∈ {0, 1}, which scales the standard deviation of the SOM’s
                                                # neighbourhood function and the learning rate of the SOM’s weight
                                                # update rule. Again a temperature parameter τδ is used to scale the
                                                # TD error.
                                                # Bigger values to smaller delta
                'SOM_size': 6,             # 36 Number of units in SOM (ie 6)
                'e_trials': 200,
                }

# From article
# U 36 Number of units in SOM
# τη 10 Temperature for calculating η
# τδ 1 Temperature for calculating δ
# σ .1 Standard deviation of the SOM neighbourhood function σ
# c .1 Constant for denominator in SOM neighbourhood function 
# α .01 Learning rate for updating the weights of the SOM 
# ρ .9 Learning rate for updating the Q values of the SOM