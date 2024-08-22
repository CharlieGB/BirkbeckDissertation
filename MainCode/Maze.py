import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import pygame
from os import path
import sys

from Enums import MazeType 
from Parameters import maze_params


# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='maze-v0',
    entry_point='Maze:MazeEnv',
    max_episode_steps=1000,
)

class AgentAction(Enum):
    UP=0
    DOWN=1
    LEFT=2
    RIGHT=3

EMPTY_VAL=0
PENALTY_VAL=-1
REWARD_VAL=1
START_VAL=2
BORDER_VAL = -2


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}



    def __init__(self, render_mode=None, render_fps=4):
        super(MazeEnv, self).__init__()


        np.random.seed(maze_params['random_seed'])

        self.type = maze_params['type']
        if self.type == MazeType.human:
            self.maze_num = maze_params['maze_num']
        self.width = maze_params['width']
        self.height = maze_params['height']
        self.num_hazards = maze_params['num_hazards']
        self.num_rewards = maze_params['num_rewards']
        self.max_steps = maze_params['max_steps']
        self.directory = None
        self.render_mode = render_mode
        self.reward = 0.0
        self.fps = render_fps
        self.window_surface = None

        self.trial_reward = 0.0
        self.trial_length = 0
        self.recordTrials = True


        self.results = {'rewards': [], 'lengths': []}
        self.mazeCopy = None


        self.ConstructMaze()
        self.reset()
        print(self.maze)


        self.action_space = spaces.Discrete(len(AgentAction))

        self.observation_space = spaces.Box(
            low=0,
            #high=np.array([self.height - 1, self.width - 1, self.height - 1, self.width - 1]),
            high=np.array([self.height - 1, self.width - 1]),
            shape=(2,),
            dtype=np.int64
        )
        if self.render_mode == 'human':
            self._init_pygame()


    def _init_pygame(self):
        pygame.init() # initialize pygame
        pygame.display.init() # Initialize the display module

        # Game clock
        self.clock = pygame.time.Clock()

        # Default font
        self.action_font = pygame.font.SysFont("Calibre",30)
        self.action_info_height = self.action_font.get_height()

        # For rendering
        self.cell_height = 64
        self.cell_width = 64
        self.cell_size = (self.cell_width, self.cell_height)        

        # Define game window size (width, height)
        self.window_size = (self.cell_width * self.width, self.cell_height * self.height + self.action_info_height)

        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size) 

        # Load & resize sprites
        file_name = path.join(path.dirname(__file__), "sprites/bot_blue.png")
        img = pygame.image.load(file_name)
        self.robot_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/floor.png")
        img = pygame.image.load(file_name)
        self.floor_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/prize.png")
        img = pygame.image.load(file_name)
        self.prize_img = pygame.transform.scale(img, self.cell_size) 

        file_name = path.join(path.dirname(__file__), "sprites/barrier.png")
        img = pygame.image.load(file_name)
        self.barrier_img = pygame.transform.scale(img, self.cell_size) 

        file_name = path.join(path.dirname(__file__), "sprites/penalty.png")
        img = pygame.image.load(file_name)
        self.penalty_img = pygame.transform.scale(img, self.cell_size) 



    def ConstructMaze(self):

        self.maze = np.zeros((self.height * self.width))

        if(self.type == MazeType.random):
            self.ConstructRandomMaze()
        elif(self.type == MazeType.direct):
            self.ConstructDirectMaze()
        elif (self.type == MazeType.obstacle1):
            self.ConstructFirstObstacleMaze()
        elif (self.type == MazeType.obstacle2):
            self.ConstructSecondObstacleMaze()
        elif (self.type == MazeType.human):
            self.ConstructHumanMaze()

        self.mazeCopy = np.copy(self.maze)

        self.start = np.squeeze(np.array(np.where(self.maze == START_VAL)))
        self.maze[self.start[0], self.start[1]] = EMPTY_VAL
        self.rewardCell =  np.squeeze(np.array(np.where(self.maze == REWARD_VAL)))

        return

    def ConstructHumanMaze(self):
        file_name = path.join(path.dirname(__file__), "mazes/m" + str(self.maze_num) + '.txt')
        with open(file_name, 'r') as textFile:
            lines = [line.strip() for line in textFile.readlines()] # remove leading/trailing white space
        maze = [list(s) for s in lines]
        mapping = {'o': PENALTY_VAL, 'e': REWARD_VAL, 'a': START_VAL, \
                   's': EMPTY_VAL, 'b': BORDER_VAL}

        for row in range(len(maze)):
            for col in range(len(maze[0])):
                maze[row][col] = mapping[maze[row][col]]

        self.maze = np.array(maze).astype(np.int64)

    def SaveMaze(self, directory):
        self.directory = directory

        plt.figure()
        plt.imshow(self.mazeCopy)
        plt.savefig(self.directory + 'Maze.pdf')
        plt.close()

        np.save(self.directory + 'Maze', self.mazeCopy)


    def ConstructRandomMaze(self):

        inds = np.random.choice(np.arange(self.height * self.width), self.num_hazards + self.num_rewards + 1,
                                replace=False)
        self.maze[inds[:self.num_hazards]] = -1
        self.maze[inds[self.num_hazards:self.num_hazards + self.num_rewards]] = 1
        self.maze[inds[-1]] = 2
        self.maze = self.maze.reshape((self.height, self.width))

        return

    def ConstructDirectMaze(self):

        self.maze = self.maze.reshape((self.height, self.width))
        self.maze[0, int(self.width / 2)] = 1
        self.maze[-1, int(self.width / 2)] = 2
        self.maze[:, :int(self.width / 2) - 2] = -1
        self.maze[:, int(self.width / 2) + 3:] = -1

        return

    def ConstructFirstObstacleMaze(self):

        self.ConstructDirectMaze()
        self.maze[int(self.height / 2) - 1, int(self.width / 2) - 1: int(self.width / 2) + 2] = -1

        return

    def ConstructSecondObstacleMaze(self):

        self.ConstructDirectMaze()
        self.maze[int(self.height / 3), int(self.width / 2) - 2:int(self.width / 2) + 1] = -1
        self.maze[int(self.height / 3) * 2, int(self.width / 2):int(self.width / 2) + 3] = -1

        return

    def GetMaze(self):

        maze = np.copy(self.maze)
        maze[self.start[0], self.start[1]] = 2

        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.


        self.working_maze = np.copy(self.maze)
        self.state = np.copy(self.start)
        self.reward = 0.0
        self.stepCount = 0
        self.trial_reward = 0.0
        self.trial_length = 0

        #observation = np.concatenate((self.state, self.rewardCell))
        observation = self.state
        info = {}

        # record incomplete last trial
        # if self.trial_length > 0:
        #     self.RecordResults(True, 0, True)

        return observation, info
    
    def render(self):
        self._process_events()

        # clear to white background, otherwise text with varying length will leave behind prior rendered portions
        self.window_surface.fill((255,255,255))

        # Print current state on console
        for r in range(self.height):
            for c in range(self.width):
                
                # Draw floor
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.floor_img, pos)

                if(self.working_maze[r,c] == BORDER_VAL):
                    # Draw barrier
                    self.window_surface.blit(self.barrier_img, pos)

                if(self.working_maze[r,c] == REWARD_VAL):
                    # Draw target
                    self.window_surface.blit(self.prize_img, pos)

                if(self.working_maze[r,c] == PENALTY_VAL):
                    # Draw target
                    self.window_surface.blit(self.penalty_img, pos)

                if(r == self.state[0] and c == self.state[1]):
                    # Draw robot
                    self.window_surface.blit(self.robot_img, pos)
                
        text_img = self.action_font.render(f'Reward: {self.reward:,.2f}', True, (0,0,0), (255,255,255))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)       

        pygame.display.update()
                
        # Limit frames per second
        self.clock.tick(self.fps)  


    
    def step(self, action):
        self.stepCount += 1
        bTrial_over = False

        prevState = self.state.copy()

        cellReward, goalReached = self.UpdateState(AgentAction(action))

        prevVertDistance = abs(prevState[0] - self.rewardCell[0])
        prevHorzDistance = abs(prevState[1] - self.rewardCell[1])
        currentVertDistance = abs(self.state[0] - self.rewardCell[0])
        currentHorzDistance = abs(self.state[1] - self.rewardCell[1])

        if currentVertDistance < prevVertDistance:
            closeReward = 0
        else:
            closeReward = 0

        cellReward += -.05 + closeReward
        self.reward = cellReward
        

        if (goalReached or self.stepCount >= self.max_steps):
            bTrial_over = True



        info = {}
        # print(f"Previous {prevState}, Current {self.state}, Action {action}, Reward {cellReward}")

        if self.render_mode == 'human':
            self.render()
        #observation = np.concatenate((self.state, self.rewardCell))
        observation = self.state

        self.RecordResults(bTrial_over, cellReward)

        return observation, cellReward, bTrial_over, False, info
        
    def isValidCell(self, cell):
        validCell = True
        if self.working_maze[cell[0], cell[1]] == BORDER_VAL \
                or cell[0] < 0 or cell[0] > self.height - 1 \
                or cell[1] < 0 or cell[1] > self.width - 1:
            validCell = False
        return validCell
        

    def UpdateState(self, action:AgentAction):

        next_state = self.state
        if (action == AgentAction.UP):
            if (self.state[0] > 0):
                next_state = self.state + np.array([-1, 0])
        elif (action == AgentAction.DOWN):
            if (self.state[0] < self.height - 1):
                next_state = self.state + np.array([1, 0])
        elif (action == AgentAction.LEFT):
            if (self.state[1] > 0):
                next_state = self.state + np.array([0, -1])
        elif (action == AgentAction.RIGHT):
            if (self.state[1] < self.width - 1):
                next_state = self.state + np.array([0, 1])
        else:
            print("Invalid action", action)
        if self.working_maze[next_state[0], next_state[1]] != BORDER_VAL:
            self.state = next_state
        if self.isValidCell(next_state):
            self.state = next_state

        goalReached = False
        cellReward = self.working_maze[self.state[0], self.state[1]]
        self.reward = cellReward

        if (cellReward > 0):
            self.working_maze[self.state[0], self.state[1]] = 0
            goalReached = True

        return cellReward, goalReached
    
    def _process_events(self):
        # Process user events, key presses
        for event in pygame.event.get():
            # User clicked on X at the top right corner of window
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit()

            if(event.type == pygame.KEYDOWN):
                # User hit escape
                if(event.key == pygame.K_ESCAPE):
                    pygame.display.quit()
                    pygame.quit()
                    sys.exit()
        return
    
    def close(self):
        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()

    def RecordResults(self, bTrial_over, reward, prematurelyEnded=False):

        self.trial_reward += reward
        if not prematurelyEnded:
            self.trial_length += 1
        if (bTrial_over):
            if prematurelyEnded:
                print('Ended prior to completion ', self.state)
            print(f"Trial Reward: {self.trial_reward:.2f}")
            if self.recordTrials and not prematurelyEnded:
                self.results['rewards'].append(round(float(self.trial_reward), 2))
            self.trial_reward = 0

            print(f"Trial Length: {self.trial_length:.0f} \n")
            if self.recordTrials and not prematurelyEnded:
                self.results['lengths'].append(round(float(self.trial_length)))
            self.trial_length = 0

        return


if __name__=="__main__":
    env = gym.make('maze-v0')

    # Use this to check our custom environment
    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    for _ in range(1000):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)

        if(terminated):
            obs = env.reset()[0]

    env.close()
    
