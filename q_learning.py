# # Q-Learning
# In this project I will be implementing the Gridworld and Frozen Lake reinforcement learning environments and train 
# an agent to navigate them using Q-learning.

# We will start by importing our required libraries. Numpy is used mainly for random number generation 
# while pygame is used to display our maze and training process.
import numpy as np
import pygame

# Prevent showing Q-values in scientific format, for better readability
np.set_printoptions(suppress=True, precision=4)


####################################################################################################################
# ## Q-Learning Implementation
# Below, we implement two separate classes; one for the maze and one for the agent. The maze class stores all 
# information relevant to the layout of the maze, such as column/row count, while the agent class implements 
# the required functions for learning to navigate the maze, such as the Q-function.
class QMaze():
    def __init__(self, maze_base:str, goal_points:float, small_points:float, step_cost:float):
        """Initialize the maze using the given maze layout"""

        self.maze_base = maze_base

        self.step_cost = step_cost
        self.field_mapping = {"S": self.step_cost, "W": 0, "G": goal_points, "T": -goal_points, "o": self.step_cost, "+": small_points, "-": -small_points}
        self.action_mapping = {0: "left", 1: "down", 2: "right", 3: "up"}

        self.maze = []
        self.column_count = None
        self.row_count = None
        self.start_field = None

        self.construct_maze()

    def construct_maze(self):
        """Get the maze fields as a list of characters and store information about the maze layout"""

        rows = self.maze_base.split("\n")
        fields_str = "".join(rows)

        for i in range(len(fields_str)):
            self.maze.append(fields_str[i])

            if fields_str[i] == "S": 
                self.start_field = i

        self.column_count = len(rows[0])
        self.row_count = len(rows)


class QAgent():
    def __init__(self, maze:QMaze, epsilon:float, min_epsilon:float, slip_chance:float, learning_rate:float, 
                 discount_rate:float, episodes:int):
        """Initialize an instance of the agent with the given hyperparameters"""

        self.maze = maze
        self.used_boni = []

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.slip_chance = slip_chance
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        self.episodes = episodes
        self.cur_episode = 0

        self.cur_state = self.maze.start_field

        self.q_table = None
        self.construct_q_table(method="zeros")

    def construct_q_table(self, method:str="zeros"):
        """Initializes the Q-table with 4 values for each field corresponding to left, down, right, up"""

        if method == "zeros":
            self.q_table = np.zeros((self.maze.row_count*self.maze.column_count, 4))
        elif method=="random":
            rng = np.random.default_rng(1) # rng with seed 1 instead of random.seed to not overwrite global random seed
            self.q_table = rng.uniform(-0.3, 0.3, (self.maze.row_count*self.maze.column_count, 4))

    def update_q_table(self):
        """Updates the Q-value of the agent for a chosen action on the current field using Q-function"""

        # Choose action to perform from current field
        intended_action, slip_action = self.choose_action()

        # Get the current Q-value for that action on that field
        old_q_val = self.q_table[self.cur_state, intended_action]

        # Get the new field the agent would land on after performing the chosen action
        new_state = self.get_new_state(slip_action)

        # Get the immediate reward of landing on that field
        immediate_reward = self.maze.field_mapping[self.maze.maze[new_state]]
        if new_state in self.used_boni: # don't give bonus rewards/punishments more than once per episode
            immediate_reward = self.maze.step_cost

        # Get the maximum Q-value of the actions that can be performed in the new field
        max_future_q_val = max(self.q_table[new_state])

        # Calculate the new Q-value for the chosen action for the current field using the Q-function
        new_q_val = old_q_val + self.learning_rate * (immediate_reward + self.discount_rate * max_future_q_val - old_q_val)

        # Update the Q-table with the value
        self.q_table[self.cur_state, intended_action] = new_q_val

        # Update the state to perform the action
        self.update_state(new_state)

    def action_possible(self, action:int) -> bool:
        """Determine if a given action is possible from current state"""

        new_state = self.get_new_state(action)

        # Prevent moving out of bounds of the maze grid
        if new_state < 0 or new_state >= len(self.maze.maze):
            return False
        # Prevent moving from end of one row to start of next or vice versa by moving right/left
        if (not self.cur_state == 0) and (
            (self.cur_state % self.maze.column_count == 0 and action == 0) or # start of row
            (self.cur_state % self.maze.column_count == self.maze.column_count - 1 and action == 2)): # end of row
            return False
        # Prevent moving onto walls
        if self.maze.maze[new_state] == "W":
            return False

        return True

    def get_possible_actions(self) -> np.ndarray:
        """Get a list of possible actions from current state"""

        all_actions = np.arange(4)
        pos_actions = []

        for act in all_actions:
            if self.action_possible(act): 
                pos_actions.append(act)

        return np.array(pos_actions)

    def get_new_state(self, action:int) -> int:
        """Get the state (i.e. field) the agent will be in after performing a given action"""

        if action == 0: # Move left
            new_state = self.cur_state - 1
        elif action == 1: # Move down
            new_state = self.cur_state + self.maze.column_count
        elif action == 2: # Move right
            new_state = self.cur_state + 1
        elif action == 3: # Move up
            new_state = self.cur_state - self.maze.column_count
        else:
            raise KeyError(f"Invalid action: {action}")

        return new_state

    def update_state(self, new_state:int):
        """Update the current state (i.e. the agents position) based on the new state"""

        # Update state
        self.cur_state = new_state

        next_maze_field = self.maze.maze[new_state]

        # Keep track of collected boni until the end of the episode in order to prevent using bonus multiple times
        if next_maze_field == "+" or next_maze_field == "-":
            self.used_boni.append(new_state)

        # If the agent lands on a goal/trap field, reset its position to the start field and forget about the collected boni
        if next_maze_field == "G" or next_maze_field == "T":
            self.cur_state = self.maze.start_field
            self.cur_episode += 1
            self.used_boni = []

    def choose_action(self) -> int:
        """Choose action using epsilon-greedy policy and incorporate slip chance"""

        pos_actions = self.get_possible_actions()
        intended_action = None
        slip_action = None

        # Decay of exploration rate over course of training to shift from exploration to exploitation behaviour later
        epsilon = max((1 - self.cur_episode / self.episodes) * self.epsilon, self.min_epsilon)

        if np.random.random() < 1-epsilon: 
            # Choose best action based on highest Q-value

            cur_q_vals = self.q_table[self.cur_state] # get the Q-values for all actions of the current field
            pos_q_vals = cur_q_vals[pos_actions] # filter out the actions that can't be performed

            # Get list of actions that have the highest Q-value (as there might be multiple)
            max_q_val = np.max(pos_q_vals)
            best_actions = np.where(cur_q_vals == max_q_val)[0]
            best_actions = [act for act in best_actions if act in pos_actions]

            # Randomly pick one of the best actions or the one best action (if there is just one)
            intended_action = np.random.choice(best_actions)
        else: 
            # Choose random action
            intended_action = pos_actions[np.random.randint(0, len(pos_actions))]

        slip_action = intended_action

        # There is a random chance for the agent to "slip" and perform an action different from the one he intended to perform
        # -> If slip_chance is set to zero during initialization of the agent (see cells below), we get the Gridworld
        #    environment, otherwise the Frozen Lake environment
        if np.random.random() < self.slip_chance:
            slip_action = pos_actions[np.random.randint(0, len(pos_actions))]

        return intended_action, slip_action


####################################################################################################################
# ## Setup
# Below, you can adjust the layout of the maze by changing the "maze_base" variable. The field mappings are as follows: 
# - "o" - empty field
# - "W" - wall/obstacle
# - "S" - start field
# - "G" - goal field
# - "T" - trap/hole in ice
# - "+" -  small reward
# - "-" - small punishment

# Chose the maze layout here 
# Note: all rows should be the same size and all columns should be the same size, but row and column counts 
# do not have to match. Don't use any characters besides the ones above (also don't use whitespaces)
maze_base = """
oo+ooo
oooooo
SoTToo
WooooG
"""[1:-1]

# Here you can assign the rewards/punishments for landing on certain fields
# +-goal_points: G/T  |  +-small_points: +/-  |  step_cost: o (note that step cost should be negative)
maze = QMaze(maze_base, goal_points=1, small_points=0.2, step_cost=-0.3)

# Here you can change the hyperparameters for the training of the agent
# epsilon: Probability of the agent to perform a random action instead of the best one he knows
# min_epsilon: Minimum value epsilon should be able to take (important because epsilon decay is implemented)
# slip_chance: Probability of the agent to "slip" and perform a random action instead of the one he intended to perform
#              -> If set to 0, then the environment becomes the Gridworld environment
#              -> If set to a value between 0 and 1, then the environment becomes the Frozen Lake environment
# learning_rate: Governs how much the agent should learn with each update
# discount_rate: Governs how important future rewards are during learning
# episodes: How many full traversals the agent should be trained for
agent = QAgent(maze, epsilon=0.15, min_epsilon=0.05, slip_chance=0.2, learning_rate=0.05, 
               discount_rate=0.9, episodes=100)


####################################################################################################################
# ### Visualization
# Here two ways of visualizing the agent traversing the maze are implemented, one using pygame and one using 
# just the terminal.

# #### Pygame
# Initialize pygame window
pygame.init()
CELL_SIZE = 80
WINDOW_WIDTH = maze.column_count * CELL_SIZE
WINDOW_HEIGHT = maze.row_count * CELL_SIZE
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Q-Learning Maze")

# Color mapping for pygame maze display
colors = {
    "S": (0, 120, 255), # Start
    "W": (60, 60, 60), # Wall
    "G": (0, 200, 0), # Goal
    "T": (200, 0, 0), # Trap
    "o": (220, 220, 220), # Empty
    "+": (195, 217, 50), # + Bonus 
    "-": (217, 100, 50) # - Bonus
}

def display_maze(fps:int):
    """Display maze using pygame"""

    # Stop running if pygame window is closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
            pygame.quit()
            exit()

    # Draw maze
    for idx, tile in enumerate(maze.maze):
        row = idx // maze.column_count
        col = idx % maze.column_count

        rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)

        color = colors.get(tile, (200, 200, 200))  # Default is light gray
        pygame.draw.rect(screen, color, rect)

        # Draw agent
        if idx == agent.cur_state:
            pygame.draw.circle(screen, (0, 0, 0), rect.center, CELL_SIZE // 3)

        pygame.draw.rect(screen, (50, 50, 50), rect, 1)  # Grid lines

    pygame.display.flip()

    # Set FPS (higher leads to faster training)
    clock.tick(fps)

# #### Terminal
from os import system

def print_maze():
    """Print maze in terminal"""

    system("cls")
    cur_maze = "".join(maze.maze)[:agent.cur_state] + "A" + "".join(maze.maze)[agent.cur_state+1:]
    j = maze.column_count
    while j < len(maze.maze):
        cur_maze = cur_maze[:j] + "\n" + cur_maze[j:]
        j += maze.column_count+1
    print(cur_maze)


####################################################################################################################
# ## Training
# Finally, the training loop is started. The agent will continue updating it's Q-values until it has reached a 
# terminal state (a trap or the goal) as often as the number of episodes specified during setup.

# You can stop training at any time by pressing any key on the keyboard.

# Training - stopped by pressing any key
while agent.cur_episode < agent.episodes:
    # Chooses an action for current state, adjusts Q-values, then performs that action
    agent.update_q_table()

    # Display maze using pygame
    display_maze(fps=60)
    
    # Display maze using terminal
    # print_maze()

pygame.quit()


# Let's also print out the Q-table at the end of training to see how the Q-values changed 
print("Maze Layout: ")
print(maze_base, end="\n\n")

print("Final Q-table |||     Actions:    Left | Down | Right | Up")
q_table = agent.q_table
for i in range(maze.row_count):
    for j in range(i * maze.column_count, (i+1) * maze.column_count):
        print(f"Q-Value at row {i+1} for column {j % maze.column_count + 1}: {q_table[j]}")

