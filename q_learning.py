import os
import numpy as np
import time
import pygame

np.set_printoptions(suppress=True, precision=3)


class QMaze():
    def __init__(self, maze_base):
        """Initialize all future required variables as well as the maze and q table"""

        self.maze_base = maze_base

        small_points = 0.5
        move_cost = -0.1
        goal_points = 10
        self.field_mapping = {"S": move_cost, "W": "wall", "G": goal_points, "o": move_cost, "+": small_points, "-": -small_points}
        self.action_mapping = {0: "left", 1: "down", 2: "right", 3: "up"}

        self.maze = []
        self.column_count = None
        self.row_count = None
        self.start_field = None

        self.q_table = None

        self.epsilon = 0.15

        self.cur_state = None # int between 0 and number of fields in the maze

        self.construct_maze()
        self.construct_q_table()

    def construct_maze(self):
        """Get the maze fields as a list of characters. Set some variables based on the maze layout"""

        rows = self.maze_base.split("\n")
        fields_str = "".join(rows)

        for i in range(len(fields_str)):
            self.maze.append(fields_str[i])

            if fields_str[i] == "S": 
                self.start_field = i
                self.cur_state = i

        self.column_count = len(rows[0])
        self.row_count = len(rows)

    def construct_q_table(self, method="zeros"):
        """Initializes the q table with 4 values for each field"""

        if method == "zeros":
            self.q_table = np.zeros((self.row_count*self.column_count, 4))
        elif method=="random":
            np.random.seed(1)
            self.q_table = np.random.uniform(-0.3, 0.3, (self.row_count*self.column_count, 4))
            # note: setting q values of the goal field to zero would be preferable, as q values of other fields could otherwise exceed goal reward (not critical, but notable)
        

    def update_q_table(self, learning_rate, discount_rate):
        """Updates the value of the previous field the agent was on based on Temporal Difference (TD)"""

        # Choose action to perform from current field
        action = self.choose_action()

        # Get the current q value for that action on that field
        old_q_val = self.q_table[self.cur_state, action]

        # Get the new field the agent would be in after performing that the chosen action
        new_state = self.get_new_state(action)

        # Get the immediate reward
        immediate_reward = self.field_mapping[self.maze[new_state]]

        # Get the maximum q value of the actions that can be performed in the new field
        max_future_q_val = max(self.q_table[new_state])

        # Calculate the new q value for the chosen action for the current field based on TD
        new_q_val = old_q_val + learning_rate * (immediate_reward + discount_rate * max_future_q_val - old_q_val)

        # Update the q table with the value
        self.q_table[self.cur_state, action] = new_q_val

        # Update the state to perform the action
        self.update_state(new_state)

    def action_possible(self, action):
        """Determine if a given action is possible from the current state"""

        new_state = self.get_new_state(action)

        # Prevent moving out of bounds of the maze grid
        if new_state < 0 or new_state >= len(self.maze):
            return False
        # Prevent moving from end of one row to start of next or vice versa by moving right/left
        if (not self.cur_state == 0) and (
            (self.cur_state % self.column_count == 0 and action == 0) or # start of row
            (self.cur_state % self.column_count == self.column_count - 1 and action == 2)): # end of row
            return False
        # Prevent moving onto walls
        if self.maze[new_state] == "W":
            return False
        
        return True

    def get_new_state(self, action):
        """Get the state the agent will be in after performing the given action"""

        if action == 0: # Move left
            new_state = self.cur_state - 1
        elif action == 1: # Move down
            new_state = self.cur_state + self.column_count
        elif action == 2: # Move right
            new_state = self.cur_state + 1
        elif action == 3: # Move up
            new_state = self.cur_state - self.column_count
        else:
            raise KeyError("Invalid action")

        return new_state

    def update_state(self, new_state):
        """Update the current state based on the new state. If the new state is the goal field then reset the agent position to the start field."""

        self.cur_state = new_state

        next_maze_field = self.maze[new_state]
        if next_maze_field == "G":
            self.cur_state = self.start_field

    def choose_action(self):
        """Choose action using epsilon-greedy policy"""
        self.reset_random_seed()

        if np.random.random() < 1-self.epsilon:
            current_state_q_vals = self.q_table[self.cur_state]
            sorted_q_vals = np.sort(current_state_q_vals)

            # Go through the sorted q values starting at the biggest and choose the action corresponding to that q value, if that action is possible
            for i in range(4)[::-1]:
                next_biggest_val = sorted_q_vals[i]

                best_actions = np.where(current_state_q_vals == next_biggest_val)[0]

                if (len(best_actions) > 1): # If there are multiple actions with the same q highest value choose a random one of those actions
                    np.random.shuffle(best_actions)

                    for act in best_actions:
                        if self.action_possible(act): 
                            return act
                else: # If there are not multiple best actions simply choose the one best action
                    index_of_max_q_val = int(np.where(current_state_q_vals == next_biggest_val)[0][0])

                    if self.action_possible(index_of_max_q_val):
                        return index_of_max_q_val
        
        # If none of the best actions are possible or if the random number is bigger than 1-epsilon simply choose a random action
        actions = np.arange(4)
        np.random.shuffle(actions)
        for act in actions:
            if self.action_possible(act): 
                return act
    
    def reset_random_seed(self):
        """Function to reintroduce randomness after random seed has been set once. Required if q table initialization is random and a random seed is set for reproducability"""

        t = 1000 * time.time()
        np.random.seed(int(t) % 2**32)


# Chose the maze layout here (note: all rows should be the same size and all columns should be the same size)
maze_base = """
oooooo
oooWWo
SoooWo
ooW-Go
"""[1:-1]

q = QMaze(maze_base)

# Copy starting q_table for later comparison
original_q_table = np.copy(q.q_table)

# Initialize pygame window
pygame.init()
CELL_SIZE = 80
WINDOW_WIDTH = q.column_count * CELL_SIZE
WINDOW_HEIGHT = (q.row_count) * CELL_SIZE
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Q-Learning Maze")

# Color mapping for pygame maze display
colors = {
    "S": (0, 0, 255),     # Start - Blue
    "W": (0, 0, 0),       # Wall - Black
    "G": (0, 255, 0),     # Goal - Green
    "o": (200, 200, 200), # Empty - Light Gray
    "+": (255, 255, 0),   # Bonus - Yellow
    "-": (255, 0, 0)      # Trap - Red
}

def display_maze():
    """Display maze using pygame"""

    # Stop running if pygame window is closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # Draw maze
    for idx, tile in enumerate(q.maze):
        row = idx // q.column_count
        col = idx % q.column_count

        rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)

        color = colors.get(tile, (200, 200, 200))  # Default is light gray
        pygame.draw.rect(screen, color, rect)

        # Draw agent
        if idx == q.cur_state:
            pygame.draw.circle(screen, (0, 0, 0), rect.center, CELL_SIZE // 3)

        pygame.draw.rect(screen, (50, 50, 50), rect, 1)  # Grid lines

    pygame.display.flip()

    # 30 FPS
    clock.tick(30)

def print_maze():
    """Print maze in terminal"""

    os.system("cls")
    print(f"EPOCH: {i}")
    cur_maze = "".join(q.maze)[:q.cur_state] + "A" + "".join(q.maze)[q.cur_state+1:]
    j = q.column_count
    while j < len(q.maze):
        cur_maze = cur_maze[:j] + "\n" + cur_maze[j:]
        j += q.column_count+1
    print(cur_maze)


# Training
epochs = 4500
for i in range(epochs):
    # Chooses an action for current state, adjusts q values, then performs that action
    q.update_q_table(0.05, 0.9)

    # Display maze using pygame
    display_maze()


# Print out q table before and after training for comparison
print(original_q_table)
print(q.q_table)
print(q.q_table - original_q_table)
