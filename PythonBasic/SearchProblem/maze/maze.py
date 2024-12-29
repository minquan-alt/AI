import sys
import numpy as np
class Node():
    def __init__(self, state, parent, action, distance=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.distance = distance
        self.cost = cost
# Build Frontier (ordered by efficiency) : Stack -> DFS, Queue -> BFS, PriorityQueue -> GBFS, A* Search
class StackFrontier():
    def __init__(self):
        self.frontier = []
    def add(self, node):
        self.frontier.append(node)
    def contain_state(self, state):
        return any(state == node.state for node in self.frontier)
    def empty(self):
        return len(self.frontier) == 0
    def remove(self):
        if self.empty():
            raise Exception('frontier empty')
        node = self.frontier[-1]
        self.frontier = self.frontier[:-1]
        return node
class QueueFrontier():
    def __init__(self):
        self.frontier = []
    def add(self, node):
        self.frontier.append(node)
    def contain_state(self, state):
        return any(state == node.state for node in self.frontier)
    def empty(self):
        return len(self.frontier) == 0
    def remove(self):
        if self.empty():
            raise Exception('frontier empty')
        node = self.frontier[0]
        self.frontier = self.frontier[1:]
        return node
class PriorityQueueFrontier():
    def __init__(self):
        self.frontier = []
    def BinarySearch(self, key, left, right):
        if left > right:
            return left
        mid = (left + right) // 2
        if key < (self.frontier[mid].distance + self.frontier[mid].cost):
            return self.BinarySearch(key, left, mid - 1)
        elif key > (self.frontier[mid].distance + self.frontier[mid].cost):
            return self.BinarySearch(key, mid + 1, right)
        return mid
    def add(self, node):
        if not self.frontier:
            self.frontier.append(node)
        else:
            # Tìm vị trí để chèn
            idx = self.BinarySearch(node.distance + node.cost, 0, len(self.frontier) - 1)
            self.frontier.insert(idx, node)
    def contain_state(self, state):
        return any(state == node.state for node in self.frontier)
    def empty(self):
        return len(self.frontier) == 0
    def remove(self):
        if self.empty():
            raise Exception("frontier empty")
        return self.frontier.pop(0)
# build heuristic function
            
class Maze():
    def __init__(self, filename):
        with open(filename) as f:
            contents = f.read()
        
        # validate start and goal
        if contents.count('A') != 1:
            raise Exception('must have exactly one start point')
        if contents.count('B') != 1:
            raise Exception('must have exactly one end point')
        # determine height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)
        
        # keep track of wall
        self.wall = []
        for i in range (self.height):
            row = []
            for j in range (self.width):
                try:
                    if contents[i][j] == 'A':
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == 'B':
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == ' ':
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.wall.append(row)
        
        # create heuristic matrix
        # self.heuristic_matrix = np.zeros(shape=(self.height, self.width))
        # for i in range(self.height):
        #     for j in range(self.width):
        #         if not self.wall[i][j]:
        #             self.heuristic_matrix[i][j] = self.manhattanDistance(self.goal, (i, j))
        # initial solution
        self.solution = None
        
    @staticmethod
    def manhattanDistance(point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
                    
    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.wall):
            for j, col in enumerate(row):
                if col:
                    print('█', end='')
                elif (i, j) == self.start:
                    print('A', end='')
                elif (i, j) == self.goal:
                    print('B', end='')
                elif solution is not None and (i, j) in solution:
                    print('*', end='')
                else:
                    print(' ', end='')
            print()
        print()
    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up", (row-1, col)),
            ("down", (row+1, col)),
            ("left", (row, col-1)),
            ("right", (row, col+1))
        ]
        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.wall[r][c]:
                result.append((action, (r,c)))
        return result
    def solve(self):        
        # keep track of number of explored
        self.num_explored = 0
        # initialize the frontier
        start = Node(state=self.start, parent=None, action=None)
        frontier = PriorityQueueFrontier()
        frontier.add(start)
        # initialize the explored
        self.explored = set()
        
        while True:
            if frontier.empty():
                raise Exception('no solution')
            # remove node and +1 num_explored
            node = frontier.remove()
            self.num_explored += 1
            
            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return
            # mark node as explored
            self.explored.add(node.state)
            # find neightbors
            for action, state in self.neighbors(node.state):
                if not frontier.contain_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action, distance=self.manhattanDistance(self.goal, state), cost=node.cost + 1)
                    frontier.add(child)
            
    def output_image(self, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw
        import matplotlib.pyplot as plt
        cell_size = 50
        cell_border = 2

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size, self.height * cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.wall):
            for j, col in enumerate(row):

                # Walls
                if col:
                    fill = (40, 40, 40)

                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)

                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)

                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)

                # Empty cell
                else:
                    fill = (237, 240, 252)

                # Draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )

        plt.figure(figsize=(self.width, self.height))
        plt.imshow(img)
        plt.axis('off')  # Turn off axis
        plt.show()


if len(sys.argv) != 2:
    sys.exit("Usage: python maze.py maze.txt")

m = Maze(sys.argv[1])
print("Maze:")
m.print()
print("Solving...")
m.solve()
print("States Explored:", m.num_explored)
print("Solution:")
m.print()
m.output_image(show_explored=True)
