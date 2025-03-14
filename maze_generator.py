import random
import networkx as netx
from PIL import Image, ImageDraw
import itertools
from tqdm import tqdm
import argparse
import os

# Maze Generator using Depth-First Search

# This implementation generates mazes via a depth-first search algorithm,
# as described in https://scipython.com/blog/making-a-maze/ by Christian Hill (April 2017).
# Extended by Gabriel Diaz (2024) to incorporate cycles, dynamic terminal selection,
# and to compute the minimum Steiner tree spanning the terminal nodes.

# Class representing a single cell in the maze.
class Cell:
    # Mapping for wall pairs.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    
    # Initialize a Cell object with its (x, y) coordinates and all walls intact.
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}
    
    # Return True if the cell still has all its walls.
    def has_all_walls(self):
        return all(self.walls.values())
    
    # Knock down the wall between this cell and another cell.
    def knock_down_wall(self, other, wall):
        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False

# Class representing the maze.
class Maze:
    # Initialize the maze grid with given dimensions.
    def __init__(self, nx, ny, ix=0, iy=0):
        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]
    
    # Return the cell object at coordinates (x, y).
    def cell_at(self, x, y):
        return self.maze_map[x][y]
    
    # Return a list of unvisited neighbour cells for a given cell.
    def find_valid_neighbours(self, cell):
        delta = [('W', (-1, 0)), ('E', (1, 0)), ('S', (0, 1)), ('N', (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours
    
    # Return the number of walls intact for a given cell.
    def count_walls(self, cell):
        return sum(cell.walls.values())
    
    # Generate the maze using depth-first search and set the terminal cells.
    def make_maze_general(self, num_terminals=5):
        # Create full maze using depth-first search.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        nv = 1
        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)
            if not neighbours:
                current_cell = cell_stack.pop()
                continue
            direction, next_cell = random.choice(neighbours)
            current_cell.knock_down_wall(next_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1

        # Remove walls randomly from cells with three intact walls (up to 3 removals).
        wall_removals = 0
        cells = [(x, y) for x in range(self.nx) for y in range(self.ny)]
        random.shuffle(cells)
        for x, y in cells:
            cell = self.cell_at(x, y)
            if self.count_walls(cell) == 3:
                removable_walls = [wall for wall, present in cell.walls.items() if present]
                wall_to_remove = random.choice(removable_walls)
                dx, dy = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}[wall_to_remove]
                x2, y2 = x + dx, y + dy
                if 0 <= x2 < self.nx and 0 <= y2 < self.ny:
                    other = self.cell_at(x2, y2)
                    if self.count_walls(other) > 0:
                        cell.knock_down_wall(other, wall_to_remove)
                        wall_removals += 1
                        if wall_removals >= 3:
                            break

        # Randomly choose start and end cells from anywhere in the maze.
        low_x, high_x = 0, self.nx
        low_y, high_y = 0, self.ny
        start_x = random.randint(low_x, high_x - 1)
        start_y = random.randint(low_y, high_y - 1)
        end_x = random.randint(low_x, high_x - 1)
        end_y = random.randint(low_y, high_y - 1)
        while start_x == end_x and start_y == end_y:
            end_x = random.randint(low_x, high_x - 1)
            end_y = random.randint(low_y, high_y - 1)
        self.start_cell = self.cell_at(start_x, start_y)
        self.end_cell = self.cell_at(end_x, end_y)

        # Generate num_terminals intermediate cells (terminals).
        intermediate_coords = []
        for _ in range(num_terminals):
            tx = random.randint(low_x, high_x - 1)
            ty = random.randint(low_y, high_y - 1)
            while (tx, ty) == (start_x, start_y) or (tx, ty) == (end_x, end_y) or (tx, ty) in intermediate_coords:
                tx = random.randint(low_x, high_x - 1)
                ty = random.randint(low_y, high_y - 1)
            intermediate_coords.append((tx, ty))
        # Terminals list: start, intermediates, then end.
        self.terminals = [self.start_cell] + [self.cell_at(x, y) for (x, y) in intermediate_coords] + [self.end_cell]

        # Create maze graph.
        self.graph = netx.Graph()
        for x in range(self.nx):
            for y in range(self.ny):
                cell = self.cell_at(x, y)
                cell_id = (x, y)
                self.graph.add_node(cell_id)
                for direction, (dx, dy) in [('N', (0, -1)), ('S', (0, 1)), ('E', (1, 0)), ('W', (-1, 0))]:
                    if not cell.walls[direction]:
                        neighbor_x, neighbor_y = x + dx, y + dy
                        if (0 <= neighbor_x < self.nx) and (0 <= neighbor_y < self.ny):
                            neighbor_id = (neighbor_x, neighbor_y)
                            self.graph.add_edge(cell_id, neighbor_id)
        return wall_removals

    # Generate an image of the maze with terminal cells highlighted.
    def generate_image_general(self, i, cell_size=2, padding=3, filename=None):
        img_width = (self.nx * 2 - 1) * cell_size + 2 * padding
        img_height = (self.ny * 2 - 1) * cell_size + 2 * padding
        img = Image.new('RGB', (img_width, img_height), 'black')
        draw = ImageDraw.Draw(img)
        terminal_coords = [(t.x, t.y) for t in self.terminals]
        for node in self.graph.nodes:
            x = node[0] * 2 * cell_size + padding
            y = node[1] * 2 * cell_size + padding
            color = (0, 255, 0) if node in terminal_coords else 'white'
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill=color)
        for edge in self.graph.edges:
            x = (edge[0][0] + edge[1][0]) * cell_size + padding
            y = (edge[0][1] + edge[1][1]) * cell_size + padding
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill='white')
        if filename is not None:
            img.save(f'{filename}/maze_{i}.png')
        else:
            img.save(f'mazes/maze_{i}.png')

    # Compute the shortest path between two cells using NetworkX.
    def shortest_path_graph(self, start, end, drawing=True):
        all_shortest_paths = netx.all_shortest_paths(self.graph, (start.x, start.y), (end.x, end.y))
        shortest_path = next(all_shortest_paths, None)
        combined_graph = self.graph.copy()
        edges_to_remove = [edge for edge in combined_graph.edges 
                           if edge not in zip(shortest_path[:-1], shortest_path[1:]) 
                           and edge not in zip(shortest_path[1:], shortest_path[:-1])]
        combined_graph.remove_edges_from(edges_to_remove)
        if drawing:
            self.draw_combined_graph(combined_graph)
        unique = len(list(all_shortest_paths)) == 0
        return shortest_path, unique

    # Generate an image overlay of the shortest path on the maze.
    def generate_shortest_path_image(self, shortest_path, i, cell_size=2, padding=3, filename=None):
        shortest_path_set = set(shortest_path)
        img_width = (self.nx * 2 - 1) * cell_size + 2 * padding
        img_height = (self.ny * 2 - 1) * cell_size + 2 * padding
        img = Image.new('RGB', (img_width, img_height), 'black')
        draw = ImageDraw.Draw(img)
        for node in self.graph.nodes:
            if node in shortest_path_set:
                x = node[0] * 2 * cell_size + padding
                y = node[1] * 2 * cell_size + padding
                draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill='white')
        for edge in zip(shortest_path[:-1], shortest_path[1:]):
            x = (edge[0][0] + edge[1][0]) * cell_size + padding
            y = (edge[0][1] + edge[1][1]) * cell_size + padding
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill='white')
        if filename is not None:
            img.save(f'{filename}/label_{i}.png')
        else:
            img.save(f'mazes/label_{i}.png')

# Precompute distances and paths between all pairs of terminal cells.
def precompute_distances(points, maze):
    distances = {}
    paths = {}
    for i, start_point in enumerate(points):
        for end_point in points[i+1:]:
            path, _ = maze.shortest_path_graph(start=start_point, end=end_point, drawing=False)
            distance = len(path)
            distances[(start_point, end_point)] = distance
            distances[(end_point, start_point)] = distance
            paths[(start_point, end_point)] = path
            paths[(end_point, start_point)] = path[::-1]
    return distances, paths

# Generate multiple mazes with terminals and compute their shortest paths.
def make_mazes(num_mazes, file_to_save=None, terminals=5, nx=11, ny=11):
    data = []
    for j in tqdm(range(num_mazes), desc="Generating mazes"):
        while True:
            maze = Maze(nx, ny)
            walls = maze.make_maze_general(num_terminals=terminals)
            maze.generate_image_general(j, filename=file_to_save)
            points = maze.terminals
            distances, paths = precompute_distances(points, maze)
            shortest_total_path_length = float('inf')
            shortest_total_path = None
            permutations = list(itertools.permutations(points))
            for perm in permutations:
                total_length = sum(distances[(perm[i], perm[i+1])] for i in range(len(perm)-1))
                if total_length < shortest_total_path_length:
                    shortest_total_path_length = total_length
                    shortest_total_path = []
                    for i in range(len(perm)-1):
                        shortest_total_path.extend(paths[(perm[i], perm[i+1])])
            unique_shortest_total_path = set(shortest_total_path)
            shortest_total_path_length = len(unique_shortest_total_path)
            if shortest_total_path_length < 100:
                break
        maze.generate_shortest_path_image(shortest_total_path, j, filename=file_to_save)
        data.append([maze.graph.number_of_edges(), shortest_total_path_length - 1])
    return data

# Main block: Parse command-line arguments and generate mazes.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mazes with terminals.")
    parser.add_argument("--num_mazes", type=int, default=2, help="Number of mazes to generate")
    parser.add_argument("--terminals", type=int, default=3, help="Number of intermediate terminals (excluding start and end)")
    parser.add_argument("--nx", type=int, default=11, help="Maze width")
    parser.add_argument("--ny", type=int, default=11, help="Maze height")
    parser.add_argument("--output", type=str, default="mazes/", help="Directory to save maze images")
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    make_mazes(num_mazes=args.num_mazes, file_to_save=args.output, terminals=args.terminals, nx=args.nx, ny=args.ny)
