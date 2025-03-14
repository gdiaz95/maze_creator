# Maze_creator

This is a Python-based maze generator that uses a depth-first search algorithm to create mazes. It is based on the algorithm described in [this article](https://scipython.com/blog/making-a-maze/) by Christian Hill (April 2017) and was extended by Gabriel Diaz (2024) to include cycles, dynamic terminal selection, and computation of the minimum Steiner tree spanning the terminal nodes.

## Features

- **Maze Generation:** Uses a depth-first search algorithm.
- **Cycles & Wall Removal:** Randomly removes additional walls to create cycles.
- **Dynamic Terminals:** Allows specification of intermediate terminal cells (the "middle" ones). The start and end cells are also randomly selected.
- **Image Output:** Generates images with dimensions:
  - **Width:** `4 * (nx + 1)` pixels
  - **Height:** `4 * (ny + 1)` pixels  
  For example, with default values `nx = 11` and `ny = 11`, the images will be 48×48 pixels.

## Requirements

Install the required packages using the provided `requirements.txt` file


## Usage

The main script is `maze_generator.py`. You can run it directly from the terminal.

### Command-Line Arguments

- `--num_mazes`: Number of mazes to generate (default: 2)
- `--terminals`: Number of intermediate terminals (excluding start and end) (default: 2)
- `--nx`: Maze width (number of cells in x-direction) (default: 11)
- `--ny`: Maze height (number of cells in y-direction) (default: 11)
- `--output`: Directory to save maze images (default: "mazes/")

### Example

Generate 3 mazes with custom dimensions and 4 intermediate terminals, saving to a custom directory:

```bash
python maze_generator.py --num_mazes 3 --nx 15 --ny 15 --terminals 4 --output custom_mazes/
