import math
import numpy as np
import picar_4wd as fc
import time
import math
import sys
import networkx as nx
np.set_printoptions(threshold=sys.maxsize)

# Constants
cell_size = 5
drive_speed = 40
turn_speed = 20
map_size = 60

# Seconds for a full rotation. ONLY WORKS WITH SPEED = 40 TODO: Measure
left_turn_time = 6.4
right_turn_time = 6.2

# Seconds to drive 100cm. ONLY WORKS WITH SPEED = 40 TODO: measure
drive_time = 3.4

# Global variables
position = (map_size // 2, map_size // 2)
orientation = 0
cur_map = np.zeros((map_size, map_size), dtype=np.int32)


def drive_forward(cell_distance=1):
    global position, orientation
    fc.forward(drive_speed)
    time.sleep(cell_distance * 0.25)
    fc.stop()
    position = (round(position[0] + cell_distance * math.cos(orientation * math.pi / 2)), round(position[1] + cell_distance * math.sin(orientation * math.pi / 2)))

def turn_right(rotation=1/4):
    global orientation
    fc.turn_right(turn_speed)
    time.sleep(right_turn_time * rotation)
    fc.stop()
    orientation -= 4 * rotation
    time.sleep(1.5)

def turn_left(rotation=1/4):
    global orientation
    fc.turn_left(turn_speed)
    time.sleep(left_turn_time * rotation)
    fc.stop()
    orientation += 4 * rotation
    time.sleep(1.5)

def scan(angle=90, num_scans=13):
    step = (2 * angle)/(num_scans - 1)
    dists = []
    angles = []
    fc.get_distance_at(-1 * angle)
    time.sleep(0.1)
    for i in range(0, num_scans):
        cur_angle = -1 * angle + i * step
        angles.append(cur_angle)
        x = fc.get_distance_at(cur_angle)
        dists.append(x if x >= 0 else 999)
        time.sleep(0.15)

    return angles, dists

def add_point_border(x_coord, y_coord, radius=2, num=1):
    global cur_map, position

    x_coord = round(x_coord)
    y_coord = round(y_coord)
    rows, cols = cur_map.shape
    x, y = np.ogrid[:rows, :cols]
    distances = np.sqrt((x - x_coord)**2 + (y - y_coord)**2)
    for r in range(radius + 1):
        cur_map[distances <= r] += 1
    cur_map[position] = 0
    
def update_map(angles, dists, max_threshold=60, min_threshold=0, num=1):
    global cur_map, position, orientation
    cur_map[position] = 7

    for angle, dist in zip(angles, dists):
        if dist < max_threshold and dist > min_threshold:
            reading_rad = angle * math.pi / 180
            
            # x, y coordinates of obstacle wrt robot.
            dx = math.cos(reading_rad + orientation * math.pi / 2) * dist / cell_size
            dy = math.sin(reading_rad + orientation * math.pi / 2) * dist / cell_size
            
            # mapped_x = dx * math.cos(orientation) - dy * math.sin(orientation) + position[0]
            # mapped_y = dx * math.sin(orientation) + dy * math.cos(orientation) + position[1]
            
            add_point_border(position[0] + dx, position[1] + dy, num=num)
            # cur_map[round(position[0] + dx), round(position[1] + dy)] = 1
    
def create_graph_from_2d_array_with_obstacles(array):
    rows, cols = array.shape
    G = nx.grid_2d_graph(rows, cols)
    # Identify and remove obstacles directly within this function
    obstacles = [(i, j) for i in range(rows) for j in range(cols) if array[i, j] > 1]
    G.remove_nodes_from(obstacles)
    return G


def update_graph_with_obstacles(G, array):
    # Identify new obstacles and remove corresponding nodes from the graph
    rows, cols = array.shape
    for i in range(rows):
        for j in range(cols):
            if array[i, j] >= 1 and (i, j) in G:
                G.remove_node((i, j))
    return G


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_required_turn(current, next):
  # Determine the required turn to face the next cell from current
  dx = next[0] - current[0]
  dy = next[1] - current[1]
  if dx == 1:  # Need to face right
      return 0.0  # 0 radians
#   elif dx == -1:  # Need to face left
#       return math.pi  # 180 degrees in radians
  elif dy == 1:  # Need to face up
      return 1  # 90 degrees in radians
  else:  # Need to face down
      return -1  # 270 degrees in radians
  
def adjust_orientation_and_move(next_position):
    global orientation
    global position
    required_orientation = get_required_turn(position, next_position)
    print("Required orientation: ", required_orientation)
    print("Orientation: ", orientation)
    # Adjust orientation
    if required_orientation == orientation + 1:
        print("TURN LEFT")
        turn_left()

    elif required_orientation == orientation - 1:
        print("TURN RIGHT")
        turn_right()
    else:
        print("CORRECT ORIENTATION")

    # Move forward by one cell
    print("DRIVE FORWARD")
    drive_forward(1)
    time.sleep(1)

def drive(goal):
    global cur_map, position, orientation
    # G = create_graph_from_2d_array_with_obstacles(cur_map)
    fc.get_distance_at(0)
    time.sleep(0.5)
    it = 0
    while position != goal:
        try:
            angles, dists = scan()
            update_map(angles, dists)
            G = create_graph_from_2d_array_with_obstacles(cur_map)
            path = nx.astar_path(G, position, goal, heuristic)
            print("Current Path:", path[:5])
            
            for i in range(1, 5):
                # Move one step along the path
                next_position = path[i]  # Next step in the path
                print("Current: ", position)
                print("Next: ", next_position)
                adjust_orientation_and_move(next_position)
                print("Moved to:", next_position)

            it += 1
            if it == 6:
                break
        except nx.NetworkXNoPath:
            print("No path found to the goal from current position.")
            return
    print("Goal reached.")

goal = (map_size // 2 + 20, map_size // 2)
drive(goal)
# drive_forward(1)
cur_map = (cur_map >= 1).astype(int)
print("------------------ LAST MAP ----------------------")
# print(np.flipud(cur_map.T))
print(cur_map)
