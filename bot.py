"""Votre bot. C'est le seul fichier que vous avez besoin de modifier.

decide() est appelée à chaque tour avec l'état du jeu et doit retourner vos ordres.
Pour l'instant, chaque déneigeuse se déplace au hasard.
"""

import random
import numpy as np
from astar import find_path

from models import GameState, Orders, GameMap, Call, Plow, Position

DEPOT_POSITION = [] #np.array([(10, 10), (10, 11), (10, 12), (10, 30), (10, 31), (10, 32), (11, 10), (11, 11), (11, 12), (11, 30), (11, 31), (11, 32), (12, 10), (12, 11), (12, 12), (12, 30), (12, 31), (12, 32)])
GARAGE_POSITION = []
first_turn = True

def decide(state: GameState) -> Orders:
    if first_turn:
        first_turn_action(state.map)
    orders = Orders()
    for plow in state.my_plows:
        if plow.broken:
            go_to_garage(plow, state.map)
        else:
            if next_to_depot(plow, state.map):
                orders.unload(plow)
            else:
                orders.move(plow, random.choice(get_possible_move(plow, state.map)))
    return orders


#===============HELPER FUNCTION=================

def first_turn_action(map: GameMap):
    # Get Depot and Garage position
    for i in range(map.height):
        for j in range(map.width):
            if map.tile(i, j) == "D":
                DEPOT_POSITION.append((i, j))
            if map.tile(i, j) == "G":
                GARAGE_POSITION.append((i, j))
    first_turn = False

def next_to_depot(plow: Plow, map: GameMap):
    result = False
    if map.tile(plow.position.x, plow.position.y+1) == "D": result = True
    if map.tile(plow.position.x, plow.position.y-1) == "D": result = True
    if map.tile(plow.position.x+1, plow.position.y) == "D": result = True
    if map.tile(plow.position.x-1, plow.position.y) == "D": result = True
    return result

def get_possible_move(plow: Plow, map: GameMap):
    move = []
    if map.passable(plow.position.x, plow.position.y+1): move.append("N")
    if map.passable(plow.position.x, plow.position.y-1): move.append("S")
    if map.passable(plow.position.x+1, plow.position.y): move.append("E")
    if map.passable(plow.position.x-1, plow.position.y): move.append("O")

    return move

def go_to_garage(plow: Plow, map: GameMap):
    garage_position = Position(GARAGE_POSITION[0], GARAGE_POSITION[1])
    next_move = astar_next_move(plow.position, garage_position, map)
    return next_move


def astar_next_move(start: Position, target: Position, map: GameMap):
    """Return the next direction on an A* path to target."""
    start = Position(start.x % map.width, start.y % map.height)
    target = Position(target.x % map.width, target.y % map.height)
    path = find_path(
        start,
        target,
        neighbors_fnct=lambda position: astar_neighbors(position, map),
        heuristic_cost_estimate_fnct=lambda position, goal: (
            abs(position.x - goal.x) + abs(position.y - goal.y)
        ),
    )
    path = list(path) if path is not None else []
    if len(path) < 2:
        return None

    next_position = path[1]
    if next_position.x == start.x + 1:
        return "E"
    if next_position.x == start.x - 1:
        return "O"
    if next_position.y == start.y + 1:
        return "N"
    if next_position.y == start.y - 1:
        return "S"
    if start.x == 0 and next_position.x == map.width - 1:
        return "O"
    if start.x == map.width - 1 and next_position.x == 0:
        return "E"
    if start.y == 0 and next_position.y == map.height - 1:
        return "S"
    return "N"


def astar_neighbors(position: Position, map: GameMap):
    for candidate in (
        Position(position.x + 1, position.y),
        Position(position.x - 1, position.y),
        Position(position.x, position.y + 1),
        Position(position.x, position.y - 1),
    ):
        candidate = Position(candidate.x % map.width, candidate.y % map.height)
        if map.passable(candidate.x, candidate.y):
            yield candidate
