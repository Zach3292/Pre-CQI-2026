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
DEPOT_TARGETS = {}
first_turn = True
MIN_SNOW_UNLOAD = 10

def decide(state: GameState) -> Orders:
    if first_turn:
        first_turn_action(state.map)
    orders = Orders()
    for plow in state.my_plows:
        if plow.broken:
            next_move = go_to_garage(plow, state.map, state.plows)
            if next_move is not None:
                orders.move(plow, next_move)
        else:
            if plow.load > MIN_SNOW_UNLOAD:
                if next_to_depot(plow, state.map):
                    DEPOT_TARGETS.pop(plow.id, None)
                    orders.unload(plow)
                else:
                    next_move = go_to_depot(plow, state.map, state.plows)
                    if next_move is not None:
                        orders.move(plow, next_move)
            else:
                orders.move(plow, random.choice(get_possible_move(plow, state.map)))
    return orders


#===============HELPER FUNCTION=================

def first_turn_action(map: GameMap):
    global first_turn
    # Get Depot and Garage position
    for i in range(map.width):
        for j in range(map.height):
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

def go_to_garage(plow: Plow, map: GameMap, plows):
    garage_position = GARAGE_POSITION[0]
    other_plows = [other.position for other in plows if other.id != plow.id]
    next_move = astar_next_move(plow.position, garage_position, map, other_plows)
    return next_move

def go_to_depot(plow: Plow, map: GameMap, plows):
    """Return the next move toward the closest depot."""
    if not DEPOT_POSITION:
        return None

    other_plows = [
        other.position for other in plows
        if other.id != plow.id
    ]
    blocked_positions = {
        Position(position.x % map.width, position.y % map.height)
        for position in other_plows
    }
    target = DEPOT_TARGETS.get(plow.id)
    if (
        target is not None
        and target not in blocked_positions
        and map.passable(target.x, target.y)
        and any(is_adjacent_to_depot(target, depot, map) for depot in DEPOT_POSITION)
    ):
        next_move = astar_next_move(plow.position, target, map, other_plows)
        if next_move is not None:
            return next_move
        DEPOT_TARGETS.pop(plow.id, None)

    depot_targets = []
    for depot_x, depot_y in DEPOT_POSITION:
        for target in (
            Position(depot_x + 1, depot_y),
            Position(depot_x - 1, depot_y),
            Position(depot_x, depot_y + 1),
            Position(depot_x, depot_y - 1),
        ):
            target = Position(target.x % map.width, target.y % map.height)
            if (
                map.passable(target.x, target.y)
                and target not in blocked_positions
            ):
                if target not in depot_targets:
                    depot_targets.append(target)

    if not depot_targets:
        return None

    depot_targets.sort(
        key=lambda position: (
            abs(plow.position.x - position.x)
            + abs(plow.position.y - position.y),
            position.x,
            position.y,
        ),
    )
    for target in depot_targets:
        next_move = astar_next_move(plow.position, target, map, other_plows)
        if next_move is not None:
            DEPOT_TARGETS[plow.id] = target
            return next_move

    return None


def is_adjacent_to_depot(position: Position, depot, map: GameMap):
    depot_x, depot_y = depot
    return (
        (position.x - depot_x) % map.width in (1, map.width - 1)
        and position.y == depot_y
    ) or (
        (position.y - depot_y) % map.height in (1, map.height - 1)
        and position.x == depot_x
    )


def astar_next_move(start: Position, target, map: GameMap, blocked_positions=()):
    """Return the next direction on an A* path to target."""
    start = Position(start.x % map.width, start.y % map.height)
    if isinstance(target, Position):
        target_x, target_y = target.x, target.y
    else:
        target_x, target_y = target
    if isinstance(target_x, (tuple, list)):
        target_x, target_y = target_x
    target = Position(target_x % map.width, target_y % map.height)
    blocked = {
        Position(position.x % map.width, position.y % map.height)
        for position in blocked_positions
    }
    path = find_path(
        start,
        target,
        neighbors_fnct=lambda position: astar_neighbors(position, map, blocked),
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


def astar_neighbors(position: Position, map: GameMap, blocked_positions=()):
    for candidate in (
        Position(position.x + 1, position.y),
        Position(position.x - 1, position.y),
        Position(position.x, position.y + 1),
        Position(position.x, position.y - 1),
    ):
        candidate = Position(candidate.x % map.width, candidate.y % map.height)
        if (
            candidate not in blocked_positions
            and map.passable(candidate.x, candidate.y)
        ):
            yield candidate
