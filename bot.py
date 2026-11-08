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
GAME_LENGTH = 1500
MINIMUM_CASH_RESERVE = 10

def decide(state: GameState) -> Orders:
    if first_turn:
        first_turn_action(state.map)
    orders = Orders()
    reserved_positions = set()
    purchase = should_buy_plow(state)
    if purchase is not None:
        orders.buy = purchase
    for plow in state.my_plows:
        if plow.broken:
            next_move = go_to_garage(
                plow, state.map, state.plows, reserved_positions
            )
            if next_move is not None:
                orders.move(plow, next_move)
                reserve_next_positions(
                    reserved_positions, plow.position, next_move, state.map
                )
        else:
            if plow.load > MIN_SNOW_UNLOAD:
                if next_to_depot(plow, state.map):
                    DEPOT_TARGETS.pop(plow.id, None)
                    orders.unload(plow)
                else:
                    next_move = go_to_depot(
                        plow, state.map, state.plows, reserved_positions
                    )
                    if next_move is not None:
                        orders.move(plow, next_move)
                        reserve_next_positions(
                            reserved_positions, plow.position, next_move, state.map
                        )
            else:
                possible_moves = [
                    direction
                    for direction in get_possible_move(plow, state.map)
                    if move_destination(
                        plow.position, direction, state.map
                    ) not in reserved_positions
                ]
                if possible_moves:
                    next_move = random.choice(possible_moves)
                    orders.move(plow, next_move)
                    reserve_next_positions(
                        reserved_positions, plow.position, next_move, state.map
                    )
    return orders


def should_buy_plow(state: GameState):
    """Return a profitable plow type to buy, or None.

    Both plow types currently have the same speed and capacity. A normal
    plow is therefore preferred because its purchase price is lower.
    """
    remaining_ticks = GAME_LENGTH - state.tick
    if remaining_ticks <= 0 or state.money <= MINIMUM_CASH_RESERVE:
        return None

    if any(plow.position == state.garage for plow in state.plows):
        return None

    normal_price = get_constant(
        state.constants,
        "normal_price",
        "price_normal",
        "plow_price",
        default=250,
    )
    big_price = get_constant(
        state.constants,
        "big_price",
        "price_big",
        "big_plow_price",
        default=400,
    )
    expected_income_per_tick = get_constant(
        state.constants,
        "expected_income_per_tick",
        "income_per_tick",
        default=1,
    )

    candidates = (
        ("normal", normal_price),
        ("big", big_price),
    )
    profitable = [
        (remaining_ticks * expected_income_per_tick - price, plow_type)
        for plow_type, price in candidates
        if state.money >= price + MINIMUM_CASH_RESERVE
        and remaining_ticks * expected_income_per_tick > price
    ]
    if not profitable:
        return None
    return max(profitable)[1]


def get_constant(constants, *names, default):
    for name in names:
        value = constants.get(name)
        if isinstance(value, (int, float)):
            return value
    return default


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
    # clean print the map
    # print("Map:")
    # for j in range(map.height):
    #     row = ""
    #     for i in range(map.width):
    #         row += map.tile(i, j)
    #     print(row)

def next_to_depot(plow: Plow, map: GameMap):
    result = False
    if map.tile(plow.position.x, plow.position.y + 1) == "D": result = True
    if map.tile(plow.position.x, plow.position.y - 1) == "D": result = True
    if map.tile(plow.position.x + 1, plow.position.y) == "D": result = True
    if map.tile(plow.position.x - 1, plow.position.y) == "D": result = True
    return result

def get_possible_move(plow: Plow, map: GameMap):
    return list(get_possible_move_from_position(plow.position, map))

def go_to_garage(plow: Plow, map: GameMap, plows, reserved_positions=()):
    garage_position = GARAGE_POSITION[0]
    other_plows = [
        other.position
        for other in plows
        if other.id != plow.id and not other.broken
    ]
    next_move = astar_next_move(
        plow.position,
        garage_position,
        map,
        other_plows + list(reserved_positions),
    )
    return next_move

def go_to_depot(plow: Plow, map: GameMap, plows, reserved_positions=()):
    """Return the next move toward the closest depot."""
    if not DEPOT_POSITION:
        return None

    other_plows = [
        other.position for other in plows
        if other.id != plow.id and not other.broken
    ]
    blocked_positions = {
        Position(position.x % map.width, position.y % map.height)
        for position in other_plows
    }
    blocked_positions.update(reserved_positions)
    target = DEPOT_TARGETS.get(plow.id)
    if (
        target is not None
        and target not in blocked_positions
        and map.passable(target.x, target.y)
        and any(is_adjacent_to_depot(target, depot, map) for depot in DEPOT_POSITION)
    ):
        next_move = astar_next_move(
            plow.position, target, map, blocked_positions
        )
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
        next_move = astar_next_move(
            plow.position, target, map, blocked_positions
        )
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
    if target in blocked or not map.passable(target.x, target.y):
        return None
    path = find_path(
        start,
        target,
        neighbors_fnct=lambda position: astar_neighbors(position, map, blocked),
        heuristic_cost_estimate_fnct=lambda position, goal: toroidal_distance(
            position, goal, map
        ),
    )
    path = list(path) if path is not None else []
    if len(path) < 2:
        return None
    if any(
        position in blocked or not map.passable(position.x, position.y)
        for position in path[1:]
    ):
        return None

    next_position = path[1]
    if not are_adjacent(start, next_position, map):
        return None
    if next_position.x == start.x + 1:
        direction = "E"
    elif next_position.x == start.x - 1:
        direction = "O"
    elif next_position.y == start.y - 1:
        direction = "N"
    elif next_position.y == start.y + 1:
        direction = "S"
    elif start.x == 0 and next_position.x == map.width - 1:
        direction = "O"
    elif start.x == map.width - 1 and next_position.x == 0:
        direction = "E"
    elif start.y == 0 and next_position.y == map.height - 1:
        direction = "N"
    elif start.y == map.height - 1 and next_position.y == 0:
        direction = "S"
    else:
        direction = "N"

    destination = move_destination(start, direction, map)
    if (
        destination != next_position
        or destination in blocked
        or not map.passable(destination.x, destination.y)
    ):
        return None
    return direction


def move_destination(position: Position, direction: str, map: GameMap):
    offsets = {
        "N": (0, -1),
        "S": (0, 1),
        "E": (1, 0),
        "O": (-1, 0),
    }
    dx, dy = offsets[direction]
    return Position(
        (position.x + dx) % map.width,
        (position.y + dy) % map.height,
    )


def reserve_next_positions(
    reserved_positions: set,
    position: Position,
    direction: str,
    map: GameMap,
):
    first_position = move_destination(position, direction, map)
    second_position = move_destination(first_position, direction, map)
    reserved_positions.add(first_position)
    reserved_positions.add(second_position)


def are_adjacent(first: Position, second: Position, map: GameMap):
    return (
        (second.x - first.x) % map.width in (1, map.width - 1)
        and second.y == first.y
    ) or (
        (second.y - first.y) % map.height in (1, map.height - 1)
        and second.x == first.x
    )


def toroidal_distance(first: Position, second: Position, map: GameMap):
    x_distance = abs(first.x - second.x)
    y_distance = abs(first.y - second.y)
    return (
        min(x_distance, map.width - x_distance)
        + min(y_distance, map.height - y_distance)
    )


def astar_neighbors(position: Position, map: GameMap, blocked_positions=()):
    for direction in get_possible_move_from_position(position, map):
        candidate = move_destination(position, direction, map)
        if candidate not in blocked_positions:
            yield candidate


def get_possible_move_from_position(
    position: Position,
    map: GameMap,
):
    directions = (
        ("N", Position(position.x, position.y - 1)),
        ("S", Position(position.x, position.y + 1)),
        ("E", Position(position.x + 1, position.y)),
        ("O", Position(position.x - 1, position.y)),
    )
    for direction, candidate in directions:
        candidate = Position(candidate.x % map.width, candidate.y % map.height)
        if map.passable(candidate.x, candidate.y):
            yield direction
