"""Votre bot. C'est le seul fichier que vous avez besoin de modifier.

decide() est appelée à chaque tour avec l'état du jeu et doit retourner vos ordres.
Pour l'instant, chaque déneigeuse se déplace au hasard.
"""

import random
import os
import time
from itertools import permutations
from astar import find_path

from models import GameState, Orders, GameMap, Plow, Position

DEPOT_POSITION = []
GARAGE_POSITION = []
DEPOT_TARGETS = {}
SNOW_PLANS = {}
CALL_TARGETS = {}
first_turn = True
MIN_SNOW_UNLOAD = 20
SNOW_CAPACITY = 20
MAX_LOOP_STOPS = 3
MAX_SNOW_CANDIDATES = 7
GAME_LENGTH = 1500
MINIMUM_CASH_RESERVE = 10
MIN_SMALL_PLOWS_FOR_BIG = 5
CALL_MIN_PROFIT_MARGIN = 0
CALL_REWARD_WEIGHT = 2
DEPOT_NEARBY_DISTANCE = 6
DECIDE_TIME_LIMIT_SECONDS = 0.35
last_decide_time_ms = 0.0
BOT_STRATEGY = os.environ.get("BOT_STRATEGY", "optimized")

def decide(state: GameState) -> Orders:
    if BOT_STRATEGY != "optimized":
        return decide_alternative(state)

    global last_decide_time_ms
    start_time = time.perf_counter()
    deadline = start_time + DECIDE_TIME_LIMIT_SECONDS
    if first_turn:
        first_turn_action(state.map)
    orders = Orders()
    reserved_positions = set()
    purchase = should_buy_plow(
        state,
        elapsed_seconds=time.perf_counter() - start_time,
    )
    if purchase is not None:
        orders.buy = purchase
    for plow in state.my_plows:
        if (
            not plow.broken
            and plow.load >= MIN_SNOW_UNLOAD
            and next_to_depot(plow, state.map)
        ):
            DEPOT_TARGETS.pop(plow.id, None)
            SNOW_PLANS.pop(plow.id, None)
            orders.unload(plow)
            continue

        if time.perf_counter() >= deadline:
            if plow.broken:
                next_move = go_to_garage(plow, state.map)
                if next_move is not None:
                    orders.move(plow, next_move)
                continue
            possible_moves = get_possible_move(plow, state.map)
            if possible_moves:
                next_move = random.choice(possible_moves)
                orders.move(plow, next_move)
                reserve_next_positions(
                    reserved_positions, plow.position, next_move, state.map
                )
            continue

        if plow.broken:
            next_move = go_to_garage(plow, state.map)
            if next_move is not None:
                orders.move(plow, next_move)
        else:
            if plow.load >= MIN_SNOW_UNLOAD:
                if next_to_depot(plow, state.map):
                    DEPOT_TARGETS.pop(plow.id, None)
                    SNOW_PLANS.pop(plow.id, None)
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
                call_target = valuable_call_target(
                    state, plow, reserved_positions, deadline
                )
                if call_target is not None:
                    next_move = astar_next_move(
                        plow.position,
                        call_target.position,
                        state.map,
                        other_plow_positions(plow, state.plows)
                        + list(reserved_positions),
                    )
                else:
                    next_move = move_on_best_snow_loop(
                        state, plow, reserved_positions, deadline
                    )
                if next_move is None:
                    possible_moves = get_possible_move(plow, state.map)
                    if possible_moves:
                        next_move = random.choice(possible_moves)
                if next_move is not None and move_destination(
                    plow.position, next_move, state.map
                ) not in reserved_positions:
                    orders.move(plow, next_move)
                    reserve_next_positions(
                        reserved_positions, plow.position, next_move, state.map
                    )
    last_decide_time_ms = (time.perf_counter() - start_time) * 1000
    return orders


def decide_alternative(state: GameState) -> Orders:
    """Run a deliberately different baseline strategy for local simulations."""
    global last_decide_time_ms
    start_time = time.perf_counter()
    if first_turn:
        first_turn_action(state.map)

    orders = Orders()
    reserved_positions = set()
    purchase = should_buy_plow(
        state,
        elapsed_seconds=time.perf_counter() - start_time,
    )
    if purchase is not None:
        orders.buy = purchase
    for plow in state.my_plows:
        if plow.broken:
            next_move = go_to_garage(plow, state.map)
        elif plow.load >= MIN_SNOW_UNLOAD and next_to_depot(plow, state.map):
            SNOW_PLANS.pop(plow.id, None)
            orders.unload(plow)
            continue
        elif plow.load >= MIN_SNOW_UNLOAD:
            next_move = go_to_depot(
                plow, state.map, state.plows, reserved_positions
            )
        else:
            target = alternative_target(state, plow)
            next_move = astar_next_move(
                plow.position,
                target,
                state.map,
                other_plow_positions(plow, state.plows)
                + list(reserved_positions),
            ) if target is not None else None

        if next_move is not None:
            destination = move_destination(plow.position, next_move, state.map)
            if destination not in reserved_positions:
                orders.move(plow, next_move)
                reserve_next_positions(
                    reserved_positions, plow.position, next_move, state.map
                )

    last_decide_time_ms = (time.perf_counter() - start_time) * 1000
    return orders


def alternative_target(state: GameState, plow: Plow):
    """Choose a target for one of the local competitor strategies."""
    blocked = other_plow_positions(plow, state.plows)
    if BOT_STRATEGY == "calls":
        call = valuable_call_target(
            state,
            plow,
            set(blocked),
            time.perf_counter() + DECIDE_TIME_LIMIT_SECONDS / 2,
        )
        if call is not None:
            return call.position

    snow_targets = [
        Position(x, y)
        for y in range(state.map.height)
        for x in range(state.map.width)
        if state.map.passable(x, y)
        and state.map.snow[y][x] > 0
        and Position(x, y) not in blocked
    ]
    if not snow_targets:
        return None

    if BOT_STRATEGY == "random":
        return random.choice(snow_targets)
    return min(
        snow_targets,
        key=lambda target: (
            toroidal_distance(plow.position, target, state.map),
            -state.map.snow[target.y][target.x],
        ),
    )


def valuable_call_target(state, plow, reserved_positions, deadline):
    """Return the best reachable call worth prioritizing."""
    if time.perf_counter() >= deadline or not state.calls:
        return None

    active_calls = {
        call.id: call
        for call in state.calls
        if call.expires_at > state.tick
        and call.position != plow.position
        and state.map.snow[call.position.y][call.position.x] > 0
        and call.position not in {
            target.position
            for plow_id, target in CALL_TARGETS.items()
            if plow_id != plow.id
        }
    }
    current_target = CALL_TARGETS.get(plow.id)
    if current_target is not None and current_target.id in active_calls:
        return current_target

    candidates = []
    for call in active_calls.values():
        distance = toroidal_distance(plow.position, call.position, state.map)
        remaining_time = call.expires_at - state.tick
        snow_amount = state.map.snow[call.position.y][call.position.x]
        enough_capacity = plow.load + snow_amount <= SNOW_CAPACITY
        if (
            enough_capacity
            and remaining_time > distance
            and call.reward * CALL_REWARD_WEIGHT
            >= distance * 2 + CALL_MIN_PROFIT_MARGIN
        ):
            candidates.append((
                call.reward * CALL_REWARD_WEIGHT - distance * 2,
                call.reward,
                -distance,
                call,
            ))

    if not candidates:
        CALL_TARGETS.pop(plow.id, None)
        return None

    target = max(candidates, key=lambda item: item[:3])[-1]
    CALL_TARGETS[plow.id] = target
    return target


def should_buy_plow(state: GameState, elapsed_seconds=0.0):
    """Return a profitable plow type to buy, or None.

    Both plow types currently have the same speed and capacity. A normal
    plow is therefore preferred because its purchase price is lower.
    """
    if (
        last_decide_time_ms >= DECIDE_TIME_LIMIT_SECONDS * 1000
        or elapsed_seconds >= DECIDE_TIME_LIMIT_SECONDS / 2
    ):
        return None

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
        default=0.85,
    )

    small_plow_count = sum(
        plow.type == "normal" for plow in state.my_plows
    )
    if small_plow_count >= MIN_SMALL_PLOWS_FOR_BIG:
        if (
            state.money >= big_price + MINIMUM_CASH_RESERVE
            and remaining_ticks * expected_income_per_tick > big_price
        ):
            return "big"
        return None

    candidates = [("normal", normal_price)]
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


def first_turn_action(map: GameMap):
    for i in range(map.width):
        for j in range(map.height):
            if map.tile(i, j) == "D":
                DEPOT_POSITION.append((i, j))
            if map.tile(i, j) == "G":
                GARAGE_POSITION.append((i, j))
    global first_turn
    first_turn = False

def next_to_depot(plow: Plow, map: GameMap):
    neighbors = (
        (plow.position.x, plow.position.y + 1),
        (plow.position.x, plow.position.y - 1),
        (plow.position.x + 1, plow.position.y),
        (plow.position.x - 1, plow.position.y),
    )
    return any(map.tile(x, y) == "D" for x, y in neighbors)

def get_possible_move(plow: Plow, map: GameMap):
    return list(get_possible_move_from_position(plow.position, map))

def go_to_garage(plow: Plow, map: GameMap):
    garage_position = GARAGE_POSITION[0]
    next_move = astar_next_move(
        plow.position,
        garage_position,
        map,
    )
    return next_move

def go_to_depot(plow: Plow, map: GameMap, plows, reserved_positions=()):
    """Return the next move toward the closest depot."""
    if not DEPOT_POSITION:
        return None

    blocked_positions = {
        Position(position.x % map.width, position.y % map.height)
        for position in other_plow_positions(plow, plows)
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


def move_on_best_snow_loop(
    state: GameState,
    plow: Plow,
    reserved_positions=(),
    deadline=None,
):
    """Follow a short route that maximizes snow collected before unloading."""
    plan = [
        target for target in SNOW_PLANS.get(plow.id, [])
        if state.map.snow[target.y][target.x] > 0
    ]
    if not plan:
        plan = best_snow_loop(state, plow, deadline)
        if not plan:
            possible_moves = get_possible_move(plow, state.map)
            return random.choice(possible_moves) if possible_moves else None

    target = plan[0]
    if plow.position == target:
        SNOW_PLANS[plow.id] = plan
        return None

    next_move = astar_next_move(
        plow.position,
        target,
        state.map,
        other_plow_positions(plow, state.plows) + list(reserved_positions),
    )
    if next_move is None:
        SNOW_PLANS.pop(plow.id, None)
        return None
    SNOW_PLANS[plow.id] = plan
    return next_move


def depot_distance(position: Position, map: GameMap):
    targets = depot_unload_targets(map)
    if not targets:
        return map.width + map.height
    return min(
        toroidal_distance(position, target, map)
        for target in targets
    )


def near_depot(position: Position, map: GameMap):
    return depot_distance(position, map) <= DEPOT_NEARBY_DISTANCE


def best_snow_loop(state: GameState, plow: Plow, deadline=None):
    """Choose up to three snow stops with a reachable unloading tile."""
    remaining = SNOW_CAPACITY - plow.load
    if remaining <= 0 or not DEPOT_POSITION:
        return []

    blocked = {
        Position(other.position.x % state.map.width,
                 other.position.y % state.map.height)
        for other in blocking_plows(plow, state.plows)
    }
    assigned_snow = {
        target
        for plow_id, plan in SNOW_PLANS.items()
        if plow_id != plow.id
        for target in plan
    }
    candidates = [
        Position(x, y)
        for y in range(state.map.height)
        for x in range(state.map.width)
        if state.map.passable(x, y)
        and state.map.snow[y][x] > 0
        and Position(x, y) not in blocked
        and Position(x, y) not in assigned_snow
    ]
    candidates.sort(
        key=lambda target: (
            (
                depot_distance(target, state.map)
                if near_depot(plow.position, state.map)
                else 0
            ),
            -state.map.snow[target.y][target.x],
            toroidal_distance(plow.position, target, state.map),
        )
    )
    candidates = candidates[:MAX_SNOW_CANDIDATES]
    unload_targets = [
        target for target in depot_unload_targets(state.map)
        if target not in blocked
    ]
    paths = {}

    def path(start, target):
        key = (start, target)
        if key not in paths:
            paths[key] = shortest_path(
                start, target, state.map, blocked
            )
        return paths[key]

    best = None
    for stop_count in range(1, MAX_LOOP_STOPS + 1):
        for route in permutations(candidates, stop_count):
            if deadline is not None and time.perf_counter() >= deadline:
                return list(best[3]) if best is not None else []
            current = plow.position
            collected = 0
            travel = 0
            valid = True
            effective_route = []
            collected_positions = set()
            for target in route:
                segment = path(current, target)
                if segment is None:
                    valid = False
                    break
                travel += len(segment) - 1
                effective_route.append(target)
                for position in segment[1:]:
                    if position not in collected_positions:
                        collected_positions.add(position)
                        collected += min(
                            state.map.snow[position.y][position.x],
                            remaining - collected,
                        )
                current = target
                if collected == remaining:
                    break
            if not valid or collected == 0:
                continue

            depot_options = []
            for target in unload_targets:
                segment = path(current, target)
                if segment is not None:
                    extra_snow = sum(
                        min(
                            state.map.snow[position.y][position.x],
                            remaining - collected,
                        )
                        for position in segment[1:]
                        if position not in collected_positions
                    )
                    depot_options.append((len(segment) - 1, extra_snow))
            if not depot_options:
                continue

            return_distance, depot_snow = max(
                depot_options,
                key=lambda option: (
                    min(option[1], remaining - collected),
                    -option[0],
                ),
            )
            collected += min(depot_snow, remaining - collected)
            total_distance = travel + return_distance
            congestion = sum(
                toroidal_distance(target, other.position, state.map) <= 2
                for target in route
                for other in blocking_plows(plow, state.plows)
            )
            score = (
                collected * 100
                + (100 if collected == remaining else 0)
                - total_distance * 5
                - congestion * 20
            )
            candidate = (score, collected, -total_distance, tuple(effective_route))
            if best is None or candidate[:3] > best[:3]:
                best = candidate

    return list(best[3]) if best is not None else []


def depot_unload_targets(map: GameMap):
    targets = set()
    for depot_x, depot_y in DEPOT_POSITION:
        for target in (
            Position(depot_x + 1, depot_y),
            Position(depot_x - 1, depot_y),
            Position(depot_x, depot_y + 1),
            Position(depot_x, depot_y - 1),
        ):
            target = Position(target.x % map.width, target.y % map.height)
            if map.passable(target.x, target.y):
                targets.add(target)
    return targets


def shortest_path_length(start, target, map: GameMap, blocked_positions=()):
    path = shortest_path(start, target, map, blocked_positions)
    return None if path is None else max(0, len(path) - 1)


def shortest_path(start, target, map: GameMap, blocked_positions=()):
    start = Position(start.x % map.width, start.y % map.height)
    target = Position(target.x % map.width, target.y % map.height)
    result = find_path(
        start,
        target,
        neighbors_fnct=lambda position: astar_neighbors(
            position, map, blocked_positions
        ),
        heuristic_cost_estimate_fnct=lambda position, goal:
            toroidal_distance(position, goal, map),
    )
    return None if result is None else list(result)


def other_plow_positions(plow: Plow, plows):
    return [
        other.position
        for other in blocking_plows(plow, plows)
    ]


def blocking_plows(plow: Plow, plows):
    """Return active plows this plow must avoid."""
    return [
        other
        for other in plows
        if (
            other.id != plow.id
            and not other.broken
            and (
                plow.type != "big"
                or other.type == "big"
                or other.team == plow.team
            )
        )
    ]


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
