"""Independent fleet bot with one explicit target state per plow."""

import time
import math
from astar import find_path
from models import GameState, Orders, Plow, Position

CAPACITY = 20
RETURN_LOAD = 12
NORMAL_COST = 250
BIG_COST = 400
DECISION_LIMIT = 0.32
MAX_CYCLE_CANDIDATES = 10

targets = {}
target_kind = {}
depots = ()


def decide(state: GameState) -> Orders:
    global depots
    started = time.perf_counter()
    if not depots:
        depots = tuple(
            Position(x, y)
            for y in range(state.map.height)
            for x in range(state.map.width)
            if state.map.tile(x, y) == "D"
        )

    orders = Orders()
    reserved = set()
    fast_mode = len(state.my_plows) > 7
    assignments = assign_targets(state, fast_mode)
    purchase = purchase_order(state, started)
    if purchase:
        orders.buy = purchase

    for plow in state.my_plows:
        out_of_time = time.perf_counter() - started >= DECISION_LIMIT
        if plow.broken:
            target = state.garage
            kind = "garage"
        elif target_kind.get(plow.id) == "depot":
            target = keep_or_choose_depot(plow, state)
            kind = "depot"
        elif plow.load >= RETURN_LOAD:
            target = keep_or_choose_depot(plow, state)
            kind = "depot"
        else:
            target = assignments.get(plow.id)
            if target is None:
                target = keep_or_choose_snow(plow, state, fast_mode)
            kind = "work"

        if target is None:
            continue
        targets[plow.id] = target
        target_kind[plow.id] = kind

        if kind == "depot" and adjacent_depot(plow.position, state):
            targets.pop(plow.id, None)
            target_kind.pop(plow.id, None)
            orders.unload(plow)
            continue

        blocked = occupied(plow, state) | reserved
        move = (
            direct_move(plow.position, target, state, blocked)
            if out_of_time
            else route_move(plow.position, target, state, blocked)
        )
        if move is None:
            continue
        destination = destination_of(plow.position, move, state)
        if destination in reserved:
            continue
        orders.move(plow, move)
        reserved.add(destination)

    return orders


def assign_targets(state, fast_mode=False):
    calls = [
        call for call in state.calls
        if call.expires_at > state.tick
        and snow_at(state, call.position) > 0
    ]
    pairs = []
    for plow in state.my_plows:
        if plow.broken or plow.load >= CAPACITY:
            continue
        for call in calls:
            if call.position == plow.position:
                continue
            distance = distance_between(plow.position, call.position, state)
            if distance < call.expires_at - state.tick:
                pairs.append((call.reward * 10 - distance, plow.id, call.position))

    assignments = {}
    used_plows = set()
    call_counts = {call.id: 0 for call in calls}
    call_slots = {
        call.id: max(1, math.ceil(snow_at(state, call.position) / CAPACITY))
        for call in calls
    }
    for _, plow_id, target in sorted(
        pairs,
        key=lambda pair: (pair[0], pair[1], pair[2].y, pair[2].x),
        reverse=True,
    ):
        call = next(call for call in calls if call.position == target)
        if (
            plow_id not in used_plows
            and call_counts[call.id] < call_slots[call.id]
        ):
            assignments[plow_id] = target
            used_plows.add(plow_id)
            call_counts[call.id] += 1

    snow_tiles = [
        Position(x, y)
        for y in range(state.map.height)
        for x in range(state.map.width)
        if state.map.passable(x, y) and state.map.snow[y][x] > 0
    ]
    available = [
        plow for plow in state.my_plows
        if not plow.broken
        and plow.load < CAPACITY
        and plow.id not in assignments
    ]
    for plow in sorted(available, key=lambda item: item.load):
        choices = [
            tile for tile in snow_tiles
            if tile != plow.position
            and tile not in assignments.values()
        ]
        if not choices:
            continue
        target = select_snow_target(plow, choices, state, fast_mode)
        assignments[plow.id] = target
    return assignments


def keep_or_choose_snow(plow, state, fast_mode=False):
    old = targets.get(plow.id)
    if (
        target_kind.get(plow.id) == "work"
        and old is not None
        and old != plow.position
        and state.map.passable(old.x, old.y)
        and snow_at(state, old) > 0
    ):
        return old
    choices = [
        Position(x, y)
        for y in range(state.map.height)
        for x in range(state.map.width)
        if state.map.passable(x, y)
        and state.map.snow[y][x] > 0
        and Position(x, y) != plow.position
    ]
    return select_snow_target(plow, choices, state, fast_mode)


def select_snow_target(plow, choices, state, fast_mode=False):
    if fast_mode:
        return min(
            choices,
            key=lambda tile: (
                distance_between(plow.position, tile, state)
                - snow_at(state, tile) * 2,
                tile.y,
                tile.x,
            ),
            default=None,
        )
    dropoffs = dropoff_tiles(state)
    choices = sorted(
        choices,
        key=lambda tile: (
            distance_between(plow.position, tile, state)
            + min(
                (distance_between(tile, dropoff, state) for dropoff in dropoffs),
                default=0,
            )
            - snow_at(state, tile) * 2,
        ),
    )[:MAX_CYCLE_CANDIDATES]
    return max(
        choices,
        key=lambda tile: cycle_value(plow, tile, state),
        default=None,
    )


def cycle_value(plow, target, state):
    outbound = path_to(plow.position, target, state, set())
    if len(outbound) < 2:
        return (-float("inf"), 0, 0, 0)
    dropoffs = dropoff_tiles(state)
    return_path = []
    shortest_return = float("inf")
    for dropoff in dropoffs:
        if distance_between(target, dropoff, state) >= shortest_return:
            continue
        path = path_to(target, dropoff, state, set())
        if path and len(path) < shortest_return:
            return_path = path
            shortest_return = len(path)
    if not return_path:
        return (-float("inf"), 0, 0, 0)
    route = outbound + return_path[1:]
    available = CAPACITY - plow.load
    collected = 0
    visited = set()
    for position in route[1:]:
        if position in visited:
            continue
        visited.add(position)
        collected += min(
            snow_at(state, position),
            max(0, available - collected),
        )
        if collected >= available:
            break
    travel = len(outbound) + len(return_path) - 2
    density = collected / max(1, travel)
    return (density, collected, -travel, -distance_to_depot(target, state))


def dropoff_tiles(state):
    return tuple(
        tile
        for depot in depots
        for tile in neighbors_of(depot, state)
        if passable_dropoff(tile, state)
    )


def keep_or_choose_depot(plow, state):
    old = targets.get(plow.id)
    if (
        target_kind.get(plow.id) == "depot"
        and old is not None
        and passable_dropoff(old, state)
    ):
        return old
    choices = [
        tile
        for depot in depots
        for tile in neighbors_of(depot, state)
        if passable_dropoff(tile, state)
    ]
    return min(
        choices,
        key=lambda tile: (distance_between(plow.position, tile, state), tile.y, tile.x),
        default=None,
    )


def route_move(start, target, state, blocked):
    if start == target:
        return None
    blocked = set(blocked)
    path = path_to(start, target, state, blocked - {target})
    if len(path) >= 2:
        return direction(start, path[1], state)

    # Dynamic blockers can make an otherwise valid route temporarily absent.
    # Replan on the static map and choose a legal first step only.
    path = path_to(start, target, state, set())
    if len(path) >= 2:
        candidate = path[1]
        if candidate not in blocked:
            return direction(start, candidate, state)

    choices = [
        tile for tile in neighbors_of(start, state)
        if state.map.passable(tile.x, tile.y) and tile not in blocked
    ]
    if not choices:
        return None
    choice = min(
        choices,
        key=lambda tile: (distance_between(tile, target, state), tile.y, tile.x),
    )
    return direction(start, choice, state)


def direct_move(start, target, state, blocked):
    choices = [
        tile for tile in neighbors_of(start, state)
        if state.map.passable(tile.x, tile.y) and tile not in blocked
    ]
    if not choices:
        return None
    choice = min(
        choices,
        key=lambda tile: (
            distance_between(tile, target, state),
            tile.y,
            tile.x,
        ),
    )
    return direction(start, choice, state)


def path_to(start, target, state, blocked):
    path = find_path(
        start,
        target,
        neighbors_fnct=lambda pos: (
            tile for tile in neighbors_of(pos, state)
            if state.map.passable(tile.x, tile.y) and tile not in blocked
        ),
        heuristic_cost_estimate_fnct=lambda pos, goal: distance_between(pos, goal, state),
    )
    return list(path) if path is not None else []


def purchase_order(state, started):
    if time.perf_counter() - started > DECISION_LIMIT / 2:
        return None
    if state.tick <= 400 and state.money >= NORMAL_COST:
        return "normal"
    if state.tick <= 750 and state.money >= BIG_COST:
        return "big"
    return None


def occupied(plow, state):
    if plow.broken:
        return set()
    return {
        other.position for other in state.plows
        if other.id != plow.id and not other.broken
        and (plow.type != "big" or other.type == "big" or other.team == plow.team)
    }


def passable_dropoff(position, state):
    return state.map.passable(position.x, position.y) and adjacent_depot(position, state)


def adjacent_depot(position, state):
    return any(state.map.tile(tile.x, tile.y) == "D" for tile in neighbors_of(position, state))


def neighbors_of(position, state):
    return tuple(
        Position((position.x + dx) % state.map.width, (position.y + dy) % state.map.height)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
    )


def destination_of(position, move, state):
    dx, dy = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "O": (-1, 0)}[move]
    return Position(
        (position.x + dx) % state.map.width,
        (position.y + dy) % state.map.height,
    )


def direction(start, end, state):
    dx = (end.x - start.x) % state.map.width
    dy = (end.y - start.y) % state.map.height
    if dx == 1:
        return "E"
    if dx == state.map.width - 1:
        return "O"
    if dy == 1:
        return "S"
    return "N"


def distance_between(first, second, state):
    dx = abs(first.x - second.x)
    dy = abs(first.y - second.y)
    return min(dx, state.map.width - dx) + min(dy, state.map.height - dy)


def distance_to_depot(position, state):
    return min(
        (distance_between(position, depot, state) for depot in depots),
        default=0,
    )


def snow_at(state, position):
    return state.map.snow[position.y][position.x]
