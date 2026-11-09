"""Votre bot. C'est le seul fichier que vous avez besoin de modifier.

decide() est appelée à chaque tour avec l'état du jeu et doit retourner vos ordres.
Pour l'instant, chaque déneigeuse se déplace au hasard.
"""

import heapq
import random

from models import Call, GameMap, GameState, Orders, Plow, Position

# Decision parameters
LAST_NORAML_BUY_TICK = 400
LAST_BIG_BUY_TICK = 750
MIN_UNLOAD_AMOUNT = 10
MIN_PLOWS_FOR_CALLS = 2

PLOW_DROPOFFS: dict[str, Position] = {}
PLOW_DROPOFF_BUILDINGS: dict[str, tuple[Position, ...]] = {}

def is_valid(pos: Position, game_map: GameMap) -> bool:
    """Return whether ``pos`` can be represented on ``game_map``."""
    return game_map.width > 0 and game_map.height > 0


def is_unblocked(
    pos: Position,
    state: GameState,
    moving_plow: Plow | None = None,
) -> bool:
    """Return whether a map cell is passable and not occupied by another plow."""
    if not state.map.passable(pos.x, pos.y):
        return False
    if moving_plow is not None and moving_plow.broken:
        return True

    for plow in state.plows:
        if plow is moving_plow or plow.position != pos or plow.broken:
            continue
        if (
            moving_plow is not None
            and moving_plow.type == "big"
            and moving_plow.team != plow.team
            and plow.type == "normal"
        ):
            continue
        return False
    return True


def calculate_h_value(pos: Position, target: Position, game_map: GameMap) -> int:
    """Return the Manhattan distance, accounting for the wrapping map edges."""
    dx = abs(pos.x - target.x)
    dy = abs(pos.y - target.y)
    return min(dx, game_map.width - dx) + min(dy, game_map.height - dy)


def get_dropoff_buildings(
    game_map: GameMap,
) -> list[tuple[tuple[Position, ...], list[Position]]]:
    """Return each 3x3 D building and its adjacent road dropoff tiles."""
    buildings: list[tuple[tuple[Position, ...], list[Position]]] = []
    visited: set[Position] = set()
    for y in range(game_map.height):
        for x in range(game_map.width):
            start = Position(x, y)
            if start in visited or game_map.tile(x, y) != "D":
                continue

            building: set[Position] = set()
            pending = [start]
            while pending:
                current = pending.pop()
                if current in building or game_map.tile(current.x, current.y) != "D":
                    continue
                building.add(current)
                visited.add(current)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    pending.append(Position(
                        (current.x + dx) % game_map.width,
                        (current.y + dy) % game_map.height,
                    ))

            if len(building) != 9:
                continue

            road_tiles: set[Position] = set()
            for tile in building:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = Position(
                        (tile.x + dx) % game_map.width,
                        (tile.y + dy) % game_map.height,
                    )
                    if neighbor not in building and game_map.tile(neighbor.x, neighbor.y) == ".":
                        road_tiles.add(neighbor)
            buildings.append((tuple(sorted(building, key=lambda tile: (tile.y, tile.x))), sorted(
                road_tiles, key=lambda tile: (tile.y, tile.x)
            )))

    return buildings


def get_dropoff_tiles(game_map: GameMap) -> set[Position]:
    """Return every road tile adjacent to a detected dropoff building."""
    return {
        tile
        for _, road_tiles in get_dropoff_buildings(game_map)
        for tile in road_tiles
    }


def trace_path(
    parents: dict[Position, Position | None],
    dest: Position,
) -> list[Position]:
    """Reconstruct a path from the parent map produced by A*."""
    if dest not in parents:
        return []

    path: list[Position] = []
    current: Position | None = dest
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path


def a_star_search(
    game_map: GameMap,
    src: Position,
    dest: Position,
    blocked: set[Position] | None = None,
) -> list[Position] | None:
    """Find a path to the destination or the closest reachable position."""
    if not is_valid(src, game_map) or not is_valid(dest, game_map):
        return None

    source = Position(src.x % game_map.width, src.y % game_map.height)
    target = Position(dest.x % game_map.width, dest.y % game_map.height)
    blocked = blocked or set()
    if not game_map.passable(source.x, source.y):
        return None
    if not game_map.passable(target.x, target.y):
        return None

    parents: dict[Position, Position | None] = {source: None}
    costs = {source: 0}
    closest = source
    closest_key = (calculate_h_value(source, target, game_map), 0)
    open_list: list[tuple[int, int, Position]] = []
    sequence = 0
    heapq.heappush(
        open_list,
        (calculate_h_value(source, target, game_map), sequence, source),
    )

    while open_list:
        _, _, current = heapq.heappop(open_list)
        if current == target:
            return trace_path(parents, target)

        current_key = (
            calculate_h_value(current, target, game_map),
            costs[current],
        )
        if current_key < closest_key:
            closest = current
            closest_key = current_key

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = Position(
                (current.x + dx) % game_map.width,
                (current.y + dy) % game_map.height,
            )
            if neighbor in blocked or not game_map.passable(neighbor.x, neighbor.y):
                continue

            new_cost = costs[current] + 1
            if new_cost >= costs.get(neighbor, float("inf")):
                continue

            costs[neighbor] = new_cost
            parents[neighbor] = current
            sequence += 1
            priority = new_cost + calculate_h_value(neighbor, target, game_map)
            heapq.heappush(open_list, (priority, sequence, neighbor))

    return trace_path(parents, closest)

def find_nearest_snow(plow: Plow, game_map: GameMap) -> Position | None:
    dropoff_centers = (Position(11, 11), Position(31, 11))
    nearest_dropoff = min(
        dropoff_centers,
        key=lambda dropoff: calculate_h_value(plow.position, dropoff, game_map),
    )

    nearest = None
    min_combined_distance = float("inf")
    for y in range(game_map.height):
        for x in range(game_map.width):
            if game_map.snow[y][x] <= 1 or not game_map.passable(x, y):
                continue

            snow_tile = Position(x, y)
            combined_distance = (
                calculate_h_value(plow.position, snow_tile, game_map)
                + calculate_h_value(snow_tile, nearest_dropoff, game_map)
            )
            if combined_distance < min_combined_distance:
                min_combined_distance = combined_distance
                nearest = snow_tile

    return nearest

def find_nearest_dropoff(plow: Plow, state: GameState) -> Position | None:
    buildings = get_dropoff_buildings(state.map)
    if not buildings:
        return None

    building_map = {building: road_tiles for building, road_tiles in buildings}
    assigned_building = PLOW_DROPOFF_BUILDINGS.get(plow.id)
    if assigned_building not in building_map:
        assigned_building = None

    if assigned_building is None:
        assigned_counts = {
            building: sum(
                PLOW_DROPOFF_BUILDINGS.get(other.id) == building
                for other in state.my_plows
            )
            for building in building_map
        }
        assigned_building = min(
            building_map,
            key=lambda building: (
                assigned_counts[building],
                min(
                    calculate_h_value(plow.position, tile, state.map)
                    for tile in building_map[building]
                ),
            ),
        )
        PLOW_DROPOFF_BUILDINGS[plow.id] = assigned_building

    dropoff_tiles = building_map[assigned_building]
    if not dropoff_tiles:
        return None

    assigned_dropoff = min(
        dropoff_tiles,
        key=lambda tile: calculate_h_value(plow.position, tile, state.map),
    )
    PLOW_DROPOFFS[plow.id] = assigned_dropoff
    return assigned_dropoff

# TODO: implement weights for calls based on reward, distance, and expiration time
def find_nearest_call(plow: Plow, state: GameState) -> Call:
    nearest = None
    min_distance = float('inf')

    for call in state.calls:
        distance = calculate_h_value(plow.position, call.position, state.map)
        if call.expires_at - state.tick < distance:
            continue
        if distance < min_distance:
            min_distance = distance
            nearest = call
    return nearest

def navigate(
    plow: Plow,
    target: Position,
    game_state: GameState,
    reserved: set[Position] | None = None,
) -> str | None:
    """Retourne la direction à prendre pour aller vers target.

    Renvoie "N", "S", "E", "O" ou None si aucune direction n'est possible.
    """

    blocked = {
        other.position
        for other in game_state.plows
        if other is not plow
        and not is_unblocked(other.position, game_state, plow)
    }
    blocked.update(reserved or ())
    path = a_star_search(game_state.map, plow.position, target, blocked)
    if path and len(path) >= 2:
        current, next_position = path[0], path[1]
    elif plow.position == target:
        return None
    else:
        candidates = []
        for direction, (dx, dy) in (
            ("E", (1, 0)),
            ("O", (-1, 0)),
            ("S", (0, 1)),
            ("N", (0, -1)),
        ):
            destination = Position(
                (plow.position.x + dx) % game_state.map.width,
                (plow.position.y + dy) % game_state.map.height,
            )
            if (
                destination not in blocked
                and game_state.map.passable(destination.x, destination.y)
            ):
                distance = calculate_h_value(destination, target, game_state.map)
                candidates.append((distance, direction))
        if not candidates:
            return None
        return min(candidates)[1]

    if (next_position.x - current.x) % game_state.map.width == 1:
        return "E"
    if (current.x - next_position.x) % game_state.map.width == 1:
        return "O"
    if (next_position.y - current.y) % game_state.map.height == 1:
        return "S"
    return "N"


def decide(state: GameState) -> Orders:
    orders = Orders()
    state.money

    # Plow buying strategy
    if state.tick <= LAST_NORAML_BUY_TICK and state.money >= 250:
        orders.buy = "normal"
    elif state.tick <= LAST_BIG_BUY_TICK and state.money >= 400:
        orders.buy = "big"
    else:
        orders.buy = None
    #===================================================
    
    # Plows movement strategy
    plow: Plow
    reserved: set[Position] = set()
    dropoff_tiles = get_dropoff_tiles(state.map)
    for plow in state.my_plows:

        if plow.broken:
            target_position = state.garage
        elif plow.load >= 20:
            target_position = find_nearest_dropoff(plow, state)
            if plow.position in dropoff_tiles:
                orders.unload(plow)
                continue
        elif plow.load > MIN_UNLOAD_AMOUNT and plow.position in dropoff_tiles:
            orders.unload(plow)
            continue
        elif len(state.my_plows) >= MIN_PLOWS_FOR_CALLS and len(state.calls) > 0:
            target_position = find_nearest_call(plow, state)
            if target_position is None:
                target_position = find_nearest_snow(plow, state.map)
            else:
                target_position = target_position.position
        else:
            target_position = find_nearest_snow(plow, state.map)
            if target_position is None:
                continue

        movement = navigate(plow, target_position, state, reserved)
        if movement is not None:
            dx, dy = {
                "E": (1, 0),
                "O": (-1, 0),
                "S": (0, 1),
                "N": (0, -1),
            }[movement]
            destination = Position(
                (plow.position.x + dx) % state.map.width,
                (plow.position.y + dy) % state.map.height,
            )
            if destination not in reserved:
                reserved.add(destination)
                orders.move(plow, movement)
    
    #===================================================
    
    return orders