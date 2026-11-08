"""Votre bot. C'est le seul fichier que vous avez besoin de modifier.

decide() est appelée à chaque tour avec l'état du jeu et doit retourner vos ordres.
Pour l'instant, chaque déneigeuse se déplace au hasard.
"""

import random
import numpy as np

from models import GameState, Orders, GameMap, Call, Plow, Position


DEPOT_POSITION = np.array([(10, 10), (10, 11), (10, 12), (10, 30), (10, 31), (10, 32), (11, 10), (11, 11), (11, 12), (11, 30), (11, 31), (11, 32), (12, 10), (12, 11), (12, 12), (12, 30), (12, 31), (12, 32)])

def decide(state: GameState) -> Orders:
    orders = Orders()
    for plow in state.my_plows:
        orders.move(plow, random.choice(get_possible_move(plow, state.map)))
        if next_to_depot(plow, state.map):
            orders.unload(plow)
        print(next_to_depot(plow, state.map))
    return orders


def next_to_depot(plow: Plow, map: GameMap):
    # Print Depot position for hard code
    # for i in range(map.height):
    #     for j in range(map.width):
    #         if map.tile(i, j) == "D":
    #             print("X: ", i, " Y: ", j)
    result = False
    if map.tile(plow.position.x, plow.position.y+1) == "D": result = True
    if map.tile(plow.position.x, plow.position.y-1) == "D": result = True
    if map.tile(plow.position.x+1, plow.position.y) == "D": result = True
    if map.tile(plow.position.x-1, plow.position.y) == "D": result = True
    return result

def get_possible_move(plow: Plow, map: GameMap):
    move = []
    if map.tile(plow.position.x, plow.position.y+1) == ".": move.append("N")
    if map.tile(plow.position.x, plow.position.y-1) == ".": move.append("S")
    if map.tile(plow.position.x+1, plow.position.y) == ".": move.append("E")
    if map.tile(plow.position.x-1, plow.position.y) == ".": move.append("O")

    return move
