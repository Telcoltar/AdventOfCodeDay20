import logging
from itertools import chain
from logging.config import fileConfig
from typing import TextIO

fileConfig("log.ini")

logger = logging.getLogger("dev")


def pprint_grid(grid: list[list[int]]) -> str:
    result: str = "\n"
    for row in grid[:-1]:
        for el in row:
            result += str(el)
            result += " "
        result += "\n"
    for el in grid[-1]:
        result += str(el)
        result += " "
    return result


def get_input_data(filename: str) -> dict[int, list[list[int]]]:
    f: TextIO = open(filename)

    tiles: dict[int, list[list[int]]] = {}
    current_tile: list[list[int]] = []
    current_line: list[int]
    current_tile_index: int = 0
    is_new_tile: bool = True

    for line in f.readlines():
        if is_new_tile:
            current_tile_index = int(line.split(" ")[1].replace(":", "").strip())
            is_new_tile = False
            continue
        if line.strip() == "":
            tiles[current_tile_index] = current_tile
            current_tile = []
            is_new_tile = True
            continue
        current_line = []
        for point in line.strip():
            if point == "#":
                current_line.append(1)
            else:
                current_line.append(0)
        current_tile.append(current_line)
    tiles[current_tile_index] = current_tile
    f.close()
    return tiles


def get_edge_codes(tile: list[list[int]]) -> list[tuple[int]]:
    return [tuple(tile[0]),
            tuple([tile[i][-1] for i in range(len(tile[-1]))]),
            tuple(tile[-1]),
            tuple([tile[i][0] for i in range(len(tile[0]))])]


def build_edge_to_tile(tiles: dict[int, list[list[int]]]) -> dict[tuple[int], list[int]]:
    edge_to_tile_dict = {}
    for num, tile in tiles.items():
        for edge in get_edge_codes(tile):
            if edge not in edge_to_tile_dict:
                edge_to_tile_dict[edge] = []
                edge_to_tile_dict[tuple(reversed(edge))] = []
            edge_to_tile_dict[edge].append(num)
            edge_to_tile_dict[tuple(reversed(edge))].append(num)
    return edge_to_tile_dict


def classify_tiles(tiles: dict[int, list[list[int]]]) -> tuple[list[int], list[int], list[int]]:
    classfified = ([], [], [])
    edge_tile_dict = build_edge_to_tile(tiles)
    for num, tile in tiles.items():
        outer_edges = [len(edge_tile_dict[code]) for code in get_edge_codes(tile)].count(1)
        classfified[outer_edges].append(num)
    return classfified


def reduce_tiles(tiles: list[int], num: int):
    red_tiles = tiles.copy()
    if len(tiles) == 1:
        return -1
    red_tiles.remove(num)
    return red_tiles[0]


def build_neigh_dict(tiles: dict[int, list[list[int]]]) -> dict[int, list[int]]:
    edge_tile_dict = build_edge_to_tile(tiles)
    neighbors_dict = {}
    for num, tile in tiles.items():
        neighbors_dict[num] = [reduce_tiles(edge_tile_dict[code], num) for code in get_edge_codes(tile)]
    return neighbors_dict


def orient_tile(neighbors: list[int], constrains: list[tuple[int, int]]) -> tuple[int, int]:
    logger.debug(neighbors)
    logger.debug(constrains)
    for i in range(4):
        if all(neighbors[c[0]] == c[1] for c in constrains):
            return i, 0
        neighbors.insert(0, neighbors.pop(-1))

    tmp = neighbors[1]
    neighbors[1] = neighbors[3]
    neighbors[3] = tmp

    for i in range(4):
        if all(neighbors[c[0]] == c[1] for c in constrains):
            return i, 1
        neighbors.insert(0, neighbors.pop(-1))


def build_grid_new(tiles) -> list[list[tuple[int, tuple[int, int]]]]:
    middles, edges, corners = classify_tiles(tiles)
    neighbors_dict = build_neigh_dict(tiles)
    start_edge = corners[0]
    neigb_tiles = neighbors_dict[start_edge]
    prev_tile = start_edge
    grid: list[list[tuple[int, tuple[int, int]]]] = [[(start_edge, orient_tile(neigb_tiles, [(0, -1), (3, -1)]))]]
    current_tile = neigb_tiles[1]
    while current_tile != -1:
        neigb_tiles = neighbors_dict[current_tile]
        grid[0].append((current_tile, orient_tile(neigb_tiles, [(0, -1), (3, prev_tile)])))
        prev_tile = current_tile
        current_tile = neigb_tiles[1]
    size = len(grid[0])
    for i in range(1, size):
        j = 0
        grid.append([])
        prev_tile = -1
        current_tile = neighbors_dict[grid[i - 1][0][0]][2]
        while current_tile != -1:
            neigb_tiles = neighbors_dict[current_tile]
            grid[i].append((current_tile, orient_tile(neigb_tiles, [(0, grid[i - 1][j][0]), (3, prev_tile)])))
            prev_tile = current_tile
            current_tile = neigb_tiles[1]
            j += 1
    return grid


def get_sea_monster() -> list[list[int]]:
    f: TextIO = open("sea_monster.txt")

    monster: list[list[int]] = []
    current_line: list[int]

    for line in f.readlines():
        current_line = []
        for symb in line.replace("\n", ""):
            if symb == "#":
                current_line.append(1)
            else:
                current_line.append(0)
        monster.append(current_line)

    max_line_length: int = max([len(line) for line in monster])

    for line in monster:
        line.extend([0 for _ in range(max_line_length - len(line))])

    f.close()

    return monster


def prod_list(input_list: list[int]):
    result: int = 1
    for el in input_list:
        result *= el
    return result


def find_monsters(grid: list[list[int]]) -> int:
    sea_monster: list[list[int]] = get_sea_monster()
    sea_m_height: int = len(sea_monster)
    sea_m_width: int = len(sea_monster[0])
    grid_height: int = len(grid)
    grid_width: int = len(grid[0])
    sea_monter_list: list[tuple[int, int]] = compile_sea_monster_list(sea_monster)
    count: int = 0
    logger.debug(f"Monster: {sea_m_height}, {sea_m_width}")
    logger.debug(f"Grid: {grid_height}, {grid_width}")
    for _ in range(4):
        count = search_loop(grid_width - sea_m_width, grid_height - sea_m_height, grid, sea_monter_list)
        if count == 0:
            grid = rotate_tile_90(grid)
        else:
            break
    if count == 0:
        grid = flip_vertical(grid)
        for _ in range(4):
            count = search_loop(grid_width - sea_m_width, grid_height - sea_m_height, grid, sea_monter_list)
            if count == 0:
                grid = rotate_tile_90(grid)
            else:
                break
    logger.debug(count)
    logger.debug(sea_monter_list)
    grid = replace_sea_monsters(grid_width - sea_m_width, grid_height - sea_m_height, grid, sea_monter_list)
    logger.debug(sum_list(grid))
    return sum_list(grid)


def sum_list(grid: list[list[int]]) -> int:
    count: int = 0
    for line in grid:
        count += sum(line)
    return count


def replace_sea_monsters(width: int, height: int, grid: list[list[int]], sea_monster_list: list[tuple[int, int]]):
    for i in range(width):
        for j in range(1, height - 1):
            if grid[j][i] == 1 and is_monster(i, j, grid, sea_monster_list):
                for n, m in sea_monster_list:
                    grid[n+j-1][m+i] = 0
    return grid


def search_loop(width: int, height: int, grid: list[list[int]], sea_monster_list: list[tuple[int, int]]) -> int:
    count: int = 0
    for i in range(width):
        for j in range(1, height - 1):
            if grid[j][i] == 1 and is_monster(i, j, grid, sea_monster_list):
                count += 1
    return count


def is_monster(i: int, j: int, grid: list[list[int]], monster: list[tuple[int, int]]) -> bool:
    for n, m in monster:
        if grid[n+j-1][m+i] != 1:
            return False
    return True


def compile_sea_monster_list(monster: list[list[int]]) -> list[tuple[int, int]]:
    list_of_ones = []
    for i, line in enumerate(monster):
        for j, symb in enumerate(line):
            if symb:
                list_of_ones.append((i, j))
    return list_of_ones


def rotate_tile_90(tile: list[list[int]]) -> list[list[int]]:
    rot_tile: list[list[int]] = []
    for i in range(len(tile)):
        rot_tile.append([tile[j][i] for j in range(len(tile[0]) - 1, -1, -1)])
    return rot_tile


def flip_vertical(tile: list[list[int]]) -> list[list[int]]:
    flip_tile: list[list[int]] = []
    for row in tile:
        flip_tile.append([row[i] for i in range(len(row) - 1, -1, -1)])
    return flip_tile


def orient_tiles(tiles: dict[int, list[list[int]]], grid: list[list[tuple[int, tuple[int, int]]]]):
    for num, orientation in chain(*grid):
        if orientation[1]:
            tiles[num] = flip_vertical(tiles[num])
        for _ in range(orientation[0]):
            tiles[num] = rotate_tile_90(tiles[num])


def solution_part_1(filename: str):
    tiles = get_input_data(filename)
    _, _, corners = classify_tiles(tiles)
    return prod_list(corners)


def solution_part_2(filename: str) -> int:
    tiles = get_input_data(filename)
    grid = build_grid_new(tiles)
    orient_tiles(tiles, grid)
    tile_height: int = len(tiles.get(list(tiles.keys())[0]))
    tile_width: int = len(tiles.get(list(tiles.keys())[0])[0])
    result_grid: list[list[int]] = []
    for i, row in enumerate(grid):
        result_grid.extend([[] for _ in range(tile_height - 2)])
        for index in row:
            for j, tile_row in enumerate(tiles[index[0]]):
                if j == 0 or j == tile_height - 1:
                    continue
                result_grid[i * (tile_height - 2) + (j - 1)].extend(tile_row[1: tile_width - 1])
    return find_monsters(result_grid)


if __name__ == '__main__':
    logger.info(solution_part_1("inputData.txt"))
    logger.info(solution_part_2("inputData.txt"))
