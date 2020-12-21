import logging
from itertools import product
from logging.config import fileConfig
from typing import TextIO, Optional

fileConfig("log.ini")

logger = logging.getLogger("dev")

right_rot = {0: 1, 1: 0, 2: 3, 3: 2}


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


def get_side_codes(tile: list[list[int]]) -> tuple[tuple[int], ...]:
    top: list[int] = tile[0]
    bottom: list[int] = tile[-1]
    left: list[int] = []
    right: list[int] = []
    for line in tile:
        left.append(line[0])
        right.append(line[-1])
    return tuple(top), tuple(right), tuple(bottom), tuple(left)


def find_adjacent(index: int, adjacent_list: list[tuple[int, int, bool]]) -> Optional[tuple[int, int, bool]]:
    for ad in adjacent_list:
        if ad[0] != index:
            return ad


def reverse_code(code: tuple[int]) -> tuple[int]:
    reversed_code = code[::-1]
    return reversed_code


def build_code_dict(tiles: dict[int, list[list[int]]]) -> dict[tuple[int], list[tuple[int, int, bool]]]:
    out: dict[tuple[int], list[tuple[int, int, bool]]] = {}
    tmp_list: list[tuple[int, int, bool]]
    for index, tile in tiles.items():
        top, right, bottom, left = get_side_codes(tile)

        tmp_list = out.get(top, [])
        tmp_list.append((index, 0, False))
        out[top] = tmp_list

        tmp_list = out.get(reverse_code(top), [])
        tmp_list.append((index, 0, True))
        out[reverse_code(top)] = tmp_list

        tmp_list = out.get(right, [])
        tmp_list.append((index, 1, False))
        out[right] = tmp_list

        tmp_list = out.get(reverse_code(right), [])
        tmp_list.append((index, 1, True))
        out[reverse_code(right)] = tmp_list

        tmp_list = out.get(bottom, [])
        tmp_list.append((index, 2, False))
        out[bottom] = tmp_list

        tmp_list = out.get(reverse_code(bottom), [])
        tmp_list.append((index, 2, True))
        out[reverse_code(bottom)] = tmp_list

        tmp_list = out.get(left, [])
        tmp_list.append((index, 3, False))
        out[left] = tmp_list

        tmp_list = out.get(reverse_code(left), [])
        tmp_list.append((index, 3, True))
        out[reverse_code(left)] = tmp_list

    return out


def classify_tiles(adjacent_tiles: dict[int, list[tuple[int, int, bool]]]) -> tuple[list[int], ...]:
    classified_tuple = ([], [], [])
    number_adjacents: int
    for index, adj_list in adjacent_tiles.items():
        number_adjacents = len(adj_list)
        if number_adjacents == 2:
            classified_tuple[0].append(index)
        elif number_adjacents == 3:
            classified_tuple[1].append(index)
        else:
            classified_tuple[2].append(index)
    return classified_tuple


def prod_list(input_list: list[int]):
    result: int = 1
    for el in input_list:
        result *= el
    return result


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


def get_adjacent_tiles(tiles: dict[int, list[list[int]]]) -> dict[int, list[tuple[int, int, bool]]]:
    adjacent_tiles: dict[int, list[tuple[int, int, bool]]] = {}
    tops: dict[int, tuple[int]] = {}
    bottoms: dict[int, tuple[int]] = {}
    rights: dict[int, tuple[int]] = {}
    lefts: dict[int, tuple[int]] = {}
    code_dict = build_code_dict(tiles)
    for index in tiles.keys():
        adjacent_tiles[index] = []
    for index, tile in tiles.items():
        top, right, bottom, left = get_side_codes(tile)
        tops[index] = top
        bottoms[index] = bottom
        rights[index] = right
        lefts[index] = left
    for index, code in tops.items():
        if find_adjacent(index, code_dict[code]):
            adjacent_tiles[index].append(find_adjacent(index, code_dict[code]))
    for index, code in rights.items():
        if find_adjacent(index, code_dict[code]):
            adjacent_tiles[index].append(find_adjacent(index, code_dict[code]))
    for index, code in bottoms.items():
        if find_adjacent(index, code_dict[code]):
            adjacent_tiles[index].append(find_adjacent(index, code_dict[code]))
    for index, code in lefts.items():
        if find_adjacent(index, code_dict[code]):
            adjacent_tiles[index].append(find_adjacent(index, code_dict[code]))
    return adjacent_tiles


def solution_part_1(filename: str) -> int:
    tiles = get_input_data(filename)
    adjacent_tiles = get_adjacent_tiles(tiles)
    logger.debug(adjacent_tiles)
    classified_tuple = classify_tiles(adjacent_tiles)
    logger.debug(classified_tuple)
    return prod_list(classified_tuple[0])


def flatten_adj_list(adjacent_tiles: dict[int, list[tuple[int, int, bool]]]) -> dict[int, list[int]]:
    flatten_list = {}
    for index, adj_list in adjacent_tiles.items():
        flatten_list[index] = [d[0] for d in adj_list]
    return flatten_list


def build_first_row(top_left_corner: int, flat_list: dict[int, list[int]], corners: list[int], edges: list[int]):
    row = [top_left_corner]
    prev_tile = top_left_corner
    current_tile = flat_list[top_left_corner][0]
    row.append(current_tile)
    while current_tile not in corners:
        for index in flat_list[current_tile]:
            if (index in corners or index in edges) and index != prev_tile:
                prev_tile = current_tile
                current_tile = index
                break
        row.append(current_tile)
    return row


def build_last_row(grid: list[list[int]], flat_list: dict[int, list[int]], corners: list[int], edges: list[int]):
    prev_tile = grid[-1][0]
    top_right_adj_list = flat_list[prev_tile].copy()
    top_right_adj_list.remove(grid[-2][0])
    current_tile = top_right_adj_list[0]
    col: int = 1
    grid[-1][col] = current_tile
    while current_tile not in corners:
        for index in flat_list[current_tile]:
            if (index in corners or index in edges) and index != prev_tile:
                prev_tile = current_tile
                current_tile = index
                break
        col += 1
        grid[-1][col] = current_tile


def build_left_row(width: int, grid: list[list[int]],
                   top_left_corner: int, flat_list: dict[int, list[int]],
                   corners: list[int], edges: list[int]):
    prev_tile = top_left_corner
    current_tile = flat_list[top_left_corner][1]
    grid.append([current_tile] + [-1 for _ in range(width - 1)])
    while current_tile not in corners:
        for index in flat_list[current_tile]:
            if (index in corners or index in edges) and index != prev_tile:
                prev_tile = current_tile
                current_tile = index
                break
        grid.append([current_tile] + [-1 for _ in range(width - 1)])


def build_right_row(grid: list[list[int]], flat_list: dict[int, list[int]], corners: list[int], edges: list[int]):
    prev_tile = grid[0][-1]
    top_right_adj_list = flat_list[prev_tile].copy()
    top_right_adj_list.remove(grid[0][-2])
    current_tile = top_right_adj_list[0]
    row: int = 1
    grid[row][-1] = current_tile
    while current_tile not in corners:
        for index in flat_list[current_tile]:
            if (index in corners or index in edges) and index != prev_tile:
                prev_tile = current_tile
                current_tile = index
                break
        row += 1
        grid[row][-1] = current_tile


def build_frame(adjacent_tiles: dict[int, list[tuple[int, int, bool]]]) -> list[list[int]]:
    classified_tuple = classify_tiles(adjacent_tiles)
    top_left_corner = classified_tuple[0][0]
    fl_list = flatten_adj_list(adjacent_tiles)
    grid: list[list[int]] = []
    # first row
    first_row = build_first_row(top_left_corner, fl_list, classified_tuple[0], classified_tuple[1])
    width: int = len(first_row)
    grid.append(first_row)
    # row left down
    build_left_row(width, grid, top_left_corner, fl_list, classified_tuple[0], classified_tuple[1])
    # row right down
    build_right_row(grid, fl_list, classified_tuple[0], classified_tuple[1])
    # bottom right down
    build_last_row(grid, fl_list, classified_tuple[0], classified_tuple[1])
    return grid


def get_next_tile(adjacent_tiles: dict[int, list[int]], left: int, top: int, top_left: int) -> int:
    for i1, i2 in product(adjacent_tiles[left], adjacent_tiles[top]):
        if i1 == i2 and i1 != top_left:
            return i1


def fill_row(grid: list[list[int]], adjacent_tiles: dict[int, list[tuple[int, int, bool]]]) -> list[list[int]]:
    height: int = len(grid)
    width: int = len(grid[0])
    flat_adj = flatten_adj_list(adjacent_tiles)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            grid[i][j] = get_next_tile(flat_adj, grid[i][j - 1], grid[i - 1][j], grid[i - 1][j - 1])
    return grid


def solution_part_2(filename):
    tiles = get_input_data(filename)
    adjacent_tiles = get_adjacent_tiles(tiles)
    grid = build_frame(adjacent_tiles)
    grid = fill_row(grid, adjacent_tiles)
    logger.debug(pprint_grid(grid))
    fine_grid = build_grid(tiles, grid)
    return find_monsters(fine_grid)


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


def get_orientation_right_down(index: int, adj_list_right: list[tuple[int, int, bool]],
                               adj_list_down: list[tuple[int, int, bool]]) -> tuple[int, int]:
    right_t: tuple[int, int, bool] = (0, 0, True)
    down_t: tuple[int, int, bool] = (0, 0, True)
    for t in adj_list_right:
        if t[0] == index:
            right_t = t
    for t in adj_list_down:
        if t[0] == index:
            down_t = t
    if (right_t[1] + 1) % 4 == down_t[1]:
        return right_rot[right_t[1]], 0
    else:
        return right_rot[right_t[1]], 1


def get_orientation_left_down(index: int, adj_list_left: list[tuple[int, int, bool]],
                              adj_list_down: list[tuple[int, int, bool]]) -> tuple[int, int]:
    rot, flip = get_orientation_right_down(index, adj_list_down, adj_list_left)
    if flip == 1:
        flip = 2
    return (rot + 1) % 4, flip


def get_orientation_right_up(index: int, adj_list_right: list[tuple[int, int, bool]],
                             adj_list_up: list[tuple[int, int, bool]]) -> tuple[int, int]:
    rot, flip = get_orientation_right_down(index, adj_list_up, adj_list_right)
    if flip == 1:
        flip = 2
    return (rot + 3) % 4, flip


def get_orientation_left_up(index: int, adj_list_left: list[tuple[int, int, bool]],
                            adj_list_up: list[tuple[int, int, bool]]) -> tuple[int, int]:
    rot, flip = get_orientation_right_down(index, adj_list_left, adj_list_up)
    return rot + 2 % 4, flip


def orient_tile(i: int, j: int, width: int, height: int, tiles: dict[int, list[list[int]]],
                grid: list[list[int]],
                adjacent_tiles: dict[int, list[tuple[int, int, bool]]]) -> list[list[int]]:
    if j < width - 1 and i < height - 1:
        rot, flip = get_orientation_right_down(grid[i][j], adjacent_tiles[grid[i][j + 1]],
                                               adjacent_tiles[grid[i + 1][j]])
    elif j == width - 1 and i < height - 1:
        rot, flip = get_orientation_left_down(grid[i][j], adjacent_tiles[grid[i][j - 1]],
                                              adjacent_tiles[grid[i + 1][j]])
    elif j < width - 1 and i == height - 1:
        rot, flip = get_orientation_right_up(grid[i][j], adjacent_tiles[grid[i][j + 1]],
                                             adjacent_tiles[grid[i - 1][j]])
    else:
        rot, flip = get_orientation_left_up(grid[i][j], adjacent_tiles[grid[i][j - 1]],
                                            adjacent_tiles[grid[i - 1][j]])
    logger.debug(f"Rot: {rot}, Flip: {flip}")
    rot_tile = rotate_tile(tiles[grid[i][j]], rot, flip)
    return rot_tile


def rotate_tile_90(tile: list[list[int]]) -> list[list[int]]:
    rot_tile: list[list[int]] = []
    for i in range(len(tile)):
        rot_tile.append([tile[j][i] for j in range(len(tile[0]) - 1, -1, -1)])
    return rot_tile


def flip_horizontal(tile: list[list[int]]) -> list[list[int]]:
    flip_tile: list[list[int]] = []
    for i in range(len(tile) - 1, -1, -1):
        flip_tile.append(tile[i])
    return flip_tile


def flip_vertical(tile: list[list[int]]) -> list[list[int]]:
    flip_tile: list[list[int]] = []
    for row in tile:
        flip_tile.append([row[i] for i in range(len(row) - 1, -1, -1)])
    return flip_tile


def rotate_tile(tile: list[list[int]], num: int, flip: int) -> list[list[int]]:
    rot_tile: list[list[int]] = tile
    for _ in range(num):
        rot_tile = rotate_tile_90(rot_tile)
    if flip == 1:
        rot_tile = flip_horizontal(rot_tile)
    elif flip == 2:
        rot_tile = flip_vertical(rot_tile)
    return rot_tile


def build_grid(tiles: dict[int, list[list[int]]], grid: list[list[int]]) -> list[list[int]]:
    result_grid: list[list[int]] = []
    height: int = len(grid)
    width: int = len(grid[0])
    tile_height: int = len(tiles.get(list(tiles.keys())[0]))
    tile_width: int = len(tiles.get(list(tiles.keys())[0])[0])
    adjacent_tiles = get_adjacent_tiles(tiles)
    for i, row in enumerate(grid):
        result_grid.extend([[] for _ in range(tile_height-2)])
        for k, index in enumerate(row):
            for j, tile_row in enumerate(orient_tile(i, k, width, height, tiles, grid, adjacent_tiles)):
                if j == 0 or j == tile_height - 1:
                    continue
                result_grid[i * (tile_height - 2) + (j - 1)].extend(tile_row[1: tile_width - 1])
    return result_grid


if __name__ == '__main__':
    logger.info(solution_part_1("inputData.txt"))
    logger.info(solution_part_2("inputData.txt"))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
