import random

import numpy as np
from pyboy import PyBoy, WindowEvent
from geneal.genetic_algorithms import ContinuousGenAlgSolver
from geneal.applications.fitness_functions.continuous import fitness_functions_continuous

pyboy = PyBoy('Tetris.gb', game_wrapper=True)
tetris = pyboy.game_wrapper()
tetris.start_game()
timestep = 0

# Actions for the AI to use
action_map = {
    'Left': [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
    'Right': [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
    'Down': [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
    'A': [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A]
}

start_y = 24


def action_turn():
    pyboy.send_input(action_map['A'][0])
    pyboy.tick()
    pyboy.send_input(action_map['A'][1])
    pyboy.tick()


def action_translate(direction):
    pyboy.send_input(action_map[direction][0])
    pyboy.tick()
    pyboy.send_input(action_map[direction][1])
    pyboy.tick()


def action_down():
    pyboy.send_input(action_map['Down'][0])
    pyboy.tick()
    pyboy.send_input(action_map['Down'][1])


def drop_down():
    started_moving = False
    while pyboy.get_memory_value(0xc201) != start_y or not started_moving:
        started_moving = True
        action_down()

    global timestep
    timestep += 1


# go n_dir spaces in direction, and rotate n_turn times
def action(translate, turn):
    # if translate is negative, move left. Else, right

    for _ in range(abs(translate)):
        if translate < 0:
            action_translate("Left")
        else:
            action_translate("Right")

    for _ in range(turn):
        action_turn()

    drop_down()


def getScore():
    return tetris.score


def getCurrentTetromino():
    tetromino = pyboy.get_memory_value(0xc203)
    if 0 <= tetromino <= 3:
        return 'L'
    elif 4 <= tetromino <= 7:
        return 'J'
    elif 8 <= tetromino <= 11:
        return 'I'
    elif 12 <= tetromino <= 15:
        return 'O'
    elif 16 <= tetromino <= 19:
        return 'Z'
    elif 20 <= tetromino <= 23:
        return 'S'
    elif 24 <= tetromino <= 27:
        return 'T'


def getStartCoords(tetromino):
    coords = []
    if tetromino == 'I':
        coords = [[1, 3], [1, 4], [1, 5], [1, 6]]
    elif tetromino == "Z":
        coords = [[1, 3], [1, 4], [2, 4], [2, 5]]
    elif tetromino == "S":
        coords = [[2, 3], [2, 4], [1, 4], [1, 5]]
    elif tetromino == "T":
        coords = [[1, 3], [1, 4], [1, 5], [2, 4]]
    elif tetromino == "O":
        coords = [[1, 4], [1, 5], [2, 4], [2, 5]]
    elif tetromino == "J":
        coords = [[1, 3], [1, 4], [1, 5], [2, 5]]
    elif tetromino == "L":
        coords = [[1, 3], [1, 4], [1, 5], [2, 3]]

    return coords


def getNextTetromino():
    return tetris.next_tetromino()


def getLinesCleared():
    return tetris.lines


def getLevel():
    return tetris.level


def calculateReward(board, w1, w2, w3, w4):
    reward = w1 * getCompleteLines(board)
    if reward > 0:
        holes = w2 * getHoles(board)
        aggregateHeight = w3 * getAggregateHeight(board)
        bumpiness = w4 * getBumpiness(board)

        multiplier = 1
        multiplier += aggregateHeight / 10
        multiplier += 1 + bumpiness / 10

        if holes > 0:
            multiplier += holes / 10

        reward = reward * multiplier

    # calculate sub-rewards
    lines = 0
    count = 0
    for row in range(len(board)):
        flag = False
        for column in range(len(board[0])):
            if board[row][column] != 0:
                if flag is False:
                    flag = True
                    lines += 1
                count += 1

    reward += (count / lines) / 10

    return reward


def getCurrentBoard():
    return np.asarray(tetris.game_area()).astype(int) - 47


# 1
# 17
def get_predicted_board(translate, turn):
    board = getCurrentBoard()
    current = getCurrentTetromino()
    num = -1

    # print(current, translate, turn)

    coords = getStartCoords(current)

    for row in range(len(coords)):
        board[coords[row][0], coords[row][1]] = 0

    for column in range(len(board[2])):
        if board[2][column] != 0:
            # print("Prediction failure.")
            return False

    while translate != 0:
        if coords[0][1] + translate < 0:
            translate += 1
        elif coords[3][1] + translate > 9:
            translate -= 1
        else:
            for row in range(0, len(coords)):
                coords[row][1] += translate
                # coords[i][0] += 1
            translate = 0

    if current != 'O':
        if current == 'I' or current == "S" or current == "Z":
            turn = turn % 2
        while turn != 0:
            tempCoords = np.add(coords, rotation[''.join([current, str(turn)])])

            if all([x > 9 for x in tempCoords[:][1]]):
                turn = turn - 1
            else:
                coords = tempCoords
                turn = 0

    drop = True
    while drop:
        for row in range(len(coords)):
            if coords[row][0] > 16 or board[coords[row][0] + 1][coords[row][1]] != 0:
                drop = False
                break

        if drop:
            for row in range(len(coords)):
                coords[row][0] += 1

    for row in range(len(coords)):
        board[coords[row][0], coords[row][1]] = num

    # print(board)
    return board


rotation = {
    'I1': [[1, 1], [0, 0], [-1, -1], [-2, -2]],
    'T1': [[-1, 1], [0, 0], [1, -1], [-1, -1]],
    'T2': [[0, 2], [0, 0], [0, -2], [-2, 0]],
    'T3': [[1, 1], [0, 0], [-1, -1], [-1, 1]],
    'T4': [[0, 0], [0, 0], [0, 0], [0, 0]],
    'S1': [[1, 1], [0, 0], [1, -1], [0, -2]],
    'Z1': [[0, 1], [-1, 0], [0, -1], [-1, -2]],  # [[-1, 1], [0, 0], [-1, -1], [0, -2]],
    'O': [[0, 0], [0, 0], [0, 0], [0, 0]],
    'J1': [[-1, 1], [0, 0], [1, -1], [0, -2]],
    'J2': [[0, 2], [0, 0], [0, -2], [-2, -2]],
    'J3': [[1, 1], [0, 0], [-1, -1], [-2, 0]],
    'J4': [[0, 0], [0, 0], [0, 0], [0, 0]],
    'L1': [[-1, 1], [0, 0], [1, -1], [-2, 0]],
    'L2': [[0, 2], [0, 0], [0, -2], [-2, 2]],
    'L3': [[1, 1], [0, 0], [-1, -1], [0, 2]],
    'L4': [[0, 0], [0, 0], [0, 0], [0, 0]]
}


# compares predicted board to real board to verify that predictions are correct
def board_check(predicted_board, real_board):
    for row in range(len(predicted_board)):
        for column in range(len(predicted_board[0])):
            if (predicted_board[row][column] == 0 and real_board[row][column] != 0) or (
                    predicted_board[row][column] != 0 and real_board[row][column] == 0):
                return False
    return True


# EVAL FUNCTIONS
def getAggregateHeight(board):
    average = 0
    for column in range(len(board[0])):
        total = 0
        for row in range(len(board)):
            if board[row][column] != 0:
                total += len(board) - row
                break
        average += total
    average = average / len(board[0])
    return average


def getBumpiness(board):
    heights = []
    bumpiness = 0
    for column in range(len(board[0])):
        for row in range(len(board)):
            if board[row][column] != 0:
                heights.append(len(board) - row)
                break
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness


def getHoles(board):
    holes = 0
    for column in range(len(board[0])):
        flag = False
        for row in range(len(board)):
            if board[row][column] != 0:
                flag = True
            elif flag:
                holes += 1
    return holes


# complete lines
def getCompleteLines(board):
    lines = 0
    for row in range(len(board)):
        count = 0
        for column in range(len(board[0])):
            if board[row][column] != 0:
                count += 1
        if count == len(board[0]):
            lines += 1
    return lines


pyboy.tick()


class tetrisAI(ContinuousGenAlgSolver):
    def __init__(self, *args, **kwargs):
        ContinuousGenAlgSolver.__init__(self, *args, **kwargs)

    # fitness function to be maximized
    def fitness_function(self, chromosome):
        # board = getCurrentBoard()
        print(chromosome)

        tetris.reset_game()
        pyboy.tick()
        pyboy.set_emulation_speed(0)

        score = 0
        while True:
            try:
                num1 = 0
                num2 = 0
                temp = 0
                for i in range(-4, 4):
                    for j in range(4):
                        # TODO: predictions don't account for next tetromino... fix this
                        # TODO: when making predictions, read the board, pause the game, solve, then play...
                        #  this gives more time for AI to solve (may even extend max prediction height)
                        prediction = calculateReward(get_predicted_board(i, j),
                                                     chromosome[0], chromosome[1],
                                                     chromosome[2], chromosome[3])

                        if prediction >= temp:
                            num1 = i
                            num2 = j
                            temp = prediction

                action(num1, num2)
                score += temp
            except:
                break
        return score

    pass


solver = tetrisAI(
    n_genes=4,  # number of variables defining the problem
    pop_size=5,  # TODO: YL set it to 1000. Set ideal population size
    max_gen=10,  # TODO: YL did 500 max moves... we have to do something else for upper limit
    mutation_rate=0.1,  # TODO: YL did 0.05... Set ideal mutation rate.
    selection_rate=0.6,  # percentage of the population to select for mating
                # TODO: YL did 0.1
    selection_strategy="roulette_wheel",  # TODO: YL did tournament style... choose an ideal strategy
    problem_type=float,
    variables_limits=(-1, 1)
)

solver.solve()

# score = 0
# while True:
#     pyboy.tick()  # what does this actually do?
#
#     # area = getCurrentBoard()
#     # area[1][2] = 1
#     # print(area)
#
#     # num1 = random.randint(-4, 4)
#     # num2 = random.randint(0, 4)
#     #
#     # temp = get_predicted_board(num1, num2)
#     # action(num1, num2)
#
#     num1 = 0
#     num2 = 0
#     temp = 0
#     for i in range(-4, 4):
#         for j in range(4):
#             prediction = calculateReward(get_predicted_board(i, j))
#
#             if prediction >= temp:
#                 num1 = i
#                 num2 = j
#                 temp = prediction
#
#     action(num1, num2)
#
#     score += calculateReward(getCurrentBoard())
#     print(score)
#
#     # print(calculateReward(getCurrentBoard()))
#
#     # print(tetris.score)
#     # print("timestep = ", timestep, calculateFitness())
