import random
import types

import numpy as np
from pyboy import PyBoy, WindowEvent
import pygad.gann
import pygad.nn

pyboy = PyBoy('Tetris.gb', game_wrapper=True)
tetris = pyboy.game_wrapper()
tetris.start_game()

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
    pyboy.tick()


def drop_down():
    started_moving = False
    while pyboy.get_memory_value(0xc201) != start_y or not started_moving:
        started_moving = True
        action_down()


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
        coords = [[1, 3], [1, 4], [2, 4], [1, 5]]
    elif tetromino == "O":
        coords = [[1, 4], [1, 5], [2, 4], [2, 5]]
    elif tetromino == "J":
        coords = [[1, 3], [1, 4], [1, 5], [2, 5]]
    elif tetromino == "L":
        coords = [[1, 3], [2, 3], [1, 4], [1, 5]]

    return coords


def getNextTetromino():
    return tetris.next_tetromino()


def getLinesCleared():
    return tetris.lines


def getLevel():
    return tetris.level


def calculateReward(board):
    reward = 0
    if board[-1][0] != 88:
        reward = getCompleteLines(board)
    if reward > 0:
        print(board)
        holes = getHoles(board)
        aggregateHeight = getAggregateHeight(board)
        bumpiness = getBumpiness(board)

        multiplier = 1
        multiplier += aggregateHeight / 10
        multiplier += 1 + bumpiness / 10

        if holes > 0:
            multiplier += holes / 10

        reward = reward * multiplier

    # # calculate sub-rewards
    # lines = 0
    # count = 0
    # for row in range(len(board)):
    #     flag = False
    #     for column in range(len(board[0])):
    #         if board[row][column] != 0:
    #             if flag is False:
    #                 flag = True
    #                 lines += 1
    #             count += 1

    reward += calculateSubRewards(board)

    return reward


def calculateSubRewards(board):
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
    return (count / (lines / 2)) / 100


def getCurrentBoard():
    return np.asarray(tetris.game_area()).astype(int) - 47


# 1
# 17
def get_predicted_board(translate, turn, tetromino=getCurrentTetromino(), startingBoard=getCurrentBoard()):
    num = -1

    coords = getStartCoords(tetromino)

    board = np.copy(startingBoard)

    for item in range(len(coords)):
        board[coords[item][0], coords[item][1]] = 0

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

    if tetromino != 'O':
        if tetromino == 'I' or tetromino == "S" or tetromino == "Z":
            turn = turn % 2
        while turn != 0:
            tempCoords = np.add(coords, rotation[''.join([tetromino, str(turn)])])

            if all([x > 9 for x in tempCoords[:][1]]):
                turn = turn - 1
            else:
                coords = tempCoords
                turn = 0

    drop = True
    while drop:
        for block in range(len(coords)):
            if coords[block][0] > 16 or board[coords[block][0] + 1][coords[block][1]] != 0:
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
    'T1': [[-1, 1], [0, 0], [-1, -1], [1, -1]],
    'T2': [[0, 2], [0, 0], [-2, 0], [0, -2]],
    'T3': [[1, 1], [0, 0], [-1, 1], [-1, -1]],
    'T4': [[0, 0], [0, 0], [0, 0], [0, 0]],
    'S1': [[1, 1], [0, 0], [1, -1], [0, -2]],
    'Z1': [[0, 1], [-1, 0], [0, -1], [-1, -2]],  # [[-1, 1], [0, 0], [-1, -1], [0, -2]],
    'O': [[0, 0], [0, 0], [0, 0], [0, 0]],
    'J1': [[-1, 1], [0, 0], [1, -1], [0, -2]],
    'J2': [[0, 2], [0, 0], [0, -2], [-2, -2]],
    'J3': [[1, 1], [0, 0], [-1, -1], [-2, 0]],
    'J4': [[0, 0], [0, 0], [0, 0], [0, 0]],
    'L1': [[-1, 1], [-2, 0], [0, 0], [1, -1]],
    'L2': [[0, 2], [-2, 2], [0, 0], [0, -2]],
    'L3': [[1, 1], [0, 2], [0, 0], [-1, -1]],
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
        for row in range(3, len(board)):
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
        nextHeight = 0
        for row in range(3, len(board)):
            if board[row][column] != 0:
                nextHeight = len(board) - row
                break
        heights.append(nextHeight)
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness


def getHoles(board):
    holes = 0
    for column in range(len(board[0])):
        flag = False
        for row in range(3, len(board)):
            if board[row][column] != 0:
                flag = True
            elif flag:
                holes += 1
    return holes


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


def calculateBestMove(w):
    turns = 0
    translate = 0
    temp = float('-inf')
    # board = getCurrentBoard()
    # print(board, "\n\n")
    for i in range(-6, 6):
        for j in range(5):
            predictedBoard = get_predicted_board(i, j, getCurrentTetromino(), getCurrentBoard())

            if isinstance(predictedBoard, bool):
                # print("breaking 1\n")
                break

            tempBoard = np.copy(predictedBoard)
            # prediction = calculateReward(predictedBoard, weights[0], weights[1], weights[2], weights[3],
            # weights[4])
            prediction = w[0] * getCompleteLines(tempBoard) + \
                         w[1] * getHoles(tempBoard) + \
                         w[2] * getAggregateHeight(tempBoard) + \
                         w[3] * getBumpiness(tempBoard)

            for i2 in range(-6, 6):
                for j2 in range(5):
                    predictedBoard2 = get_predicted_board(i2, j2, getNextTetromino(), tempBoard)

                    if isinstance(predictedBoard2, bool):
                        # print("breaking 2 \n")
                        break
                    # prediction2 = calculateReward(predictedBoard2,
                    #                               weights[0], weights[1],
                    #                               weights[2], weights[3], weights[4])

                    prediction2 = w[0] * getCompleteLines(predictedBoard2) + \
                                  w[1] * getHoles(predictedBoard2) + \
                                  w[2] * getAggregateHeight(predictedBoard2) + \
                                  w[3] * getBumpiness(predictedBoard2)

                    if prediction + prediction2 > temp:
                        # print("temp check")
                        turns = i
                        translate = j
                        temp = prediction + prediction2
    # print("temp = ", temp)
    if temp == float('-inf'):
        print("bad")
        pass
    return [turns, translate, temp]


# fitness function to be maximized
def fitness_function(solution, solution_idx):
    global GANN_instance

    # print("::: ", solution, "\n")

    # board = getCurrentBoard()

    tetris.reset_game()
    pyboy.tick()
    pyboy.set_emulation_speed(0)

    score = 0
    while solution_idx is not None:
        board = getCurrentBoard()

        data_inputs = np.array([[getHoles(board), getBumpiness(board), getAggregateHeight(board),
                                pyboy.get_memory_value(0xc203), pyboy.get_memory_value(0xC213)]])
        # print(solution_idx, data_inputs)
        predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[solution_idx],
                                       data_inputs=data_inputs,
                                       problem_type="regression")
        # print("predictions =  ", predictions[0])
        bestMove = calculateBestMove(predictions[0])
        predictedBoard = get_predicted_board(bestMove[0], bestMove[1], getCurrentTetromino(), board)
        if bestMove[2] == float('-inf') or isinstance(predictedBoard, bool):
            break
        action(bestMove[0], bestMove[1])
        # score += bestMove[2]
        score += calculateReward(predictedBoard)
        pyboy.tick()

    print(solution_idx, score, getLinesCleared())
    return score


GANN_instance = pygad.gann.GANN(num_solutions=50,
                                num_neurons_input=5,
                                num_neurons_hidden_layers=[5],
                                num_neurons_output=5,
                                hidden_activations=["relu"],
                                output_activation="softmax")
population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)
print(" . ", population_vectors)

num_generations = 5
num_parents_mating = 4  # percentage of total population (sol_per_pop * 0.1)

initial_population = population_vectors.copy()
# sol_per_pop = 5
# num_genes = 5

# init_range_low = -1
# init_range_high = 1

parent_selection_type = "tournament"
keep_parents = 1
crossover_type = "single_point"
mutation_type = "adaptive"  # "adaptive" is IMPROVEMENT TO YL
mutation_percent_genes = [50, 10]


def on_gen(ga):
    global GANN_instance

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
                                                            population_vectors=ga.population)
    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print("Fitness of the best solution :", ga.best_solution()[1])
    print("Generation : ", ga.generations_completed)
    print()


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       initial_population=initial_population,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=on_gen)

ga_instance.run()  # run GA
ga_instance.plot_fitness()  # plot graph

# save current model
filename = 'genetic'
ga_instance.save(filename=filename)

# load current model
# loaded_ga_instance = pygad.load(filename=filename)

# get info about best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
