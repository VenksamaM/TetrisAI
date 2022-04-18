import random

import numpy as np
from pyboy import PyBoy, WindowEvent
import pygad
import pygad.gann
# from geneal.genetic_algorithms import ContinuousGenAlgSolver
# from geneal.applications.fitness_functions.continuous import fitness_functions_continuous

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


def calculateReward(board, w1, w2, w3, w4, w5):
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

    reward += abs(w5) * ((count / lines) / 100)

    return reward


def getCurrentBoard():
    return np.asarray(tetris.game_area()).astype(int) - 47


# 1
# 17
def get_predicted_board(translate, turn, tetromino=getCurrentTetromino(), startingBoard=getCurrentBoard()):
    num = -1

    # print(current, translate, turn)

    coords = getStartCoords(tetromino)

    board = np.copy(startingBoard)
    # print(getCurrentTetromino(), tetromino)
    # print(board)

    for item in range(len(coords)):
        # print("start ", item, ": ", board[coords[item][0], coords[item][1]])
        board[coords[item][0], coords[item][1]] = 0

    # print(board)

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


def calculateBestMove(weights):
    score = 0
    turns = 0
    translate = 0
    temp = 0
    board = getCurrentBoard()
    # print(board, "\n\n")
    count = 0
    for i in range(-5, 5):
        for j in range(4):
            try:
                predictedBoard = get_predicted_board(i, j, getCurrentTetromino(), getCurrentBoard())
                tempBoard = np.copy(predictedBoard)
                prediction = calculateReward(predictedBoard, weights[0], weights[1], weights[2], weights[3], weights[4])

                for i2 in range(-5, 5):
                    for j2 in range(4):
                        try:
                            predictedBoard2 = get_predicted_board(i2, j2, getNextTetromino(), tempBoard)
                            prediction2 = calculateReward(predictedBoard2,
                                                          weights[0], weights[1],
                                                          weights[2], weights[3], weights[4])

                            if prediction + prediction2 > temp:
                                turns = i
                                translate = j
                                temp = prediction + prediction2
                            # print(count)
                            # count += 1
                        except:
                            pass
            except:
                pass
    return [turns, translate, temp]


# class tetrisAI(ContinuousGenAlgSolver):
#     def __init__(self, *args, **kwargs):
#         ContinuousGenAlgSolver.__init__(self, *args, **kwargs)


# fitness function to be maximized
def fitness_function(solution, solution_idx):
    global GANN_instance

    # print("::: ", predictions, "\n")

    # board = getCurrentBoard()

    tetris.reset_game()
    pyboy.tick()
    pyboy.set_emulation_speed(0)

    score = 0
    while solution_idx != None:
        board = getCurrentBoard()
        data_inputs = np.array([[getHoles(board)], [getBumpiness(board)], [getAggregateHeight(board)],
                                [pyboy.get_memory_value(0xc203)], [pyboy.get_memory_value(0xC213)]])
        predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[solution_idx],
                                       data_inputs=data_inputs,
                                       problem_type="regression")
        bestMove = calculateBestMove(predictions[0])
        if bestMove[2] == 0:
            break
        action(bestMove[0], bestMove[1])
        score += bestMove[2]

    print(solution_idx, score)
    return score


GANN_instance = pygad.gann.GANN(num_solutions=50,
                                num_neurons_input=1,
                                num_neurons_hidden_layers=[5, 7, 5],
                                num_neurons_output=5  ,
                                hidden_activations=["relu", "relu", "relu"],
                                output_activation="softmax")
population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)
print(" . ", population_vectors)


num_generations = 20
num_parents_mating = 2  # percentage of total population (sol_per_pop * 0.1)

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

# solver = tetrisAI(
#     n_genes=5,  # number of variables defining the problem
#     pop_size=100,  # TODO: YL set it to 1000. Set ideal population size
#     max_gen=10,  # TODO: YL did 500 max moves... we have to do something else for upper limit
#     mutation_rate=0.1,  # TODO: YL did 0.05... Set ideal mutation rate.
#     selection_rate=0.4,  # percentage of the population to select for mating
#     # TODO: YL did 0.1
#     selection_strategy="tournament",  # TODO: YL did tournament style... choose an ideal strategy
#     problem_type=float,
#     variables_limits=(-1, 1)
# )
#
# solver.solve()


