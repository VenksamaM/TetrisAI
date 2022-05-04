import math

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


def drop_down(translate, turn):
    # started_moving = False
    # while pyboy.get_memory_value(0xc201) != start_y or not started_moving:
    #     started_moving = True
    #     action_down()

    temp = np.asarray(tetris.game_area().copy())
    action_down()
    # pyboy.send_input(action_map['Down'][0])
    pyboy.tick()
    pyboy.tick()
    while not np.array_equal(temp[2, :], tetris.game_area()[2, :]):
        print("moving")
        temp = np.asarray(tetris.game_area().copy())
        pyboy.send_input(action_map['Down'][0])
        pyboy.tick()
        pyboy.send_input(action_map['Down'][1])
        pyboy.tick()
        for _ in range(20):
            pyboy.tick()

    # pyboy.send_input(action_map['Down'][1])


# go n_dir spaces in direction, and rotate n_turn times
def action(translate, turn):

    # print(translate, turn)

    coords = getStartCoords(getCurrentTetromino())
    # if translate is negative, move left. Else, right
    flag = 0
    if translate <= -5:
        flag = -1
    elif translate >= 5:
        flag = 1

    for _ in range(abs(translate)):
        if translate < 0:
            action_translate("Left")
        else:
            action_translate("Right")

    # for _ in range(abs(translate)):
    #     if translate < 0:
    #         move = True
    #         for x in coords:
    #             if x[1]-1 != 0 and [x[0], x[1]-1] not in coords:
    #                 move = False
    #         if move:
    #             action_translate("Left")
    #             pyboy.tick()
    #             for x in coords:
    #                 x[1] -= 1
    #         translate -= 1
    #     else:
    #         move = True
    #         for x in coords:
    #             if x[1] + 1 != 0 and [x[0], x[1] + 1] not in coords:
    #                 move = False
    #         if move:
    #             action_translate("Right")
    #             pyboy.tick()
    #             for x in coords:
    #                 x[1] += 1
    #         translate -= 1

    tetromino = getCurrentTetromino()
    for _ in range(turn):
        action_turn()
        # if tetromino != 'O':
        #     if tetromino == 'I' or tetromino == "S" or tetromino == "Z":
        #         turn = turn % 2
        #     while turn != 0:
        #         tempCoords = np.add(coords, rotation[''.join([tetromino, str(turn)])])
        #
        #         if all([x > 9 for x in tempCoords[:][1]]):
        #             turn = turn - 1
        #         else:
        #             coords = tempCoords
        #             turn = 0

    if flag == -1:
        action_translate("Left")
        action_translate("Left")
        # for _ in range(2):
        #     move = True
        #     for x in coords:
        #         if x[1] - 1 != 0 and [x[0], x[1] - 1] not in coords:
        #             move = False
        #     if move:
        #         action_translate("Left")
        #         for x in coords:
        #             x[1] += 1

    elif flag == 1:
        action_translate("Right")
        action_translate("Right")
        # for _ in range(2):
        #     move = True
        #     for x in coords:
        #         if x[1] + 1 != 0 and [x[0], x[1] + 1] not in coords:
        #             move = False
        #     if move:
        #         action_translate("Right")
        #         for x in coords:
        #             x[1] += 1

    finished = False

    # print(coords, getCurrentTetromino())
    # print()
    tet_y = pyboy.get_memory_value(0xFF93)
    pyboy.set_memory_value(0xFF99, 0)
    for _ in range(4):
        pyboy.tick()
    pyboy.set_memory_value(0xFF99, 42)

    # print(tet_y, pyboy.get_memory_value(0xFF93), 9999999999999999)

    while pyboy.get_memory_value(0xFF93) > tet_y:
        # print(tet_y, pyboy.get_memory_value(0xFF93), 999)
        tet_y = pyboy.get_memory_value(0xFF93)
        pyboy.set_memory_value(0xFF99, 0)
        for _ in range(3):
            pyboy.tick()
        pyboy.set_memory_value(0xFF99, 42)

    #     print("line clear in mem: ", pyboy.get_memory_value(0xFF9C))
    # print()

    # while not finished:
    #     # print(pyboy.get_memory_value(0xFF9A), pyboy.get_memory_value(0xFF99), pyboy.get_memory_value(0xFF98))
    #     #
    #
    #
    #
    #     # while tet_y >= 17 and
    #
    #     # for x in coords:
    #     #     next = np.array([x[0]+1, x[1]])
    #     #     # print(coords)
    #     # #     print(x[0] + 1, x[1], end=' ')
    #     # #     if x[0] + 1 < 18:
    #     # # #         print( getCurrentBoard()[x[0] + 1, x[1]], [x[0] + 1, x[1]] in coords, end='')
    #     # # #         print(getCurrentBoard()[x[0] + 1, x[1]] != 0, [x[0] + 1, x[1]] not in coords)
    #     # #         print(type(coords), type(x))
    #     # #         print(next, " in ", coords, "? ", any(np.array_equal(next, c) for c in coords))
    #     #     if x[0] + 1 > 17 :
    #     #         finished = True
    #     #         break
    #     #     #  [x[0] + 1, x[1]] not in coords
    #     #     # any(np.array_equal(x, c) for c in coords)
    #     #     elif getCurrentBoard()[x[0] + 1, x[1]] != 0 and not any(np.array_equal(next, c) for c in coords):
    #     #         finished = True
    #     #         break
    #     # # pyboy.set_memory_value(0xFF9A, 1)
    #     # # print()
    #     # if not finished:
    #     #     # action_down()
    #     #     pyboy.set_memory_value(0xFF99, 0)
    #     #     for _ in range(3):
    #     #         pyboy.tick()
    #     #     pyboy.set_memory_value(0xFF99, 42)
    #     #     print(pyboy.get_memory_value(0xFF92) / 16, pyboy.get_memory_value(0xFF93) / 8,
    #     #           pyboy.get_memory_value(0xFF9B))
    #     #     print()
    #     #
    #     #
    #     #     for x in coords:
    #     #         x[0] += 1
    #
    #     print("turn complete")
    # print(coords)
    # print()
    # for _ in range(10):
    #     pyboy.tick()


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
        # print(board)
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
    flag = 0

    coords = getStartCoords(tetromino)

    board = np.copy(startingBoard)

    for item in range(len(coords)):
        board[coords[item][0], coords[item][1]] = 0

    for column in range(len(board[2])):
        if board[4][column] != 0:
            # print("Prediction failure.")
            return False

    if translate >= 5:
        flag = 1
    elif translate <= -5:
        flag = -1

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

    move_to_edge = True
    while flag == -1:
        for row in range(0, len(coords)):
            if coords[row][1] - 1 < 0:
                move_to_edge = False
                flag = 0
                break

        if move_to_edge:
            for row in range(0, len(coords)):
                coords[row][1] -= 1
                # coords[i][0] +=  1
    while flag == 1:
        for row in range(0, len(coords)):
            if coords[row][1] + 1 >= 10:
                move_to_edge = False
                flag = 0
                break
        if move_to_edge:
            for row in range(0, len(coords)):
                coords[row][1] += 1
                # coords[i][0] +=  1

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
    # if temp == float('-inf'):
    #     print("bad")
    #     pass
    return [turns, translate, temp]


currentGen = 0


# fitness function to be maximized
def fitness_function(sol, sol_idx):
    global GANN_instance, ga_instance, currentGen

    # if ga_instance.generations_completed != currentGen:
    #     currentGen = ga_instance.generations_completed
    #     print("Generation = {generation}".format(generation=ga_instance.generations_completed), currentGen)
    #     ga_instance.save(filename='genetic')

    # board = getCurrentBoard()

    tetris.reset_game()
    pyboy.tick()
    pyboy.set_emulation_speed(0)

    score = 0
    while True:
        tetris.level = 0
        board = getCurrentBoard()

        data_inputs = np.array([[getHoles(board), getBumpiness(board), getAggregateHeight(board),
                                 pyboy.get_memory_value(0xc203), pyboy.get_memory_value(0xC213)]])
        # print(solution_idx, data_inputs)
        predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                       data_inputs=data_inputs,
                                       problem_type="regression")
        # print("predictions =  ", predictions[0])
        bestMove = calculateBestMove(predictions[0])

        # pyboy.tick()
        predictedBoard = get_predicted_board(bestMove[0], bestMove[1], getCurrentTetromino(), board)
        if bestMove[2] == float('-inf') or isinstance(predictedBoard, bool):
            print("test1")
            # print(getCurrentBoard())
            # print(predictedBoard)
            # print(bestMove[0], bestMove[1])
            break
        action(bestMove[0], bestMove[1])
        if tetris.game_over():
            print("test2222")
            break
        # print(board_check(predictedBoard, getCurrentBoard()), bestMove[0], bestMove[1]))
        # if not board_check(predictedBoard, getCurrentBoard()):
        #     print(bestMove[0], bestMove[1])
        #     print(predictedBoard)
        #     print()
        #     print(getCurrentBoard())

        if board_check(predictedBoard, getCurrentBoard()):
            # score += bestMove[2]
            score += calculateReward(predictedBoard)
            pyboy.tick()
        else:
            print("game Over")
            # print(bestMove[0], bestMove[1])
            # print(predictedBoard)
            # print()
            # print(getCurrentBoard())

        for i in range(50):
            pyboy.tick()

    print(sol_idx, score, getLinesCleared())
    return score


GANN_instance = pygad.gann.GANN(num_solutions=250,
                                num_neurons_input=5,
                                num_neurons_hidden_layers=[5],
                                num_neurons_output=5,
                                hidden_activations=["relu"],
                                output_activation="softmax")
population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)
print(" . ", population_vectors)

num_generations = 15
num_parents_mating = 2  # percentage of total population (sol_per_pop * 0.1)

initial_population = population_vectors.copy()
# print("\n ", len(population_vectors), len(initial_population), len(population_vectors[0]), len(initial_population[0]))
# sol_per_pop = 5
# num_genes = 5

# init_range_low = -1
# init_range_high = 1

parent_selection_type = "tournament"
keep_parents = 1
crossover_type = "single_point"
mutation_type = "random"  # "adaptive" is IMPROVEMENT TO YL
mutation_percent_genes = np.array([50, 10])

filename = 'genetic'


def on_gen(ga):
    global GANN_instance

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
                                                            population_vectors=ga.population)
    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print("Generation : ", ga.generations_completed)
    print("\n\n\n\n\n\n")

    ga_instance.save(filename=filename)
    ga_instance.plot_fitness()  # plot graph


# def callback_generation(ga_instance):
#     global GANN_instance, last_fitness
#
#     population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
#                                                             population_vectors=ga_instance.population)
#
#     GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)
#
#     print("Generation = {generation}".format(generation=ga_instance.generations_completed))
#     print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
#     print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
#
#     last_fitness = ga_instance.best_solution()[1].copy()
#     last_fitness.save(filename=filename)
#
#     return "stop"


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       initial_population=initial_population,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       on_generation=on_gen)

# load current model
loaded_ga_instance = pygad.load(filename=filename)

ga_instance = loaded_ga_instance

ga_instance.run()  # run GA
ga_instance.plot_fitness()  # plot graph

# save current model
ga_instance.save(filename=filename)

# get info about best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
