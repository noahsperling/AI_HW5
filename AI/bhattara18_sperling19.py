import random
import sys
import numpy as np


sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


##
# AIPlayer
# Description: This AI player uses Minimax with Alpha/Beta pruning to search for the best move
# based on the evaluate_state heuristic function
# Assignment: Homework #3: Minimax AI
#
# Due Date: April 7th, 2017
#
# @names: Noah Sperling, Avaya Bhattara
##
class AIPlayer(Player):

    #list of nodes for search tree
    node_list = []

    #maximum depth
    max_depth = 3

    #current index - for recursive function
    cur_array_index = 0

    #highest evaluated move - to be reset every time the generate_states method is called
    highest_evaluated_move = None

    #highest move score - useful for finding highest evaluated move - to be reset
    highest_move_eval = -1

    #this AI's playerID
    me = -1

    #whether or not the playerID has been set up yet
    me_set_up = False

    # the matrix of weights from the input to the first layer
    first_weight_matrix = np.matrix([[1.0, -2.1, 1.3, -0.5, 2.138, -2.138, 1.2, -0.9, 1.26, 0.11, 2.1, -1.5],
                                     [-0.9, -0.75, 1.75, 0.256, 2.0, -1.25, -1.24, -2.0, 1.98, 0.5, 0.6, 0.7],
                                     [0.1, -0.89, 1.456, 2.013, 0.564, -2.136, 1.789, -0.2, -0.1, 1.03, 0.745, 1.111],
                                     [-2.0, -2.1, 1.023, 0.654, 0.213, -0.5, 0.7, 0.645, 1.236, 2.1, -1.111, 1.756],
                                     [-1.54, 2.113, 0.215, -0.555, -0.721, -1.231, 1.45, -0.2, 1.98, -0.89, -0.7, 1.0],
                                     [0.1, -0.1, 0.23, -0.23, 1.69, 1.420, 2.003, 0.001, 0.592, -0.25, 0.75, 1.0],
                                     [-0.125, 1.1, -0.8, 0.9, 2.131, -2.138, 0.141, 0.56, 1.7, 0.9, 2.0, -2.0],
                                     [1.0, -2.1, 1.45, -0.2, 1.0, 1.1, -0.5, -0.6, -0.7, -0.8, 2.0, 1.1],
                                     [1.1, 0.7, -0.2, 0.9, 1.9, -0.987, -1.2, 2.1, -1.75, 0.4, -1.352, 0.312]])

    # the matrix of weights from the first layer to the output layer
    second_weight_matrix = np.matrix([[3.0984, 1.21, -2.15, 0.6, 1.212, -2.56, -2.0, 1.105, -3.0984]])

    # the learning rate
    alpha = 8.0

    # number of states to read if setting a finite value
    states_to_train = 2759

    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Neural Network AI")

    # Method to create a node containing the state, evaluation, move, current depth,
    # the parent node, and the index
    def create_node(self, state, evaluation, move, current_depth, parent_index, index):
        node = [state, evaluation, move, current_depth, parent_index, index]
        self.node_list.append(node)

    ##
    # getPlacement
    #
    # Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    # Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    # Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        # implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:  # stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:  # stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    # Choose any x location
                    x = random.randint(0, 9)
                    # Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    # Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        # Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]


    ##
    # getMove
    # Description: Gets the next move from the Player.
    #
    # Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    # Return: The Move to be made
    ##
    def getMove(self, currentState):

        if not self.me_set_up:
            self.me = currentState.whoseTurn

        #searches for best move
        selectedMove = self.move_search(currentState, 0, -(float)("inf"), (float)("inf"))

        #if not None, return move, if None, end turn
        if not selectedMove == None:
            return selectedMove
        else:
            return Move(END, None, None)

    ##
    # getAttack
    # Description: Gets the attack to be made from the Player
    #
    # Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        # Attack a random enemy.
        return enemyLocations[0]


    ##
    # move_search - recursive
    #
    # uses Minimax with alpha beta pruning to search for best next move
    #
    # Parameters:
    #   game_state - current state
    #   curr_depth - current search depth
    #   alpha      - the parent node's alpha value
    #   beta       - the previous node's beta value
    #
    # Return
    #   returns a move object
    ##
    def move_search(self, game_state, curr_depth, alpha, beta):

        #if max depth surpassed, return state evaluation
        if curr_depth == self.max_depth + 1:
            return self.evaluate_state(game_state)

        #list all legal moves
        move_list = listAllLegalMoves(game_state)

        #remove end turn move if the list isn't empty
        if not len(move_list) == 1:
            move_list.pop()

        # list of nodes, which contain the state, move, and eval
        node_list = []

        #generate states based on moves, evaluate them and put them into a list in node_list
        for move in move_list:
            state_eval = 0
            state = getNextStateAdversarial(game_state, move)
            state_eval = self.evaluate_state(state)
            #print(state_eval)
            if not state_eval == 0.00001:
                node_list.append([state, move, state_eval])


        self.mergeSort(node_list)

        if not self.me == game_state.whoseTurn:
            move_list.reverse()

        best_nodes = []

        for i in range(0, 5): # temporary
            if not len(node_list) == 0:
                best_nodes.append(node_list.pop())



        #best_val = -1

        #if not at the max depth, expand all the nodes in node_list and return
        if curr_depth <= self.max_depth:
            for node in best_nodes:
                score = self.move_search(node[0], curr_depth + 1, alpha, beta)
                if game_state.whoseTurn == self.me:
                    if score > alpha:
                        alpha = score
                    if alpha >= beta:
                        #print("Pruned")
                        break
                else:
                    if score < beta:
                        beta = score
                    if alpha >= beta:
                        #print("Pruned")
                        break



        #if not curr_depth == 0:
        if game_state.whoseTurn == self.me and not curr_depth == 0:
            return alpha
        elif not game_state == self.me and not curr_depth == 0:
            return beta
        else:
            best_eval = -1
            best_node = []

            for node in best_nodes:
                if node[2] > best_eval:
                    best_eval = node[2]
                    best_node = node

            #print(len(best_node))
            if not best_node == []:
                return best_node[1]
            else:
                return None


    ##
    # get_closest_enemy_dist - helper function
    #
    # returns distance to closest enemy from an ant
    ##
    def get_closest_enemy_dist(self, my_ant_coords, enemy_ants):
        closest_dist = 100
        for ant in enemy_ants:
            if not ant.type == WORKER:
                dist = approxDist(my_ant_coords, ant.coords)
                if dist < closest_dist:
                    closest_dist = dist
        return closest_dist


    ##
    # get_closest_enemy_worker_dist - helper function
    #
    # returns distance to closest enemy worker ant
    ##
    def get_closest_enemy_worker_dist(self, my_ant_coords, enemy_ants):
        closest_dist = 100
        for ant in enemy_ants:
            if ant.type == WORKER:
                dist = approxDist(my_ant_coords, ant.coords)
                if dist < closest_dist:
                    closest_dist = dist
        return closest_dist


    ##
    # get_closest_enemy_food_dist - helper function
    #
    # returns distance to closest enemy food
    ##
    def get_closest_enemy_food_dist(self, my_ant_coords, enemy_food_coords):

        enemy_food1_dist = approxDist(my_ant_coords, enemy_food_coords[0])
        enemy_food2_dist = approxDist(my_ant_coords, enemy_food_coords[1])

        if enemy_food1_dist < enemy_food2_dist:
            return enemy_food1_dist
        else:
            return enemy_food2_dist


    ##
    # evaluate_state
    #
    # Evaluates and scores a GameState Object
    #
    # Parameters
    #   state - the GameState object to evaluate
    #
    # Return
    #   a double between 0 and 1 inclusive
    ##
    def evaluate_state(self, state):
        # The AI's player ID
        me = state.whoseTurn
        # The opponent's ID
        enemy = (state.whoseTurn + 1) % 2

        # Get a reference to the player's inventory
        my_inv = state.inventories[me]
        # Get a reference to the enemy player's inventory
        enemy_inv = state.inventories[enemy]

        # Gets both the player's queens
        my_queen = getAntList(state, me, (QUEEN,))
        enemy_queen = getAntList(state, enemy, (QUEEN,))

        # Sees if winning or loosing conditions are already met
        if (my_inv.foodCount == 11) or (enemy_queen is None):
            return 1.0
        if (enemy_inv.foodCount == 11) or (my_queen is None):
            return 0.0

        #the starting value, not winning or losing
        eval = 0.0

        #important number
        worker_count = 0
        drone_count = 0

        food_coords = []
        enemy_food_coords = []

        foods = getConstrList(state, None, (FOOD,))

        # Gets a list of all of the food coords
        for food in foods:
            if food.coords[1] < 5:
                food_coords.append(food.coords)
            else:
                enemy_food_coords.append(food.coords)

        #coordinates of this AI's tunnel
        tunnel = my_inv.getTunnels()
        t_coords = tunnel[0].coords

        #coordinates of this AI's anthill
        ah_coords = my_inv.getAnthill().coords

        #A list that stores the evaluations of each worker
        wEval = []

        #A list that stores the evaluations of each drone, if they exist
        dEval = []

        #queen evaluation
        qEval = 0

        #iterates through ants and scores positioning
        for ant in my_inv.ants:

            #scores queen
            if ant.type == QUEEN:

                qEval = 50.0

                #if queen is on anthill, tunnel, or food it's bad
                if ant.coords == ah_coords or ant.coords == t_coords or ant.coords == food_coords[0] or ant.coords == food_coords[1]:
                    qEval -= 10

                #if queen is out of rows 0 or 1 it's bad
                if ant.coords[0] > 1:
                    qEval -= 10

                # the father from enemy ants, the better
                qEval -= 2 * self.get_closest_enemy_dist(ant.coords, enemy_inv.ants)

            #scores worker to incentivize food gathering
            elif ant.type == WORKER:

                #tallies up workers
                worker_count += 1

                #if carrying, the closer to the anthill or tunnel, the better
                if ant.carrying:

                    wEval.append(100.0)

                    #distance to anthill
                    ah_dist = approxDist(ant.coords, ah_coords)

                    #distance to tunnel
                    t_dist = approxDist(ant.coords, t_coords)

                    #finds closest and scores
                    #if ant.coords == ah_coords or ant.coords == t_coords:
                        #print("PHill")
                        #eval += 100000000
                    if t_dist < ah_dist:
                        wEval[worker_count - 1] -= 5 * t_dist
                    else:
                        wEval[worker_count - 1] -= 5 * ah_dist

                #if not carrying, the closer to food, the better
                else:

                    wEval.append(80.0)

                    #distance to foods
                    f1_dist = approxDist(ant.coords, food_coords[0])
                    f2_dist = approxDist(ant.coords, food_coords[1])

                    #finds closest and scores
                    #if ant.coords == food_coords[0] or ant.coords == food_coords[1]:
                        #print("PFood")
                        #eval += 500

                    if f1_dist < f2_dist:
                        wEval[worker_count - 1] -= 5 * f1_dist
                    else:
                        wEval[worker_count - 1] -= 5 * f2_dist

                #the father from enemy ants, the better
                #eval += -5 + self.get_closest_enemy_dist(ant.coords, enemy_inv.ants)

            #scores soldiers to incentivize the disruption of the enemy economy
            else:

                #tallies up soldiers
                drone_count += 1

                dEval.append(50.0)

                nearest_enemy_worker_dist = self.get_closest_enemy_worker_dist(ant.coords, enemy_inv.ants)

                #if there is an enemy worker
                if not nearest_enemy_worker_dist == 100:
                     dEval[drone_count - 1] -= 5 * nearest_enemy_worker_dist

                #if there isn't an enemy worker, go to the food
                else:
                    dEval[drone_count - 1] -= 5 * self.get_closest_enemy_food_dist(ant.coords, enemy_food_coords)

        #scores other important things

        #state eval
        sEval = 0

        #assesses worker inventory
        if worker_count == 2:
            sEval += 50
        elif worker_count < 2:
            sEval -= 10
        elif worker_count > 2:
            eval_num = 0.00001
            return eval_num

        #assesses drone inventory
        if drone_count == 2:
            sEval += 50
        elif drone_count < 2:
            sEval -= 10
        elif drone_count > 2:
            eval_num = 0.00001
            return eval_num

        #assesses food
        if not sEval == 0:
            sEval += 20 * my_inv.foodCount

        #a temporary variable
        temp = 0

        #scores workers
        for val in wEval:
            temp += val
        if worker_count == 0:
            wEvalAv = 0
        else:
            wEvalAv = temp/worker_count

        temp = 0

        #scores drones
        for val in dEval:
            temp += val

        if not drone_count == 0:
            dEvalAv = temp/drone_count
        else:
            dEvalAv = 0

        #total possible score
        total_possible = 100.0 + 50.0 + 50.0 + 300.0

        #scores total evaluation and returns
        eval = (qEval + wEvalAv + dEvalAv + sEval)/ total_possible
        if eval <= 0:
            eval = 0.00002

        # print "wste called from heuristic"
        self.write_state_to_file(state, eval, "states_laptop.txt")
        # self.generate_input_matrix(state, eval)

        return eval


    ##
    # generate_input_matrix
    #
    # generates an input matrix for the neural network
    #
    # Parameters
    #   state - the gamestate object to assess
    #   h_eval - the heuristic function's evaluation of the state
    #          - to be ignored after training
    #
    # Return:
    #   returns a double between zero and one
    #
    ##
    def generate_input_matrix(self, state, h_eval, matrix_or_eval):

        # food inventory numbers
        player_food = 0
        enemy_food = 0

        # queen info
        queen_health = 0
        queen_dist_to_closest_enemy = 0
        queen_on_construct_not_grass = 0
        queen_in_rows_0_or_1 = 0

        # worker info
        worker_count = 0
        num_carrying = 0
        w_distance_to_move = 0
        distance_to_food = 0

        # drone info
        drone_count = 0
        d_distance_to_move = 0

        # The AI's player ID
        me = state.whoseTurn
        # The opponent's ID
        enemy = (state.whoseTurn + 1) % 2

        # Get a reference to the player's inventory
        my_inv = state.inventories[me]
        # Get a reference to the enemy player's inventory
        enemy_inv = state.inventories[enemy]

        # Gets both the player's queens
        my_queen = getAntList(state, me, (QUEEN,))
        enemy_queen = getAntList(state, enemy, (QUEEN,))

        player_food += my_inv.foodCount
        enemy_food += enemy_inv.foodCount

        food_coords = []
        enemy_food_coords = []

        foods = getConstrList(state, None, (FOOD,))

        # Gets a list of all of the food coords
        for food in foods:
            if food.coords[1] < 5:
                food_coords.append(food.coords)
            else:
                enemy_food_coords.append(food.coords)

        # coordinates of this AI's tunnel
        tunnel = my_inv.getTunnels()
        t_coords = tunnel[0].coords

        # coordinates of this AI's anthill
        ah_coords = my_inv.getAnthill().coords

        # iterates through ants and scores positioning
        for ant in my_inv.ants:

            # scores queen
            if ant.type == QUEEN:

                # if queen is on anthill, tunnel, or food it's bad
                if ant.coords == ah_coords or ant.coords == t_coords or ant.coords == food_coords[0] or ant.coords == \
                        food_coords[1]:
                    queen_on_construct_not_grass += 1

                # if queen is out of rows 0 or 1 it's bad
                if ant.coords[0] > 1:
                    queen_in_rows_0_or_1 += 1

                # the father from enemy ants, the better
                queen_dist_to_closest_enemy += self.get_closest_enemy_dist(ant.coords, enemy_inv.ants)

            # scores worker to incentivize food gathering
            elif ant.type == WORKER:

                # tallies up workers
                worker_count += 1

                # if carrying, the closer to the anthill or tunnel, the better
                if ant.carrying:

                    # distance to anthill
                    ah_dist = approxDist(ant.coords, ah_coords)

                    # distance to tunnel
                    t_dist = approxDist(ant.coords, t_coords)

                    # finds closest and scores
                    # if ant.coords == ah_coords or ant.coords == t_coords:
                    # print("PHill")
                    # eval += 100000000
                    if t_dist < ah_dist:
                        w_distance_to_move += t_dist
                    else:
                        w_distance_to_move += ah_dist

                # if not carrying, the closer to food, the better
                else:

                    # distance to foods
                    f1_dist = approxDist(ant.coords, food_coords[0])
                    f2_dist = approxDist(ant.coords, food_coords[1])

                    # finds closest and scores
                    # if ant.coords == food_coords[0] or ant.coords == food_coords[1]:
                    # print("PFood")
                    # eval += 500

                    if f1_dist < f2_dist:
                        w_distance_to_move += f1_dist
                    else:
                        w_distance_to_move += f2_dist

                        # the father from enemy ants, the better
                        # eval += -5 + self.get_closest_enemy_dist(ant.coords, enemy_inv.ants)

            # scores soldiers to incentivize the disruption of the enemy economy
            else:

                # tallies up soldiers
                drone_count += 1

                nearest_enemy_worker_dist = self.get_closest_enemy_worker_dist(ant.coords, enemy_inv.ants)

                # if there is an enemy worker
                if not nearest_enemy_worker_dist == 100:
                    d_distance_to_move += nearest_enemy_worker_dist

                # if there isn't an enemy worker, go to the food
                else:
                    d_distance_to_move += self.get_closest_enemy_food_dist(ant.coords, enemy_food_coords)

        #print(player_food, enemy_food, queen_health, queen_dist_to_closest_enemy, queen_on_construct_not_grass,
        #      queen_in_rows_0_or_1, worker_count, num_carrying, w_distance_to_move, distance_to_food, drone_count,
        #      d_distance_to_move)
        output = np.matrix([[player_food], [enemy_food], [queen_health], [queen_dist_to_closest_enemy],
                [queen_on_construct_not_grass], [queen_in_rows_0_or_1], [worker_count], [num_carrying],
                [w_distance_to_move], [distance_to_food], [drone_count], [d_distance_to_move]])
        if matrix_or_eval == 0:
            return output
        else:
            return self.neural_network(output, h_eval)


    ##
    # neural_network
    #
    # simulates the neural network based on the input matrix and learns the
    # heuristic function
    #
    # Parameters
    #   input_matrix - a matrix generated by the self
    #
    # Return
    #   returns a state evaluation from the neural network
    #
    ##
    def neural_network(self, input_matrix, h_eval):

        first_layer_input = np.matmul(self.first_weight_matrix, input_matrix)

        temp_list = []

        for x in np.nditer(first_layer_input):
            temp_list.append(self.g(x))

        first_layer_output = np.matrix([[temp_list[0]], [temp_list[1]], [temp_list[2]], [temp_list[3]], [temp_list[4]],
                                        [temp_list[5]], [temp_list[6]], [temp_list[7]], [temp_list[8]]])

        #print self.second_weight_matrix
        #print first_layer_output

        second_layer_input = np.matmul(self.second_weight_matrix, first_layer_output)

        second_layer_output = self.g(second_layer_input)

        # this is bad code but I'm still new to numpy and it should work

        output = 0

        for y in np.nditer(second_layer_output):
            output += y
            break

        #print output

        error = h_eval - output

        print "Error:", error

        # print output

        #return output

        # back propogation

        index = 0

        delta_array = []

        swl = []

        for x in np.nditer(self.second_weight_matrix):
            delta = self.g_derivative(second_layer_input, False) * error
            #g_deriv = self.g_derivative(second_layer_input, True)
            new_weight = x + self.alpha * delta * output
            swl.append(new_weight)
            index += 1

        self.second_weight_matrix = np.matrix([[swl[0], swl[1], swl[2], swl[3], swl[4], swl[5], swl[6], swl[7],
                                                swl[8]]])

        weight_list = []

        for item in np.nditer(self.second_weight_matrix):
            weight_list.append(item)

        for x in range(9):
            delta_array.append(self.g_derivative(first_layer_output[x], True) * weight_list[x] * delta)

        for x in range(9):
            for y in range(12):

                #print self.first_weight_matrix[x, y]

                self.first_weight_matrix[x, y] = self.first_weight_matrix[x, y] + self.alpha * delta_array[x] * input_matrix[y, 0]

                #print self.first_weight_matrix[x, y]

        #for x in np.nditer(self.first_weight_matrix):


        return output


    ##
    # g(x)
    #
    # Parameters
    #   x - number to input to sigmoid function
    #
    # Return
    #   returns the output value of the sigmoid function
    ##
    def g(self, x):
        return 1 / (1 + np.exp(-x).item())


    ##
    # g_derivative
    #
    # Parameters
    #   x - number to input to sigmoid function
    #
    # Return
    #   returns the derivative of the sigmoid function at x
    ##
    def g_derivative(self, x, g_of_x):
        if not g_of_x:
            output = (1 / (1+np.exp(-x).item())) * (1 - (1 / (1+np.exp(-x).item())))
            return output
        else:
            return x * (1 - x)


    ##
    # merge_sort
    #
    # useful for sorting the move list from least to greatest in nlog(n) time
    ##
    def mergeSort(self, alist):
        if len(alist) > 1:
            mid = len(alist) // 2
            lefthalf = alist[:mid]
            righthalf = alist[mid:]

            self.mergeSort(lefthalf)
            self.mergeSort(righthalf)

            i = 0
            j = 0
            k = 0
            while i < len(lefthalf) and j < len(righthalf):
                if lefthalf[i][2] < righthalf[j][2]:
                    alist[k] = lefthalf[i]
                    i = i + 1
                else:
                    alist[k] = righthalf[j]
                    j = j + 1
                k = k + 1

            while i < len(lefthalf):
                alist[k] = lefthalf[i]
                i = i + 1
                k = k + 1

            while j < len(righthalf):
                alist[k] = righthalf[j]
                j = j + 1
                k = k + 1

    def write_state_to_file(self, state, state_eval, file_name):

        # the input matrix to save
        values = self.generate_input_matrix(state, state_eval, 0)

        # a list of values to print the matrix more easily
        lv = []

        # loops through the matrix and adds the values to a list
        for v in np.nditer(values):
            lv.append(v)

        #checks to make sure nothing went wrong
        if not len(lv) == 12:
            print "Error"

        # if nothing went wrong
        else:
            try:
                file = open(file_name, "a")
                # print "File opened"
                line_to_write = lv[0], lv[1], lv[2], lv[3], lv[4], lv[5], lv[6], lv[7], lv[8], lv[9], lv[10], lv[11], state_eval
                file.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (lv[0], lv[1], lv[2], lv[3], lv[4], lv[5], lv[6], lv[7], lv[8], lv[9], lv[10], lv[11], state_eval))
                file.close()
                # print "File closed"
            except Exception:
                print("Something went wrong opening and appending to the text file.")
        return

    def read_states_from_file_and_train_neural_network(self, file_name):
        with open(file_name) as f:

            print self.first_weight_matrix

            line_count = 0

            for line in f:
                ls = line.split(",")
                ls2 = []
                for num in ls:
                    ls2.append(float(num))
                input_matrix = np.matrix([[ls2[0]], [ls2[1]], [ls2[2]], [ls2[3]], [ls2[4]], [ls2[5]], [ls2[6]],
                                          [ls2[7]], [ls2[8]], [ls2[9]], [ls2[10]], [ls2[11]]])
                state_eval = ls2[12]
                self.neural_network(input_matrix, state_eval)
                line_count += 1

                if line_count >= self.states_to_train:
                    print self.first_weight_matrix
                    break
        f.close()
        return


# AIP = AIPlayer(PLAYER_ONE)
# AIP.read_states_from_file_and_train_neural_network("C:/Users/theem/PycharmProjects/AI_HW5/states.txt")
# AIP.read_states_from_file_and_train_neural_network("C:/Users/Noah/PycharmProjects/AI_HW5/states_laptop.txt")
# unit tests
# testPlayer = AIPlayer(PLAYER_ONE)
#test get_closest_enemy_dist
# testAntList = [Ant((2,4), 4, None), Ant((3,5), 2, None), Ant((2,5), 3, None), Ant((2,2), 1, None)]
# val = AIPlayer.get_closest_enemy_dist(testPlayer, (2,1), testAntList)
# assert (AIPlayer.get_closest_enemy_dist(testPlayer, (2,1), testAntList)==3), "get_closest_enemy_dist isn't working right(returned %d)" % val

# test get_closest_enemy_worker_dist
# testAntList = [Ant((2,4), 1, None), Ant((3,5), 1, None), Ant((2,5), 1, None), Ant((2,2), 2, None)]
# val = AIPlayer.get_closest_enemy_worker_dist(testPlayer, (2,1), testAntList)
# assert (AIPlayer.get_closest_enemy_worker_dist(testPlayer, (2,1), testAntList)==3), "get_closest_enemy_worker_dist isn't working right(returned %d)" % val

# test get_closest_enemy_food_dist
# val = AIPlayer.get_closest_enemy_food_dist(testPlayer, (2,3), [(2,4), (2,5)])
# assert (AIPlayer.get_closest_enemy_food_dist(testPlayer, (2,3), [(2,4), (2,5)])==1), "get_closest_enemy_food_dist isn't working right(returned %d)" % val

# test evaluate_state
# board = [[Location((col, row)) for row in xrange(0,BOARD_LENGTH)] for col in xrange(0,BOARD_LENGTH)]
# testConstrList1=[Construction((1,1), ANTHILL), Construction((1,2), TUNNEL), Construction((9,1), FOOD), Construction((9,2), FOOD)]
# testConstrList2=[Construction((9,9), ANTHILL), Construction((9,8), TUNNEL), Construction((1,8), FOOD), Construction((1,9), FOOD)]
# p1Inventory = Inventory(PLAYER_ONE, [Ant((1,1), 0, PLAYER_ONE), Ant((1,5), 1, PLAYER_ONE)], testConstrList1, 0)
# p2Inventory = Inventory(PLAYER_TWO, [Ant((1,2), 2, PLAYER_ONE), Ant((1,6), 2, PLAYER_ONE)], testConstrList2, 0)
# neutralInventory = Inventory(NEUTRAL, [], [], 0)
# testState1 = GameState(board, [p1Inventory, p2Inventory, neutralInventory], MENU_PHASE, PLAYER_ONE)
# eval1 = AIPlayer.evaluate_state(testPlayer, testState1)
# board = [[Location((col, row)) for row in xrange(0,BOARD_LENGTH)] for col in xrange(0,BOARD_LENGTH)]
# p1Inventory = Inventory(PLAYER_ONE, [Ant((1,1), 2, PLAYER_ONE), Ant((1,5), 2, PLAYER_ONE)], [Construction((1,1), ANTHILL), Construction((1,2), TUNNEL)], 0)
# p2Inventory = Inventory(PLAYER_TWO, [Ant((1,2), 0, PLAYER_ONE), Ant((1,6), 1, PLAYER_ONE)], [Construction((9,9), ANTHILL), Construction((9,8), TUNNEL)], 0)
# neutralInventory = Inventory(NEUTRAL, [], [], 0)
# testState2 = GameState(board, [p1Inventory, p2Inventory, neutralInventory], MENU_PHASE, PLAYER_ONE)
# eval2 = AIPlayer.evaluate_state(testPlayer, testState2)
# assert(eval1<eval2), "evaluate_state is broken (returned %d and %d)" % (eval1, eval2)


