import heapq
import numpy as np
import copy




def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for i in range(len(from_state)):
        x = 0
        x_d = 0
        y = i
        y_d = i
        if from_state[i] != 0:
            if i < 3:
                y = i
            elif 3<= i < 6:
                x = 1
                y = i - 3
            else:
                x = 2
                y = i - 6
            if from_state[i] <4:
                x_d = 0
                y_d = from_state[i] - 1
            elif 4<=from_state[i]< 7:
                x_d = 1
                y_d = from_state[i] - 4
            elif 7<=from_state[i]< 8:
                x_d = 2
                y_d = from_state[i] - 7
            distance += abs(x-x_d) + abs(y-y_d)

    return distance




def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = []
    state_mtx = np.array(state).reshape(3, 3)
    z_1 = state.index(0)
    z_2 = state.index(0, z_1 + 1)
    x_1 = 0
    y_1 = 0
    x_2 = 0
    y_2 = 0
    if z_1 < 3:
        x_1 = 0
        y_1 = z_1
    if 2 < z_1 < 6:
        x_1 = 1
        y_1 = z_1 - 3
    if z_1 > 5:
        x_1 = 2
        y_1 = z_1 - 6

    if z_2 < 3:
        x_2 = 0
        y_2 = z_2
    if 2 < z_2 < 6:
        x_2 = 1
        y_2 = z_2 - 3
    if z_2 > 5:
        x_2 = 2
        y_2 = z_2 - 6
    
    if x_1 - 1 >=0 and state_mtx[x_1 - 1][y_1] != 0:
        state_temp = copy.deepcopy(state)
        state_temp = np.array(state_temp).reshape(3, 3)
        state_temp[x_1][y_1] = state_mtx[x_1 - 1][y_1]
        state_temp[x_1 - 1][y_1] = 0
        state_temp = state_temp.reshape(1, 9).tolist()
        succ_states.append(state_temp[0])
    
    if x_1 + 1 < 3 and state_mtx[x_1 + 1][y_1] != 0:
        state_temp = copy.deepcopy(state)
        state_temp = np.array(state_temp).reshape(3, 3)
        state_temp[x_1][y_1] = state_mtx[x_1 + 1][y_1]
        state_temp[x_1 + 1][y_1] = 0
        state_temp = state_temp.reshape(1, 9).tolist()
        succ_states.append(state_temp[0])
    
    if x_2 - 1 >=0 and state_mtx[x_2 - 1][y_2] != 0:
        state_temp = copy.deepcopy(state)
        state_temp = np.array(state_temp).reshape(3, 3)
        state_temp[x_2][y_2] = state_mtx[x_2 - 1][y_2]
        state_temp[x_2 - 1][y_2] = 0
        state_temp = state_temp.reshape(1, 9).tolist()
        succ_states.append(state_temp[0])

    if x_2 + 1 < 3 and state_mtx[x_2 + 1][y_2] != 0:
        state_temp = copy.deepcopy(state)
        state_temp = np.array(state_temp).reshape(3, 3)
        state_temp[x_2][y_2] = state_mtx[x_2 + 1][y_2]
        state_temp[x_2 + 1][y_2] = 0
        state_temp = state_temp.reshape(1, 9).tolist()
        succ_states.append(state_temp[0])

    if y_1 - 1 >=0 and state_mtx[x_1][y_1 - 1] != 0:
        state_temp = copy.deepcopy(state)
        state_temp = np.array(state_temp).reshape(3, 3)
        state_temp[x_1][y_1] = state_mtx[x_1][y_1 - 1]
        state_temp[x_1][y_1 - 1] = 0
        state_temp = state_temp.reshape(1, 9).tolist()
        succ_states.append(state_temp[0])
    
    if y_1 + 1 < 3 and state_mtx[x_1][y_1 + 1] != 0:
        state_temp = copy.deepcopy(state)
        state_temp = np.array(state_temp).reshape(3, 3)
        state_temp[x_1][y_1] = state_mtx[x_1][y_1 + 1]
        state_temp[x_1][y_1 + 1] = 0
        state_temp = state_temp.reshape(1, 9).tolist()
        succ_states.append(state_temp[0])

    if y_2 - 1 >=0 and state_mtx[x_2][y_2 - 1] != 0:
        state_temp = copy.deepcopy(state)
        state_temp = np.array(state_temp).reshape(3, 3)
        state_temp[x_2][y_2] = state_mtx[x_2][y_2 - 1]
        state_temp[x_2][y_2 - 1] = 0
        state_temp = state_temp.reshape(1, 9).tolist()
        succ_states.append(state_temp[0])
    
    if y_2 + 1 < 3 and state_mtx[x_2][y_2 + 1] != 0:
        state_temp = copy.deepcopy(state)
        state_temp = np.array(state_temp).reshape(3, 3)
        state_temp[x_2][y_2] = state_mtx[x_2][y_2 + 1]
        state_temp[x_2][y_2 + 1] = 0
        state_temp = state_temp.reshape(1, 9).tolist()
        succ_states.append(state_temp[0])
    
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.

    """
    pq = [] 
    visited = []
    parent_dict = {} 
    child_index = 0
    cost = 0 
    parent_index = -1
    max_length = 0

    h_0 = get_manhattan_distance(state,goal_state)
    heapq.heappush(pq, (h_0+cost, state, (cost, h_0, parent_index,child_index)))
    parent_dict[child_index] = (state, cost, h_0, parent_index)

    while pq:    
        max_length = len(pq)
        curr = heapq.heappop(pq)
        visited.append(curr)
        curr_state = curr[1]
        parent_dict[curr[2][3]] = (curr[1],curr[2][0],curr[2][1],curr[2][2])

        if curr_state == goal_state:
            path = []
            index = curr[2][3]
            p_ind = curr[2][2]
            curr = parent_dict[index]
            path.insert(0,curr)

            while p_ind != -1:
                parent = parent_dict[p_ind]
                path.insert(0,parent)
                p_ind = parent_dict[p_ind][3]

            for s in path:
                s_state = s[0]
                s_move = s[1]
                s_h = s[2]
                print(f'{s_state} h={s_h} moves: {s_move}')
            
            print(f'Max queue length: {max_length}')
            return

        successors = get_succ(curr_state)
        parent_index = curr[2][3]
        cost = curr[2][0]+1
        
        for s in successors:
            indicator = 0
            h = get_manhattan_distance(s, goal_state)

            for list in visited:
                if (s == list[1]):
                    indicator = 1

            if (indicator != 1):
                child_index += 1
                heapq.heappush(pq, (h+cost, s, (cost, h, parent_index,child_index)))


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    #print_succ([5, 2, 3, 0, 6, 4, 7, 1, 0])
    #get_succ([1, 7, 0, 6, 3, 2, 0, 4, 5])
    #print()

    #print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    #print()

    solve([4,3,0,5,1,6,7,2,0])
    print()
