import copy
import random

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        drop_phase = self.is_drop_phase(state)  # TODO: detect drop phase

        if not drop_phase:
            ls = self.succ(state, False, self.my_piece)
            sc = []
            for i in ls:
                sc.append(self.heuristic_minmax(i, 2, False))
            # print(sc)
            choice = ls[sc.index(max(sc))]
            st = ()
            ed = ()
            for i in range(5):
                for j in range(5):
                    if state[i][j] == choice[i][j]:
                        continue
                    if state[i][j] == ' ':
                        ed = (i,j)
                    else:
                        st = (i,j)
            return [ed,st]


        # select an unoccupied space randomly

        # TODO: implement a minimax algorithm to play better
        ls = self.succ(state, True, self.my_piece)
        sc = []
        for i in ls:
            sc.append(self.heuristic_minmax(state, 1, False))
        print(sc)
        choice = ls[sc.index(max(sc))]
        ed = ()
        for i in range(5):
            for j in range(5):
                if state[i][j] == choice[i][j]:
                    continue
                if state[i][j] == ' ':
                    ed = (i, j)
        # print(state)
        # print(choice)
        # print(ed)
        return [ed]

    @staticmethod
    def is_drop_phase(state):
        count = 0
        for row in state:
            for ele in row:
                if ele != ' ':
                    count += 1
        return count < 8

    def heuristic(self, state):
        sc = 0
        ls_ai = [[2,2]]
        ls_op = [[2,2]]
        for i in range(5):
            for j in range(5):
                if state[i][j] == self.my_piece:
                    ls_ai.append([i, j])
                elif state[i][j] == self.opp:
                    ls_op.append([i, j])

        for loc1 in ls_ai:
            for loc2 in ls_ai:
                if loc1[0] == loc2[0] and abs(loc2[1]-loc1[1]) <= 1:
                    sc += 0.02
                if loc1[1] == loc2[1] and abs(loc2[0] - loc1[0]) <= 1:
                    sc += 0.02

        for loc1 in ls_op:
            for loc2 in ls_op:
                if loc1[0] == loc2[0] and abs(loc2[1] - loc1[1]) <= 1:
                    sc -= 0.02
                if loc1[1] == loc2[1] and abs(loc2[0] - loc1[0]) <= 1:
                    sc -= 0.02
        ## print("----------------")
        # print(ls_ai)
        # print(ls_op)
        # print(sc)

        return sc

    def heuristic_minmax(self, state, depth, my_turn):
        curr_val = self.game_value(state)
        if curr_val != 0:
            return curr_val
        if depth == 0:
            return self.heuristic(state)
        if my_turn:
            sc = -1
            possible_moves = self.succ(state, self.is_drop_phase(state), self.my_piece)
            for i in possible_moves:
                sc = max(sc, self.heuristic_minmax(i, depth - 1, False))
            return sc

        sc = 1
        possible_moves = self.succ(state, self.is_drop_phase(state), self.opp)
        for i in possible_moves:
            sc = min(sc, self.heuristic_minmax(i, depth - 1, True))
        return sc

    # noinspection SpellCheckingInspection
    @staticmethod
    def succ(state, drop_phase, player_piece):
        """
        takes in a board state and returns a list of the legal successors

        """

        count = 0
        possible_moves = []
        if drop_phase:
            ## print("run")

            for row in range(len(state)):
                for col in range(len(state[row])):
                    if state[row][col] == ' ':
                        curr = copy.deepcopy(state)
                        curr[row][col] = player_piece
                        possible_moves.append(curr)

            return possible_moves
        for row in range(len(state)):
            for col in range(len(state[row])):
                if state[row][col] == player_piece:
                    for i in range(3):
                        if row - 1 + i < 0 or row - 1 + i > 4:
                            continue
                        for j in range(3):
                            if col - 1 + j < 0 or col - 1 + j > 4:
                                continue
                            if state[row - 1 + i][col - 1 + j] == ' ':
                                curr = copy.deepcopy(state)
                                curr[row - 1 + i][col - 1 + j] = curr[row][col]
                                curr[row][col] = ' '
                                possible_moves.append(curr)
        return possible_moves

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][
                    col]:
                    return 1 if state[i][col] == self.my_piece else -1

        # check \ diagonal wins
        for st_row in range(2):
            for st_col in range(2):
                if state[st_row][st_col] != ' ' and state[st_row][st_col] == state[st_row + 1][st_col + 1] == \
                        state[st_row + 2][st_col + 2] == state[st_row + 3][st_col + 3]:
                    return 1 if state[st_row][st_col] == self.my_piece else -1

        # check / diagonal wins
        for st_row in range(2):
            for st_col in range(2):
                if state[st_row][3 - st_col] != ' ' and state[st_row][3 - st_col] == state[st_row + 1][2 - st_col] == \
                        state[st_row + 2][1 - st_col] == state[st_row + 3][st_col]:
                    return 1 if state[st_row][3 - st_col] == self.my_piece else -1
        # check box wins
        for st_row in range(4):
            for st_col in range(4):
                if state[st_row][st_col] != ' ' and state[st_row][st_col] == state[st_row + 1][st_col] == state[st_row][st_col + 1] == state[st_row + 1][st_col + 1]:
                    return 1 if state[st_row][st_col] == self.my_piece else -1

        return 0  # no winner yet


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0
    """
    test = [['r', 'r', 'r', ' ', 'r'],[' ', 'b', 'b', '', ' '],['b', 'b', ' ', ' ', ' '],[' ', ' ', ' ', ' ', ' '],[' ', ' ', ' ', ' ', ' ']]
    print(test)
    ls = ai.succ(test, ai.is_drop_phase(test), ai.my_piece)
    sc = []
    for i in ls:
        sc.append(ai.heuristic_minmax(i, 2, True))
        print(i)
        print(ai.heuristic_minmax(i, 2, True))
    print(ls[sc.index(max(sc))])
    print(sc)
    """
    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")
if __name__ == "__main__":
    main()
