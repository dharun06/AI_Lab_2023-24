# Ex.No: 3  Implementation of Minimax Search
### DATE: 26/08/2025                                                                      
### REGISTER NUMBER : 212223060028
### AIM: 
Write a mini-max search algorithm to find the optimal value of MAX Player from the given graph.
### Algorithm:
1. Start the program
2. import the math package
3. Specify the score value of leaf nodes and find the depth of binary tree from leaf nodes.
4. Define the minimax function
5. If maximum depth is reached then get the score value of leaf node.
6. Max player find the maximum value by calling the minmax function recursively.
7. Min player find the minimum value by calling the minmax function recursively.
8. Call the minimax function  and print the optimum value of Max player.
9. Stop the program. 

### Program:
```
import math

def game_won(board, player):
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in wins:
        if board[a] == board[b] == board[c] == player:
            return True
    return False

def game_tied(board):
    return " " not in board

def get_available_moves(board):
    return [i for i, spot in enumerate(board) if spot == " "]

def make_move(board, move, player):
    board[move] = player

def undo_move(board, move):
    board[move] = " "

def minimax(board, depth, isMaximizingPlayer):
    if game_won(board, "O"):
        return 10
    if game_won(board, "X"):
        return -10
    if game_tied(board):
        return 0
    if isMaximizingPlayer:
        bestScore = -math.inf
        for move in get_available_moves(board):
            make_move(board, move, "O")
            score = minimax(board, depth + 1, False)
            undo_move(board, move)
            bestScore = max(score, bestScore)
        return bestScore
    else:
        bestScore = math.inf
        for move in get_available_moves(board):
            make_move(board, move, "X")
            score = minimax(board, depth + 1, True)
            undo_move(board, move)
            bestScore = min(score, bestScore)
        return bestScore

if __name__ == "__main__":
    # Board where O has already won across the top row
    board = ["O","O","O",
             "X","X"," ",
             " "," "," "]
    optimal_value = minimax(board, 0, True)
    print(f"The optimal value for MAX player is: {optimal_value}")

```

### Output:

<img width="483" height="162" alt="image" src="https://github.com/user-attachments/assets/b6e411c1-ae15-46fb-bf83-d77f58d13a71" />




### Result:
Thus the optimum value of max player was found using minimax search.
