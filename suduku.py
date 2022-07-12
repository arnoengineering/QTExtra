import numpy as np

board = np.random.randint(1,9, (9,9))

row = 2
col = 1
same_sq = [row // 3, col // 3]  # for row col
# n = 3


"""
posible use---
def check_block_4_n(s, block, row_c=False):
    if row_c:
        block = block // 3
    rc = block*3
    bc = rc+2
    return s in board[rc[0]:bc[0], rc[1]:bc[1]]
"""
##########
# asuming incomplete,
def check_block_4_n(s, block):  # assuming
    if isinstance(block,int):
        block = np.array(((block-1) % 3, (block-1) // 3))
    rc = block*3
    bc = rc+2
    return s in board[rc[0]:bc[0], rc[1]:bc[1]]

def check_row_4_n(n, r):
    pass

def check_col_4_n(n,c):
    pass

def check_loc(loc):
    pass


"""find max count number then search for viablespaces for each block row or coll chech if only one space set n, else set posible spaces, if all posible spaces are in row,col, can set that row, col to be filled and not check there"""
def check_n(n):
    row conda
    col contains
    block contains fast search
    b = 0
    m = 0
    for r in board:
        if n in row
    locs = np.where(board == n)

