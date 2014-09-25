import time
import sys
import random
import numpy as np
import Bitboard as bb
import GameTree as gt

from base_client import LiacBot

WHITE = True
BLACK = False

class Bot(LiacBot):
    name = 'Random Bot'

    def __init__(self):
        super(RandomBot, self).__init__()
        self.last_move = None

    def on_move(self, state):
        color = WHITE
        if(state['who_moves']== -1):
            color = BLACK
        s = gt.State(color)
        s.loadState(state['board'])
        s.printState()
        g = gt.GameTree(color)
        g.alphaBeta(s,3,-200000,200000,color)
        m = s.getBestMove()
        move_from = m.getFrom()
        move_to = m.getTo()

        self.move_to(move_from, move_to)


    def on_game_over(self, state):
        print 'Game Over.'
        sys.exit()