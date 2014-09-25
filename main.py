import numpy as np
import Bitboard as bb
import GameTree as gt
import time
from random import randrange

import Bitboard as bb
#all = pow(2,34) + pow(2,6) + pow(2,2)
#bb.printBitBoard(all)
#m = bb.getRookMoveMask(all,2)
#bb.printBitBoard(m)

#all = pow(2,17) + pow(2,22) + pow(2,43) + pow(2,3) + pow(2,19)
s = gt.State(True)
s.createStartConfig()
g = gt.GameTree(True)
#s.printState()
print len(s.getChildren())
g.alphaBeta(s,3,-200,200,True)
print len(s.getChildren())
s2 = s.getBestMove()
s2.printState()
#print g.getCount()


'''
moves = g.generateAllMoves(s)
print "white starts"
for m in moves:
	pass
	#m.printMove()

new = g.makeMove(s,moves[1])
print "white moves knight"
print"blacks turn"
moves = g.generateAllMoves(new)
for m in moves:
	m.printMove()

for i in range(30):
	moves = g.generateAllMoves(s)
	r = randrange(len(moves))
	s = g.makeMove(s,moves[r])
'''
#bb.debug(40)	