import numpy as np
cimport numpy as np

'''
def createBitSet():
	cdef np.ndarray BitSet = np.zeros(64, dtype = np.uint64)
	for i in range(64):
		BitSet[i] = 1 << i
	return BitSet


cdef np.ndarray BitSet = createBitSet()
#create move bitboards for each piece and square makes 3p*64sq*64bit = 1,5kB -> fits in cache

def createMovementBoards():
	cdef np.uint64 movementBoard[192]
	#cdef np.ndarray = np.zeros(192, dtype = np.uint64)
	for i in range(64):
		#pawn movements
	for i in range(63,128):
		#knights
	for i in range(128,192):
'''
'''
def printBitBoard(b):
	bitboard = np.binary_repr(b,width=64)
	for i in range(8):
		squares = bitboard[i*8:8+i*8]
		spaced_squares = ''
		for s in squares:
			spaced_squares += s + ' '
		print  ('%d|' + spaced_squares) % i
	print '  ---------------'
	print '  A B C D E F G H'


def createRookMoveMask(i,j):
	cdef np.uint64 column 
	column = BitSet[i] | BitSet[i+8] | BitSet[i+16] | BitSet[24] | BitSet[32] | BitSet[40] | BitSet[48] | BitSet[56]
	cdef np.uint64 row = 255 << j*8
	return column ^ row

# __builtin_ctzl to find trailing zeros in c efficient

cdef class State:
	#using bitboards so chessboard can be represented with 40 bytes
	cdef np.uint64 white_pieces
	cdef np.uint64 black_pieces
	cdef np.uint64 pawns
	cdef np.uint64 knights
	cdef np.uint64 rooks

	cdef boolean isLeave
	cdef int child_count
	cdef State* children

	def __init__(self):
		self.child_count = 0
		self.createStartConfig()

	def createStartConfig(self):
		#place pawns
		white_pieces = 255 << 8
		black_pieces = 255 << 48
		pawns = white_pieces | black_pieces

		#place knights
		knights = 1 << 1 | 1 << 6 | 1 << 57 | 1 << 62
		#single bits are best set using bit table
		white_pieces = white_pieces | 2 | 1 << 6
		black_pieces = black_pieces | 1 << 57 | 1 << 62

		#place rooks
		rooks = 1 | 1<<7 | 1<<56 | 1<<63
		white_pieces = white_pieces | 1 | 1<<7
		black_pieces = black_pieces | 1<<56 | 1<<63
	
	def evaluateState(self):
		#isLeave -> use evaluateFunction
		#else use children
		pass

cdef class GameTree:
	cdef State* root
	cdef int count

	def __init__(self):
		pass
'''

