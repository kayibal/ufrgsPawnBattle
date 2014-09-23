import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t
#from cpython.mem cimport PyMem_Malloc, PyMem_Free

#constants
cdef enum:
	A = 0
	B = 1
	C = 2
	D = 3
	E = 4
	F = 5
	G = 6
	H = 7

cdef np.ndarray Index64 = np.array([
    0,  1, 48,  2, 57, 49, 28,  3,
   61, 58, 50, 42, 38, 29, 17,  4,
   62, 55, 59, 36, 53, 51, 43, 22,
   45, 39, 33, 30, 24, 18, 12,  5,
   63, 47, 56, 27, 60, 41, 37, 16,
   54, 35, 52, 21, 44, 32, 23, 11,
   46, 26, 40, 15, 34, 20, 31, 10,
   25, 14, 19,  9, 13,  8,  7,  6
])
cdef uint64_t debrujin64 = 0x03f79d71b4cb0a89

cdef uint64_t k1 = 0x5555555555555555
cdef uint64_t k2 = 0x3333333333333333
cdef uint64_t k4 = 0x0f0f0f0f0f0f0f0f
cdef uint64_t kf = 0x0101010101010101

#just some change
'''
Masks to set bits at i-th position
'''
def createBitSet():
	cdef np.ndarray BitSet = np.zeros(64, dtype = np.uint64)
	for i in range(64):
		BitSet[i] = 1 << i
	return BitSet
'''
Bit Reversal from Bit Twiddling Hacks
'''
def bitReversal(uint64_t x):
	x = (((x & 0xaaaaaaaaaaaaaaaa) >> 1) | ((x & 0x5555555555555555) << 1))
	x = (((x & 0xcccccccccccccccc) >> 2) | ((x & 0x3333333333333333) << 2))
	x = (((x & 0xf0f0f0f0f0f0f0f0) >> 4) | ((x & 0x0f0f0f0f0f0f0f0f) << 4))
	x = (((x & 0xff00ff00ff00ff00) >> 8) | ((x & 0x00ff00ff00ff00ff) << 8))
	x = (((x & 0xffff0000ffff0000) >> 16) | ((x & 0x0000ffff0000ffff) << 16))
	cdef uint64_t result = <uint64_t>((x >> 32) | (x << 32))
	return result
'''
Population count taken from https://chessprogramming.wikispaces.com/Population+Count
'''
def popcount(uint64_t x):
	x =  x       - ((x >> 1)  & k1)
	x = (x & k2) + ((x >> 2)  & k2) 
	x = (x       +  (x >> 4)) & k4  
	x = (x * kf) >> 56
	return int(x)


cdef np.ndarray BitSet = createBitSet()

def bitSet(int i):
	return BitSet[i]

def bitScanForward(uint64_t bb):
	#speed up by using gcc __builtin_ctzll
	if(bb==0):
		return -1
	return Index64[((bb & -bb) * debrujin64)>>58]
'''
Mapping to abstract bitboards as a 2-dimensional matrix
'''
def getSquareNo(int i, int j):
	if (j>=0 and j<=7 and i>=0 and i<=7):
		#index out of bounds
		return j*8+i
	return -1
'''
Invers Mapping to abstract bitboards as a 2-dimensional matrix
TODO: not sure if working
'''
def getSquare(int no):
	if (no>63 or no < 0):
		#out of bounds
		return -1,-1

	return no%8, <int> (no/8)

'''
Creates legal pawn movements on a empty board
also considering that pawns at start position are allowed to jump one squares
WARNING DO NOT USE WITH J=7 OR J=0
'''
def createPawnMoveMask(int i, int j, bint color):
	cdef uint64_t mask = 0
	if (j>=1 and j<=6):
		if(color):
			#white
			mask = BitSet[getSquareNo(i,j+1)]
			if(j==1):
				mask = <uint64_t>mask | <uint64_t>BitSet[getSquareNo(i,j+2)]

		else:
			mask = BitSet[getSquareNo(i,j-1)]
			if(j==6):
				mask = <uint64_t>mask | <uint64_t>BitSet[getSquareNo(i,j-2)]
			#black
	return mask

'''
Creates legal pawn attack masks on a empty board
'''
def createPawnAttackMask(int i, int j, bint color):
	cdef uint64_t mask = 0
	if (j>=1 and j<=6 and i >= 1 and i<=6):
		if(color):
			#white
			mask = BitSet[getSquareNo(i-1,j+1)] | BitSet[getSquareNo(i+1,j+1)]
		else:
			mask = BitSet[getSquareNo(i-1,j-1)] | BitSet[getSquareNo(i+1,j-1)]
	if(i==0):
		if(color):
			mask = BitSet[getSquareNo(i+1,j+1)]
		else:
			mask = BitSet[getSquareNo(i+1,j-1)]
	if (i== 7):
		if(color):
			mask = BitSet[getSquareNo(i-1,j+1)]
		else:
			mask = BitSet[getSquareNo(i-1,j-1)]
	return mask

'''
Creates a mask for legal rook movements on a empty board
Not used
'''
def createRookMoveMask(i,j):
	cdef uint64_t column 
	column = BitSet[i] | BitSet[i+8] | BitSet[i+16] | BitSet[i+24] | BitSet[i+32] | BitSet[i+40] | BitSet[i+48] | BitSet[i+56]
	cdef uint64_t row = 255 << j*8
	return column ^ row
'''
very dirty hack to convert an variable to because python throws an error
'''
def makeU(n):
	cdef uint64_t result = 0
	cdef int i
	for i in range(1,64):
		if(pow(2,i) & n):
			result += pow(2,i)
	return result
'''
still buggy for example for 0,1,8
'''
def getRookMoveMask(uint64_t all_pieces, int sqno):
	i,j = getSquare(sqno)
	cdef uint64_t row_mask = (255 << j*8)
	cdef uint64_t current_pos = BitSet[sqno]
	cdef uint64_t temp = <uint64_t> (<uint64_t>bitReversal(all_pieces)- <uint64_t>bitReversal(current_pos))
	cdef uint64_t temp2 = <uint64_t>(temp - <uint64_t>bitReversal(current_pos))
	cdef uint64_t line_moves = row_mask & (all_pieces - 2*current_pos ^ <uint64_t>bitReversal(temp2))
	#cdef line_moves_east = (all_pieces - 2*current_pos) ^ all_pieces

	cdef uint64_t file_moves_north = getFileMoves(all_pieces, sqno)
	sqno_f = getSquareNo(i,7-j)
	cdef uint64_t all_pieces_f = flipV(all_pieces)

	cdef uint64_t file_moves_south = flipV( getFileMoves(all_pieces_f, sqno_f) )

	return line_moves | file_moves_south | file_moves_north

def getFileMoves(uint64_t all_pieces, int sqno):
	cdef uint64_t current_pos = BitSet[sqno]
	cdef uint64_t fm = getFileMask(sqno%8)
	cdef uint64_t temp = <uint64_t> ((all_pieces & fm) - 2*current_pos)
	cdef uint64_t file_moves_north =  (temp ^ all_pieces) & fm
	return file_moves_north
'''
def getLineMovesWest(uint64_t all_pieces, int sqno):
	cdef int i
	cdef int j
	cdef uint64_t row_mask = ~(255 << j*8)
	cdef uint64_t current_pos = BitSet[sqno]
	cdef uint64_t line_moves = bitReversal(bitReversal(all_pieces)- 2 * bitReversal(current_pos)) ^ bitReversal(all_pieces)
	return row_mask & line_moves


def getLineMovesEast(uint64_t all_pieces, int sqno, int n):
	cdef int j = <int> (sqno/8)
	cdef uint64_t row_mask = ~(255 << n*8)
	printBitBoard(255 << n*8)
	print j
	cdef uint64_t current_pos = BitSet[sqno]
	cdef line_moves_east = (all_pieces - 2*current_pos) ^ all_pieces
	return row_mask & line_moves_east

'''
def rotate180(uint64_t bb):
	return mirrorH(flipV(bb))

def flipV(uint64_t x):
	return  ( (x << 56) ) | ( (x << 40) & 0x00ff000000000000 ) | ( (x << 24) & 0x0000ff0000000000 ) |( (x <<  8) & 0x000000ff00000000 ) |( (x >>  8) & 0x00000000ff000000 ) |( (x >> 24) & 0x0000000000ff0000 ) | ( (x >> 40) & 0x000000000000ff00 ) |( (x >> 56) )

def mirrorH(uint64_t x):
	x = ((x >> 1) & k1) | ((x & k1) << 1)
	x = ((x >> 2) & k2) | ((x & k2) << 2)
	x = ((x >> 4) & k4) | ((x & k4) << 4)
	return x

def getFileMask(int i):
	cdef uint64_t column
	if (i>=0 and i<=7):
		column = BitSet[i] | BitSet[i+8] | BitSet[i+16] | BitSet[i+24] | BitSet[i+32] | BitSet[i+40] | BitSet[i+48] | BitSet[i+56]
		return column
	else:
		return 0

'''
Creates legal knight movements on a empty board
'''
def createKnightMoveMask(int i, int j):
	#check if we are in boundaries were knight cannot move free
	cdef uint64_t mask = 0
	cdef np.ndarray pos = np.zeros(16,dtype=np.int)
	cdef int x = 0
	cdef int y = 0
	#upper right
	pos[0:4] =  [i+1,j+2,i+2,j+1]
	#lower right
	pos[4:8] =  [i+1,j-2,i+2,j-1]
	#upper left
	pos[8:12] = [i-1,j+2,i-2,j+1]
	#lower left
	pos[12:16]= [i-1,j-2,i-2,j-1]
	if ( j>1 and j<6 and i>1 and i<6):
		#completely in bounds no evaluation necessary
		for k in xrange(0,16,2):
			x = pos[k]
			y = pos[k+1]
			mask = mask | <uint64_t>BitSet[getSquareNo(x,y)]
	else:
		#out of bounds
		#print 'OoB'
		#caculate all possible positions
		#and evaluate if they are in bounds
		for k in xrange(0,16,2):
			x = pos[k]
			y = pos[k+1]
			no = getSquareNo(x,y)
			if (no > 0):
				mask = mask | <uint64_t>BitSet[no]
	return mask


def printBitBoard(uint64_t b):
	bitboard = np.binary_repr(b,width=64)
	for i in range(8):
		r = 7-i
		squares = bitboard[i*8:8+i*8]
		squares = squares[::-1]
		spaced_squares = ''
		for s in squares:
			spaced_squares += s + ' '
		print  ('%d|' + spaced_squares) % r
	print '  ---------------'
	print '  0 1 2 3 4 5 6 7'
	#print '  A B C D E F G H'

