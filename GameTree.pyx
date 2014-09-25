import numpy as np
cimport numpy as np
import Bitboard as bb
from libc.stdint cimport uint64_t
import bitops
from random import randrange

cdef enum:
	PAWN = 1
	KNIGHT = 2
	ROOK = 4

cdef class State:
	#using bitboards so chessboard can be represented with 40 bytes
	cdef uint64_t white_pieces
	cdef uint64_t black_pieces
	cdef uint64_t pawns
	cdef uint64_t knights
	cdef uint64_t rooks

	cdef int value
	cdef Move previous_move
	cdef int depth

	cdef bint color #true for white
	cdef bint isLeave
	cdef int child_count
	#Using simple python list for testing purposes a cython implementation of a linked list would be maybe better
	#should benchmark against an array list first if reasonable speedup is achieved a list will do even better in space
	#---------------------------
	#no problem with this being a list or even slow will be used only at the end
	#adding need to be fast

	#TODO: make this a c list
	children = []
	
	def __init__(self, bint color):
		self.color = color
		self.child_count = 0
		#self.createStartConfig()

	def createStartConfig(self):
		#place pawns
		self.white_pieces = 255 << 8
		self.black_pieces = 255 << 48
		self.pawns = self.white_pieces | self.black_pieces

		#place knights
		self.knights = 1 << 1 | 1 << 6 | 1 << 57 | 1 << 62
		#single bits are best set using bit table
		self.white_pieces = self.white_pieces | 2 | 1 << 6
		self.black_pieces = self.black_pieces | 1 << 57 | 1 << 62

		#place rooks
		self.rooks = 1 | 1<<7 | 1<<56 | 1<<63
		self.white_pieces = self.white_pieces | 1 | 1<<7
		self.black_pieces = self.black_pieces | 1<<56 | 1<<63

	def getPawns(self):
		return self.pawns
	def setPawns(self,pawns):
		self.pawns = pawns
	def getKnights(self):
		return self.knights
	def setKnights(self, knights):
		self.knights = knights

	def getRooks(self):
		return self.rooks
	def setRooks(self, rooks):
		self.rooks = rooks

	def getWhite(self):
		return self.white_pieces
	def setWhite(self, white):
		self.white_pieces = white

	def getBlack(self):
		return self.black_pieces
	def setBlack(self, black):
		self.black_pieces = black

	def getCount(self):
		return self.child_count

	def getColor(self):
		return self.color

	def isLeave(self):
		return self.isLeave


	def addChild(self, State s):
		self.children.append(s)
		self.child_count += 1

	def resetChildren(self):
		self.children[:] = []

	def getChildren(self):
		return self.children
	#hm maybe at a child counter and call getNextChild?
	def getChild(self, int i):
		return self.children[i]

	def evaluateState(self):
		r = randrange(100)
		#print "evaluates to:", r
		return r
	def getValue(self):
		return self.value
	def setValue(self,int v):
		self.value = v

	def getMove(self):
		return self.previous_move
	def setMove(self, move):
		self.previous_move = move

	def getBestMove(self):
		#return child with highest value
		self.children.sort(state_compare)
		return self.children[0]

	def printState(self):
		board = ' '*64
		board = list(board)
		cdef uint64_t rooks = self.rooks
		cdef uint64_t pawns = self.pawns
		cdef uint64_t knights = self.knights

		while(rooks):
			sqno = bb.bitScanForward(rooks)
			if(self.black_pieces & bb.bitSet(sqno)):
				board[sqno] = 'R'
			else:
				board[sqno] = 'r'
			rooks = rooks & (rooks-1)
		while(pawns):
			sqno = bb.bitScanForward(pawns)
			if(self.black_pieces & bb.bitSet(sqno)):
				board[sqno] = 'P'
			else:
				board[sqno] = 'p'
			pawns = pawns & (pawns-1)
		while(knights):
			sqno = bb.bitScanForward(knights)
			if(self.black_pieces & bb.bitSet(sqno)):
				board[sqno] = 'K'
			else:
				board[sqno] = 'k'
			knights = knights & (knights-1)


		bitboard = ''.join(board)
		print '  --------------------------------'
		for i in range(8):
			r = 7-i
			squares = bitboard[i*8:8+i*8]
			squares = squares[::-1]
			spaced_squares = ''
			for s in squares:
				spaced_squares += s + ' | '
			print  ('%d| ' + spaced_squares) % r
			print '  --------------------------------'
		print ' | A | B | C | D | E | F | G | H |'

def move_compare(Move a, Move b):
	if (a.capture > b.capture):
		return 1
	elif (a.capture == b.capture):
		return 0
	else:
		return -1

def state_compare(State a, State b):
	if (a.value > b.value):
		return 1
	elif (a.value == b.value):
		return 0
	else:
		return -1

cdef class Move:
	cdef int piece
	cdef uint64_t origin
	cdef uint64_t target
	cdef int capture

	def __init__(self, piece, uint64_t origin, uint64_t target, capture):
		self.piece = piece
		self.origin = origin
		self.target = target
		self.capture = capture
	def printMove(self):
		if(self.piece == PAWN): 
			print "pawn"
		if(self.piece == ROOK): 
			print "rook"
		if(self.piece == KNIGHT): 
			print "knight"
		print "capture: " ,self.capture
		bb.printBitBoard(self.origin | self.target)

	def getPiece(self):
		return self.piece
	def getOrigin(self):
		return self.origin
	def getTarget(self):
		return self.target
	def getCapture(self):
		return self.capture


cdef class GameTree:
	cdef State root
	cdef int count

	#BitSet array used to set single bits in a word
	cdef np.ndarray BitSet

	#precalculated movement masks
	cdef np.ndarray white_pawn_attack
	cdef np.ndarray white_pawn_move
	cdef np.ndarray black_pawn_attack
	cdef np.ndarray black_pawn_move
	cdef np.ndarray knight_move
	cdef np.ndarray rook_move

	#copy current node's state we are inspecting
	cdef State current
	cdef uint64_t pawns
	cdef uint64_t knights
	cdef uint64_t rooks
	cdef uint64_t white
	cdef uint64_t black

	def printEnum(self):
		print PAWN, KNIGHT, ROOK

	def getCount(self):
		return self.count

	def __init__(self, color):
		#init saved moves
		self.white_pawn_attack = np.zeros(64,dtype=np.uint64)
		self.white_pawn_move = np.zeros(64,dtype=np.uint64)
		self.black_pawn_attack = np.zeros(64,dtype=np.uint64)
		self.black_pawn_move = np.zeros(64,dtype=np.uint64)
		self.knight_move = np.zeros(64,dtype=np.uint64)
		self.rook_move = np.zeros(64,dtype=np.uint64)
		self.generateAllMoveMasks();

		#init BitSet
		#self.BitSet = bb.createBitSet()

		#init root white starts
		self.root = State(True)
		self.count = 1
	'''
	Copys current state in class's buffer mainly to be able to iterate move by move
	'''
	def loadState(self, State s):
		self.current = s
		self.pawns = s.getPawns()
		self.knights = s.getKnights()
		self.rooks = s.getRooks()
		self.white = s.getWhite()
		self.black = s.getBlack()

	def generateAllMoves(self, State s):
		self.loadState(s)
		white = s.getColor();
		cdef uint64_t own_pieces = self.black
		cdef uint64_t opponent_pieces = self.white
		cdef uint64_t all_pieces = self.black | self.white
		cdef uint64_t row = 255 << 48 
		cdef uint64_t invalid_start_bit
		cdef np.ndarray pawn_movements = self.black_pawn_move
		cdef np.ndarray pawn_attacks = self.black_pawn_attack
		if(white):
			own_pieces = self.white
			opponent_pieces = self.black
			pawn_movements = self.white_pawn_move
			pawn_attacks = self.white_pawn_attack
			row = 255 << 8

		#get all colors pawns
		cdef uint64_t pawns = own_pieces & self.pawns
		cdef uint64_t knights = own_pieces & self.knights
		cdef uint64_t rooks = own_pieces & self.rooks

		#some variables for loops
		cdef int current_sq_no 
		cdef uint64_t current_pos
		cdef uint64_t move_mask 
		cdef uint64_t attack_mask

		#are attacks possible?
		cdef uint64_t capture_knights
		cdef uint64_t capture_rooks 
		cdef uint64_t capture_pawns 
		#will store all moves --- figures this should be an array for fast sorting
		moves = []
		while(pawns):
			#get data
			current_sq_no = bb.bitScanForward(pawns)
			current_pos = bb.bitSet(current_sq_no)
			move_mask = pawn_movements[current_sq_no]
			attack_mask = pawn_attacks[current_sq_no]

			#are attacks possible?
			capture_knights = attack_mask & (self.knights & opponent_pieces)
			capture_rooks = attack_mask & (self.rooks & opponent_pieces)
			capture_pawns = attack_mask & (self.pawns & opponent_pieces)

			moves += self.createMoves(capture_knights,current_pos,PAWN,KNIGHT)
			moves += self.createMoves(capture_rooks,current_pos,PAWN,ROOK)
			moves += self.createMoves(capture_pawns,current_pos,PAWN,PAWN)
			#are moves possible? 
			#special treatment for startposition
			move_mask = move_mask & ~(all_pieces)

			if(row & current_pos):
				#if pawn in startposition check if movemask is legal
				invalid_start_bit = current_pos >> 8
				if (white): invalid_start_bit = current_pos << 8
				if(~(invalid_start_bit & all_pieces)):
					moves += self.createMoves(move_mask,current_pos,PAWN,0)
			else:
				moves += self.createMoves(move_mask,current_pos,PAWN,0)

			#delete current pawn
			pawns = pawns & (pawns-1)

		#knight movements
		while(knights):
			current_square_no = bb.bitScanForward(knights)
			current_pos = bb.bitSet(current_square_no)
			move_mask = self.knight_move[current_square_no]

			move_mask = move_mask & ~(own_pieces)
			capture_knights = move_mask & (self.knights & opponent_pieces)
			capture_rooks = move_mask & (self.rooks & opponent_pieces)
			capture_pawns = move_mask & (self.pawns & opponent_pieces)

			#remove captures from move mask
			move_mask = move_mask & ~capture_knights & ~capture_rooks & ~capture_pawns

			moves += self.createMoves(move_mask,current_pos,KNIGHT,0)
			moves += self.createMoves(capture_knights,current_pos,KNIGHT,KNIGHT)
			moves += self.createMoves(capture_rooks,current_pos,KNIGHT,ROOK)
			moves += self.createMoves(capture_pawns,current_pos,KNIGHT,PAWN)
			#delete current knight
			knights = knights & (knights-1)

		while(rooks):
			current_square_no = bb.bitScanForward(rooks)
			current_pos = bb.bitSet(current_square_no)
			move_mask = bb.getRookMoveMask(all_pieces,current_square_no)
			#cannot capture own pieces
			move_mask = ~own_pieces & move_mask

			capture_knights = move_mask & (self.knights & opponent_pieces)
			capture_rooks = move_mask & (self.rooks & opponent_pieces)
			capture_pawns = move_mask & (self.pawns & opponent_pieces)

			moves += self.createMoves(move_mask,current_pos,ROOK,0)
			moves += self.createMoves(capture_knights,current_pos,ROOK,KNIGHT)
			moves += self.createMoves(capture_rooks,current_pos,ROOK,ROOK)
			moves += self.createMoves(capture_pawns,current_pos,ROOK,PAWN)

			#delete current rook
			rooks = rooks & (rooks-1)

		return moves

	def alphaBeta(self, State node, int depth, int alpha, int beta, bint maximizing):
		cdef int i
		cdef State new_node
		if(depth<=0):
			value = node.evaluateState()
			return value
		moves = self.generateAllMoves(node)
		moves.sort(move_compare)
		if (maximizing):
			#print "white"
			#print "moves possible: ", len(moves)
			for i in range(len(moves)):
				#moves[i].printMove()
				#print "max" ,i 
				new_node = self.makeMove(node,moves[i])
				#print  str(node)+ ' is father of ' + str(new_node)
				alpha = max(alpha, self.alphaBeta(new_node,depth-1,alpha,beta,False))
				node.setValue(alpha)
				if (beta <= alpha):
					break
				#node.depth = depth
				if (depth ==3): 
					node.addChild(new_node)
				#self.count += 1
			return alpha
		else:
			#print "black"
			#print "moves possible: ", len(moves)
			for i in range(len(moves)):
				#print "black"
				#moves[i].printMove()
				#print "min" ,i 
				new_node = self.makeMove(node,moves[i])
				#print  str(node)+ ' is father of ' + str(new_node)
				beta = min(beta,self.alphaBeta(new_node,depth-1,alpha,beta,True))
				node.setValue(beta)
				if (beta <= alpha):
					break
				if (depth ==3): 
					node.addChild(new_node)
				#node.depth = depth
				#self.count += 1
			return beta

			
	def makeMove(self, State current, Move move):
		cdef State new
		cdef uint64_t rooks
		cdef uint64_t pawns
		cdef uint64_t knights
		cdef uint64_t own_pieces 
		cdef uint64_t opponent_pieces
		cdef uint64_t target = move.getTarget()
		new = State(not(current.getColor()))
		#print "number of new node children ", len(new.getChildren())
		#new.resetChildren()
		new.setMove(move)

		assert((target & own_pieces == 0) and target)

		own_pieces = current.getBlack()
		opponent_pieces = current.getWhite()

		if(current.getColor()):
			own_pieces = current.getWhite()
			opponent_pieces = current.getBlack()
		
		own_pieces = (own_pieces | move.getTarget()) & ~move.getOrigin()

		if(current.getColor()):
			#print "white making a move"
			new.setWhite(own_pieces)
			new.setBlack(current.getBlack())
		else:
			#print "black making a move"
			new.setBlack(own_pieces)
			new.setWhite(current.getWhite())

		if(move.getPiece() == PAWN):
			pawns = current.getPawns() | move.getTarget()
			pawns = pawns & ~move.getOrigin()
			new.setPawns(pawns)
			#print "moving pawns"
			new.setKnights(current.getKnights())
			new.setRooks(current.getRooks())
		elif(move.getPiece() == ROOK):
			rooks = current.getRooks() | move.getTarget()
			rooks = rooks & ~move.getOrigin()
			new.setRooks(rooks)
			#print "moving rooks"
			new.setKnights(current.getKnights())
			new.setPawns(current.getPawns())
		elif(move.getPiece() == KNIGHT):
			knights = current.getKnights() | move.getTarget()
			knights = knights & ~move.getOrigin()
			new.setKnights(knights)
			#print "moving knights"
			new.setRooks(current.getRooks())
			new.setPawns(current.getPawns())
		if(move.getCapture>0):
			opponent_pieces = opponent_pieces & ~move.getTarget()
			if(move.getCapture == PAWN):
				new.setPawns(new.getPawns()& ~move.getTarget())
			if(move.getCapture == KNIGHT):
				new.setKnights(new.getKnights()& ~move.getTarget())
			if(move.getCapture == ROOK):
				new.setKnights(new.getKnights()& ~move.getTarget())
		#move.printMove()

		return new

	def createMoves(self, uint64_t move_mask, uint64_t current_pos, int piece, int capture):
		moves=[]
		cdef uint64_t target
		while(move_mask):
			#get possible move
			target = bb.bitSet(bb.bitScanForward(move_mask))
			m = Move(piece, current_pos, target, capture)
			moves.append(m)
			#delete this move
			move_mask = move_mask & (move_mask-1)
		return moves
	'''
	Calculates all bitboards for all possible moves a piece can make from any position
	'''
	def generateAllMoveMasks(self):
		cdef int i,j
		for j in range(8):
			for i in range(8):
				no = bb.getSquareNo(i,j)
				#pawns :)
				self.white_pawn_attack[no] = bb.createPawnAttackMask(i,j,True)
				self.black_pawn_attack[no] = bb.createPawnAttackMask(i,j,False)
				self.white_pawn_move[no] = bb.createPawnMoveMask(i,j,True)
				self.black_pawn_move[no] = bb.createPawnMoveMask(i,j,False)
				#rooks
				#self.rook_move[no] = bb.createRookMoveMask(i,j)
				#knights
				self.knight_move[no] = bb.createKnightMoveMask(i,j)

	'''
	returns a valid child state after moving a piece
	this method should find the best move possible to make from current state
	and return this state so we can achieve best efficiency from alphabeta pruning

	could make sense later but will skip first for simplicity
	
	def generateNextMove(self, State s):
		# check all moves in this order
		# todo set variable to check in which phase of move genration we left of
		# 1. pawn moving to last row
		# 2. pawn capturing last pawn
		# later add here killer moves
		# 3. pawns capturing rooks or knights
		# 4. rooks or knights capturing rooks or knights
		# 5. pawns capturing pawns
		# 6. knights or rooks capturing pawns
		# 7. all the other moves
		self.loadState(s)
		clr = s.getColor();

		#decide if move this to self....
		#this here a lot can be simplified by using opponent pieces???
		cdef uint64_t last_row = 255 << 8 #stays always the same should go in contructor
		cdef uint64_t pawns = self.black & self.pawns 
		cdef uint64_t opponent_pieces = self.white
		cdef np.ndarray movements = self.black_pawn_move
		cdef np.ndarray attacks = self.black_pawn_attack
		if (clr):
			last_row = last_row << 48
			pawns = self.white & self.pawns
			movements = self.white_pawn_move
			attacks = self.white_pawn_attack
			opponent_pieces = self.black

		# 1. pawn moving to last row
		#simply check if some pawn is in the row before end
		if(pawns & last_row):
			#win with next move
			pass
		# 2. pawn capturing last pawn
		# check if opponent has only one pawn left?
		cdef uint64_t lp = opponent_pieces & pawns
		if ( (lp & (lp-1)) == 0 ):
			pass #only one pawn left can it be beaten
			#must check for all 4 rooks and knights and for sourrounding pawns = 6 pieces
		
		#3. pawns capturing rooks or knights
		#are there any pawn who can do this? if yes directly generate move
		#check costs 

		#compute all possible pawn attacks
	'''