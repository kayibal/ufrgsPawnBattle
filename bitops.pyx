from libc.stdint cimport uint64_t
cimport cbitops
cdef class BitOps:

	def __cinit__(self):
		pass

	cdef clz(self, uint64_t bs):
		return cbitops.gcc_clz(bs)

	cdef ctz(self, uint64_t bs):
		return cbitops.gcc_ctz(bs)

	def py_clz(self,string):
		return self.clz(string)

