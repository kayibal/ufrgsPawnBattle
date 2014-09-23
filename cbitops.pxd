from libc.stdint cimport uint64_t
cdef extern from "bitops_lib/bitops.h":

	int gcc_ctz(uint64_t bs)

	int gcc_clz(uint64_t bs)