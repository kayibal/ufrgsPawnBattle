#include "bitops.h"

int gcc_ctz(uint64_t bs){
	unsigned long long string = (unsigned long long)bs;
	return __builtin_ctzll(string);
}

int gcc_clz(uint64_t bs){
		unsigned long long string = (unsigned long long)bs;
	return __builtin_clzll(string);
}
/*
CFLAGS="-I/Users/Alan/Documents/Ufrgs/1er\ semestre/ai/bitops_lib"  \
LDFLAGS="-L/Users/Alan/Documents/Ufrgs/1er\ semestre/ai/bitops_lib"     \
    python comp_bf.py build_ext -i
*/