CXX_FLAGS =-std=c++11 -I../include -L../lib -llwtensor -llwblas -Xcompiler=-fopenmp

all:
	lwcc contraction_autotuning.lw -o  contraction_autotuning ${CXX_FLAGS}
	lwcc contraction.lw -o  contraction ${CXX_FLAGS}
	lwcc elementwise_binary.lw -o  elementwise_binary ${CXX_FLAGS}
	lwcc elementwise_permute.lw -o  elementwise_permute ${CXX_FLAGS}
	lwcc elementwise_trinary.lw -o  elementwise_trinary ${CXX_FLAGS}
	lwcc vectorization.lw -o  vectorization ${CXX_FLAGS}
	lwcc reduction.lw -o  reduction ${CXX_FLAGS}
