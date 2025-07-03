
# NOTE: if you add new source files here, you must add the same source
# files into the !.nact file too.

MICROBENCH_CPPFILES :=

MICROBENCH_CPPFILES += bench.cpp
MICROBENCH_CPPFILES += bench_ogl.cpp
MICROBENCH_CPPFILES += glprogram.cpp
MICROBENCH_CPPFILES += options.cpp
MICROBENCH_CPPFILES += timer.cpp
MICROBENCH_CPPFILES += utils.cpp
MICROBENCH_CPPFILES += cases/cbf.cpp
MICROBENCH_CPPFILES += cases/constantbind.cpp
MICROBENCH_CPPFILES += cases/clear_buffer.cpp
MICROBENCH_CPPFILES += cases/cpu_overhead.cpp
MICROBENCH_CPPFILES += cases/drawtest.cpp
MICROBENCH_CPPFILES += cases/fillrate.cpp
MICROBENCH_CPPFILES += cases/fillrate_ogl.cpp
MICROBENCH_CPPFILES += cases/fillrate_compute.cpp
MICROBENCH_CPPFILES += cases/gpfifo.cpp
MICROBENCH_CPPFILES += cases/gpu_time.cpp
MICROBENCH_CPPFILES += cases/kickoff.cpp
MICROBENCH_CPPFILES += cases/malloc_perf.cpp
MICROBENCH_CPPFILES += cases/multibind.cpp
MICROBENCH_CPPFILES += cases/pool_flush.cpp
MICROBENCH_CPPFILES += cases/shaderbind.cpp
MICROBENCH_CPPFILES += cases/shaderperf.cpp
MICROBENCH_CPPFILES += cases/shaderperf/dce.cpp
MICROBENCH_CPPFILES += cases/shaderperf/expr.cpp
MICROBENCH_CPPFILES += cases/shaderperf/hoistDiscards.cpp
MICROBENCH_CPPFILES += cases/shaderperf/ipa.cpp
MICROBENCH_CPPFILES += cases/tex.cpp
MICROBENCH_CPPFILES += cases/tex_init.cpp
MICROBENCH_CPPFILES += cases/tiled_cache.cpp
MICROBENCH_CPPFILES += cases/trirate.cpp
MICROBENCH_CPPFILES += cases/trirate_ogl.cpp
MICROBENCH_CPPFILES += cases/warp_lwlling.cpp
MICROBENCH_CPPFILES += ../samples/common/lwnutil.cpp

ifeq ($(MICROBENCH_PLATFORM),win)
MICROBENCH_CPPFILES    += windows_main.cpp
endif

ifeq ($(MICROBENCH_PLATFORM),hos)
MICROBENCH_CPPFILES    += hos_main.cpp
endif

ifeq ($(MICROBENCH_PLATFORM),linux)
MICROBENCH_CPPFILES += linux_main.cpp
MICROBENCH_CPPFILES += ../common/lwnWin/lwn_win.cpp
MICROBENCH_CPPFILES += ../common/lwnWin/lwwinsys_win.cpp
MICROBENCH_CPPFILES += ../common/lwnWin/wayland_win.cpp
MICROBENCH_CPPFILES += ../common/lwnWin/dmabuf-protocol.c
MICROBENCH_CPPFILES += ../common/lwnWin/synchronization-protocol.c
endif
