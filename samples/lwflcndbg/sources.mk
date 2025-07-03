#
# Common make fragment to be included by all the flcndbg builds;
# the individual builds should include this file and then use
# $(FLCNDBG_COMMON_SOURCES).  If you're adding a source file that applies
# to all builds of flcndbg, add it here.
#
# Note this makefile fragment needs to be interpretable both by Microsoft
# nmake and gnumake
#

FLCNDBG_COMMON_SOURCES =           \
    chip.c                         \
    br04.c                         \
	dac.c                          \
    dumpSession.c                  \
    exts.c                         \
    falcon0400.c                   \
    ./flcngdb/flcngdb.c            \
    ./flcngdb/flcngdbUI.cpp        \
    ./flcngdb/flcngdbUtils.cpp     \
    ./flcngdb/flcngdbUtilsWrapper.cpp \
    hal.c                          \
    halstubs.c                     \
    i2c.c                          \
    lwutil.c                       \
    os.c                           \
    print.c                        \
    pmugk107.c                     \
    pmugm107.c                     \
    dpu0200.c                      \
    dpu0201.c                      \
    dpu0205.c                      \
    socbrdgt124.c                  \
    tegrasys.c                     \
    tegrasyst124.c                 \
    tegrasyst114.c                 \
    tegrasyst30.c                  \
    tegrasyslw50.c                 \
    vgpu.c

