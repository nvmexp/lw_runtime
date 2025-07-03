################ Tests go here ################

TARGETS_MAKEFILE ?=
# LwSwitch SRT Test Makefile
TARGETS_MAKEFILE += lwswitch.lwmk

# This isn't a normal SRT test, it's an exelwtable that will display various
# info about the GPU.  It is meant to be used at runtime in conjunction with
# scripts to report which tests are/aren't available on the current
# platform.
TARGETS_MAKEFILE += utils/testList/testListHelper.lwmk
