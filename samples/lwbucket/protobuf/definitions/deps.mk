#
# Cross-platform support to build the protobuf files
# Basic idea is to establish source file dependencies for RM .proto files
# This is handled for gcc by the -M option but we don't have that here
#
# When this file is include'd the following 'make' var must be set:
#
#     LWRM_PROTOBUF_ROOT              - path to root of rm tree
#     LWRM_PROTOBUF_OUTPUTDIR         - path to the build directory

ifndef LWRM_PROTOBUF_ROOT
#$(warning LWRM_PROTOBUF_ROOT is not set)
LWRM_PROTOBUF_SRC :=
else
LWRM_PROTOBUF_SRC := $(LWRM_PROTOBUF_ROOT)/kernel/inc/protobuf/
endif


# Data Collection (DCL) primitives imported by all_dcl.proto
# Break these out to make it easier to add more later
DCL_PROTO_DEPS  =
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)bsp.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)dplib.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)engines.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)fifo.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)fcln.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)gr.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)journal.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)notifier.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)perf.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)rc.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)regs.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)smu.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)vbios.proto
DCL_PROTO_DEPS += $(LWRM_PROTOBUF_SRC)vp.proto

# Need one of these for each .proto file that imports another .proto file.
$(LWRM_PROTOBUF_SRC)all_dcl.proto: $(DCL_PROTO_DEPS)
$(LWRM_PROTOBUF_SRC)fifo.proto:    $(LWRM_PROTOBUF_SRC)regs.proto $(LWRM_PROTOBUF_SRC)lw4_fifo.proto
$(LWRM_PROTOBUF_SRC)gr.proto:      $(LWRM_PROTOBUF_SRC)regs.proto
$(LWRM_PROTOBUF_SRC)mc.proto:      $(LWRM_PROTOBUF_SRC)all_dcl.proto
$(LWRM_PROTOBUF_SRC)lwdebug.proto: $(LWRM_PROTOBUF_SRC)all_dcl.proto
$(LWRM_PROTOBUF_SRC)lwdebug.proto: $(LWRM_PROTOBUF_SRC)engines.proto
$(LWRM_PROTOBUF_SRC)lwdebug.proto: $(LWRM_PROTOBUF_SRC)pmu.proto
$(LWRM_PROTOBUF_SRC)lwdebug.proto: $(LWRM_PROTOBUF_SRC)rtos.proto
$(LWRM_PROTOBUF_SRC)perf.proto:    $(LWRM_PROTOBUF_SRC)regs.proto
$(LWRM_PROTOBUF_SRC)pmu.proto:     $(LWRM_PROTOBUF_SRC)regs.proto

# all_dcls dependencies as generated .h files
DCL_H_DEPS = $(DCL_PROTO_DEPS:$(LWRM_PROTOBUF_SRC)%.proto=$(LWRM_PROTOBUF_OUTPUTDIR)/g_%_pb.h)

# and one of these for each .proto that imports another
$(LWRM_PROTOBUF_OUTPUTDIR)/g_all_dcl_pb.h: $(DCL_H_DEPS)
$(LWRM_PROTOBUF_OUTPUTDIR)/g_fifo_pb.h:    $(LWRM_PROTOBUF_OUTPUTDIR)/g_regs_pb.h
$(LWRM_PROTOBUF_OUTPUTDIR)/g_fifo_pb.h:    $(LWRM_PROTOBUF_OUTPUTDIR)/g_lw4_fifo_pb.h
$(LWRM_PROTOBUF_OUTPUTDIR)/g_gr_pb.h:      $(LWRM_PROTOBUF_OUTPUTDIR)/g_regs_pb.h
$(LWRM_PROTOBUF_OUTPUTDIR)/g_mc_pb.h:      $(LWRM_PROTOBUF_OUTPUTDIR)/g_all_dcl_pb.h
$(LWRM_PROTOBUF_OUTPUTDIR)/g_lwdebug_pb.h: $(LWRM_PROTOBUF_OUTPUTDIR)/g_all_dcl_pb.h
$(LWRM_PROTOBUF_OUTPUTDIR)/g_lwdebug_pb.h: $(LWRM_PROTOBUF_OUTPUTDIR)/g_engines_pb.h
$(LWRM_PROTOBUF_OUTPUTDIR)/g_lwdebug_pb.h: $(LWRM_PROTOBUF_OUTPUTDIR)/g_pmu_pb.h
$(LWRM_PROTOBUF_OUTPUTDIR)/g_lwdebug_pb.h: $(LWRM_PROTOBUF_OUTPUTDIR)/g_rtos_pb.h
$(LWRM_PROTOBUF_OUTPUTDIR)/g_perf_pb.h:    $(LWRM_PROTOBUF_OUTPUTDIR)/g_regs_pb.h
$(LWRM_PROTOBUF_OUTPUTDIR)/g_pmu_pb.h:     $(LWRM_PROTOBUF_OUTPUTDIR)/g_regs_pb.h
