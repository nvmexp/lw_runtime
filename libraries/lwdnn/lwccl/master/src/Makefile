#
# Copyright (c) 2015-2019, LWPU CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../makefiles/common.mk
include ../makefiles/version.mk

##### src files
INCEXPORTS  := lwcl.h nccl_net.h
LIBSRCFILES := init.cc channel.cc bootstrap.cc transport.cc enqueue.cc \
                misc/group.cc misc/lwmlwrap.cc misc/ibvwrap.cc misc/rings.cc misc/utils.cc misc/argcheck.cc misc/trees.cc misc/topo.cc \
		transport/p2p.cc transport/shm.cc transport/net.cc transport/net_socket.cc transport/net_ib.cc \
                collectives/all_reduce.cc collectives/all_gather.cc collectives/broadcast.cc collectives/reduce.cc collectives/reduce_scatter.cc

##### lib files
LIBNAME     := libnccl.so
STATICLIBNAME := libnccl_static.a
##### pkgconfig files
PKGCONFIGFILE := lwcl.pc
##### dirs
BUILDDIR ?= $(abspath ../build)
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
PKGDIR := $(BUILDDIR)/lib/pkgconfig
##### target files
LWDARTLIB  ?= lwdart_static
INCTARGETS := $(INCEXPORTS:%=$(INCDIR)/%)
LIBSONAME  := $(LIBNAME:%=%.$(NCCL_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH))
STATICLIBTARGET := $(STATICLIBNAME)
PKGTARGET  := $(PKGCONFIGFILE)
LIBOBJ     := $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o)
DEPFILES   := $(LIBOBJ:%.o=%.d)
LDFLAGS    += -L${LWDA_LIB} -l$(LWDARTLIB) -lpthread -lrt -ldl

DEVICELIB  := $(BUILDDIR)/obj/collectives/device/colldevice.a

##### rules
build : lib staticlib

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) $(PKGDIR)/$(PKGTARGET)

staticlib : $(LIBDIR)/$(STATICLIBTARGET)

$(DEVICELIB): ALWAYS_REBUILD
	$(MAKE) -C collectives/device

# Empty target to force rebuild
ALWAYS_REBUILD:

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

$(INCDIR)/lwcl.h : lwcl.h.in
# NCCL_VERSION(X,Y,Z) ((X) * 1000 + (Y) * 100 + (Z))
	@$(eval NCCL_VERSION := $(shell printf "%d%d%02d" $(NCCL_MAJOR) $(NCCL_MINOR) $(NCCL_PATCH)))
	mkdir -p $(INCDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${lwcl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${lwcl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${lwcl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${lwcl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${lwcl:Version}/$(NCCL_VERSION)/g" \
	    $< > $@

$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) $(DEVICELIB)
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $(DEVICELIB) $(LDFLAGS)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(STATICLIBTARGET): $(LIBOBJ) $(DEVICELIB)
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(eval TMP := $(shell mktemp -d))
	cp $(LIBOBJ) $(TMP)
	cd $(TMP) && ar x $(DEVICELIB) && cd -
	ar cr $@ $(LIBOBJ) $(TMP)/*.o
	rm -Rf $(TMP)

$(PKGDIR)/lwcl.pc : lwcl.pc.in
	mkdir -p $(PKGDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e 's|$${lwcl:Prefix}|\$(PREFIX)|g' \
	    -e "s/\$${lwcl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${lwcl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${lwcl:Patch}/$(NCCL_PATCH)/g" \
	    $< > $@

$(INCDIR)/%.h : %.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	cp -f $< $@

$(INCDIR)/nccl_%.h : include/nccl_%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	cp -f $< $@

$(PKGDIR)/%.pc : %.pc
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(PKGDIR)
	cp -f $< $@

$(OBJDIR)/%.o : %.cc
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) -Iinclude -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) -Iinclude -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	rm -rf ${INCDIR} ${LIBDIR} ${PKGDIR} ${OBJDIR}
	$(MAKE) -C collectives/device clean

install : lib
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig
	mkdir -p $(PREFIX)/include
	cp -P -v $(BUILDDIR)/lib/lib* $(PREFIX)/lib/
	cp -P -v $(BUILDDIR)/lib/pkgconfig/* $(PREFIX)/lib/pkgconfig/
	cp -v $(BUILDDIR)/include/* $(PREFIX)/include/

FILESTOFORMAT := $(shell find . -name ".\#*" -prune -o \( -name "*.cc" -o -name "*.h" \) -print | grep -v -E 'ibvwrap.h|lwmlwrap.h|lwcl.h')
# Note that formatting.mk defines a new target so in order to not overwrite the default target,
# it shouldn't be included at the top. Also, it uses the above definition of FILESTOFORMAT as well
# as the BUILDDIR variable.
include ../makefiles/formatting.mk
