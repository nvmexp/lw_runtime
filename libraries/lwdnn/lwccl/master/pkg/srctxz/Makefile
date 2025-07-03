#
# Copyright (c) 2015-2019, LWPU CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../../makefiles/common.mk
include ../../makefiles/version.mk
BUILDDIR ?= $(abspath ../../build)
TXZPREPDIR  := $(BUILDDIR)/srctxz
PKGDIR  := $(BUILDDIR)/pkg/srctxz/

TXZGEN_IN  := $(wildcard *.in)
TXZGEN     := $(TXZGEN_IN:.in=)
TXZTARGETS := $(patsubst %, $(TXZPREPDIR)/%, $(TXZGEN))

PKG_REVISION   ?= 3
PKG_ARCH       := $(shell uname -m)

prep: $(TXZTARGETS)

build: prep
	$(MAKE) -C ../../src clean
	@printf "Building source tar.xz package\n"
	(cd $(BUILDDIR); bash srctxz/create_srctxz.sh)
	mkdir -p $(PKGDIR)
	mv $(BUILDDIR)/../../lwcl-src*.txz $(PKGDIR)

clean:
	rm -Rf $(TXZPREPDIR) $(PKGDIR)

$(TXZPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(TXZPREPDIR)
	sed -e "s/\$${lwcl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${lwcl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${lwcl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${lwcl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    $< > $@
