#
# Copyright (c) 2015-2019, LWPU CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../../makefiles/common.mk
include ../../makefiles/version.mk
BUILDDIR ?= $(abspath ../../build)
TXZPREPDIR  := $(BUILDDIR)/txz
PKGDIR  := $(BUILDDIR)/pkg/txz/

TXZGEN_IN  := $(wildcard *.in)
TXZGEN     := $(TXZGEN_IN:.in=)
TXZTARGETS := $(patsubst %, $(TXZPREPDIR)/%, $(TXZGEN))

PKG_ARCH   := $(shell uname -m)

prep: $(TXZTARGETS)
	$(MAKE) -C ../.. lic BUILDDIR=$(BUILDDIR)

build: prep
	$(MAKE) -C ../.. src.build BUILDDIR=$(BUILDDIR)
	@printf "Building tar.xz package\n"
	(cd $(BUILDDIR); bash txz/create_txz.sh)
	mkdir -p $(PKGDIR)
	mv $(BUILDDIR)/../lwcl*.txz $(PKGDIR)

clean:
	rm -Rf $(TXZPREPDIR) $(PKGDIR)

$(TXZPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(TXZPREPDIR)
	sed -e "s/\$${lwcl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${lwcl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${lwcl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${lwcl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${lwca:Major}/$(LWDA_MAJOR)/g" \
	    -e "s/\$${lwca:Minor}/$(LWDA_MINOR)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    $< > $@
