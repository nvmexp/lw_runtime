#
# _LWRM_COPYRIGHT_BEGIN_
#
# Copyright 2012 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# _LWRM_COPYRIGHT_END_
#

MAN1 = \
	drf-lookup.1

TARGETS = \
	$(addprefix man/man1/,$(MAN1)) \
	$(addprefix html/,$(addsuffix .html,$(MAN1)))

all: $(TARGETS)

man/man1/%: asciidoc/%.txt
	a2x -f manpage -D $(dir $@) $<

html/%.html: asciidoc/%.txt
	asciidoc -d manpage -a quirks -o $@ $<
