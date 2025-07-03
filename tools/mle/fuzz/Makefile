#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2020 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#

libfuzzer = 1

ifeq ($(libfuzzer),1)
CFLAGS += -fsanitize=fuzzer,address
CFLAGS += -DFUZZ_MLE=1
else
CFLAGS += -DFUZZ_MLE=0
endif
CFLAGS += -g

fuzz_mle: ../mle.c
	@$(CC) ../mle.c $(CFLAGS)  -o $@

clean:
	@rm -f fuzz_mle

.PHONY: clean
