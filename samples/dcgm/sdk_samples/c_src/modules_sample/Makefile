# Copyright 1993-2018 LWPU Corporation.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.

IDIR := /usr/include
CXXFLAGS = -I $(IDIR)

LDIR := /usr/lib
LDFLAGS= -L$(LDIR) -ldcgm

modules_sample: modules_sample.o
	$(CXX) -o $@ $< $(CXXFLAGS) $(LDFLAGS)

clean:
	-@rm -f modules_samples.o
	-@rm -f modules_sample
