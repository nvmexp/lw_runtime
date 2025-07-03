################################################################################
#
# Copyright 1993-2015 LWPU Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to LWPU ownership rights under U.S. and
# international Copyright laws.
#
# LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms
#
################################################################################

# Target rules
all:
	+$(MAKE) -C configuration_sample
	+$(MAKE) -C health_diag_sample
	+$(MAKE) -C process_stats_sample
	+$(MAKE) -C policy_sample
	+$(MAKE) -C field_value_sample

clean:
	+$(MAKE) clean -C configuration_sample
	+$(MAKE) clean -C health_diag_sample
	+$(MAKE) clean -C process_stats_sample
	+$(MAKE) clean -C policy_sample
	+$(MAKE) clean -C field_value_sample
