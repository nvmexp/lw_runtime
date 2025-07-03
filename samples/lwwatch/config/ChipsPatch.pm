#************************ BEGIN COPYRIGHT NOTICE ***************************#
#                                                                           #
#          Copyright (c) LWPU Corporation.  All rights reserved.          #
#                                                                           #
# All information contained herein is proprietary and confidential to       #
# LWPU Corporation.  Any use, reproduction, or disclosure without the     #
# written permission of LWPU Corporation is prohibited.                   #
#                                                                           #
#************************** END COPYRIGHT NOTICE ***************************#

package ChipsPatch;
use warnings 'all';
use strict;

#
# ChipsPatch.pm is the patch file for Chips.pm.  It overrides fields for matched
# chips.  'FLAGS' is the only supported field so far.  Chip-config gives errors
# for non-supported fields or if no matched chip found. 
#

our $chipsPatchRef = [

    #
    # Please keep the list as short as possible
    #

    # Example Usage:
    #
    # T124 => {
    #     FLAGS  => '',   # override ':OBSOLETE' to force the chip enabled
    # },

];  # chip aliases


# return the patch chips defined above
sub new {
    return {
        CHIPS    => $chipsPatchRef,
    };
}

1;
