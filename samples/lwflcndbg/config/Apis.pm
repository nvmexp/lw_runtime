#************************ BEGIN COPYRIGHT NOTICE ***************************#
#                                                                           #
#          Copyright (c) LWPU Corporation.  All rights reserved.          #
#                                                                           #
# All information contained herein is proprietary and confidential to       #
# LWPU Corporation.  Any use, reproduction, or disclosure without the     #
# written permission of LWPU Corporation is prohibited.                   #
#                                                                           #
#************************** END COPYRIGHT NOTICE ***************************#
#
#
# lwwatch-config file that specifies all known lwwatch-config api calls.
#
# This file is used by rmconfig.pl.
#

package Apis;
use warnings 'all';
no warnings qw(bareword);       # barewords makes this file easier to read
                                # and not so important for error checks here

use Carp;                       # for 'croak', 'cluck', etc.

use Groups;                     # Apis is derived from 'Groups'

@ISA = qw(Groups);

# lwwatch-config does not manage API's, so this list is empty
my $apisRef = [
];


# Create the item group
sub new {

    @_ == 1 or croak 'usage: obj->new()';

    my $type = shift;

    my $self = Groups->new("api", $apisRef);

    return bless $self, $type;
}

# end of the module
1;
