#************************ BEGIN COPYRIGHT NOTICE ***************************#
#                                                                           #
#          Copyright 2009 (c) LWPU Corporation.  All rights reserved.          #
#                                                                           #
# All information contained herein is proprietary and confidential to       #
# LWPU Corporation.  Any use, reproduction, or disclosure without the     #
# written permission of LWPU Corporation is prohibited.                   #
#                                                                           #
#************************** END COPYRIGHT NOTICE ***************************#
#
#
# lwwatch-config file that specifies all known lwflcndbg engines
#

package Engines;
use warnings 'all';
no warnings qw(bareword);       # barewords makes this file easier to read
                                # and not so important for error checks here

use Carp;                       # for 'croak', 'cluck', etc.

use Groups;                     # Engines is derived from 'Groups'

@ISA = qw(Groups);

#
# The actual engine definitions.
# This list contains all engines that lwwatch-config is aware of.
#
my $enginesRef = [

        PMU =>
        {
        },
        DPU =>
        {
        },
        SOCBRDG =>
        {
        },
        TEGRASYS =>
        {
        },
];


# Create the item group
sub new {

    @_ >= 1 or croak 'usage: obj->new()';

    my $type = shift;

    my $self = Groups->new("engine", $enginesRef);
    
    return bless $self, $type;
}

# end of the module
1;
