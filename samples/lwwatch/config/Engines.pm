#************************ BEGIN COPYRIGHT NOTICE ***************************#
#                                                                           #
#    Copyright 2009-2017 (c) LWPU Corporation.  All rights reserved.      #
#                                                                           #
# All information contained herein is proprietary and confidential to       #
# LWPU Corporation.  Any use, reproduction, or disclosure without the     #
# written permission of LWPU Corporation is prohibited.                   #
#                                                                           #
#************************** END COPYRIGHT NOTICE ***************************#
#
#
# lwwatch-config file that specifies all known lwwatch engines
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

        CE =>
        {
        },
        CLK =>
        {
        },
        DISP =>
        {
        },
        DPAUX =>
        {
        },
        FALCON =>
        {
        },
        FB =>
        {
            DESCRIPTION => "Frame Buffer",
        },
        FBFLCN =>
        {
           DESCRIPTION  => "FBFalcon engine for GV100+",
           CHIPS_SUPPORTED      => [ VOLTA_and_later, ],

        },
        FIFO =>
        {
        },
        GR =>
        {
           DESCRIPTION  => "Graphics",
        },
        GSP =>
        {
           DESCRIPTION  => "GSP for Volta+",
           CHIPS_SUPPORTED      => [ Volta_and_later, ],
        },
        OFA =>
        {
           DESCRIPTION  => "OFA for Ampere+",
           CHIPS_SUPPORTED      => [ Ampere_and_later, ],
        },
        INSTMEM =>
        {
        },
        BUS =>
        {
        },
        BIF =>
        {
        },
        CIPHER =>
        {
        },
        PMU =>
        {
        },
        DPU =>
        {
        },
        PMGR =>
        {
        },
        MMU =>
        {
        },
        MSDEC =>
        {
        },
        SIG =>
        {
        },
        HWPROD =>
        {
        },
        VIC =>
        {
        },
        VIRT =>
        {
        },
        VMEM =>
        {
        },
        MSENC =>
        {
        },
        HDA =>
        {
        },
        MC =>
        {
        },
        PRIV =>
        {
        },
        ELPG =>
        {
        },
        SMBPBI =>
        {
        },
        TEGRASYS =>
        {
        },
        FECS =>
        {
        },
        LWDEC =>
        {
            DESCRIPTION => "Unified Video Decode Engine on Maxwell+ chips",
        },
        LWJPG =>
        {
            DESCRIPTION => "JPEG Picture Decode Engine on TU102+ chips",
        },
        ACR =>
        {
            DESCRIPTION => "Access Controlled Region on GM20X+ chips",
        },
        PSDL =>
        {
            DESCRIPTION => "PrivSec Debug License on GM20X+ chips",
        },
        SEC2 =>
        {
            DESCRIPTION => "SEC2 on GM10X+ chips",
        },
        FALCPHYS =>
        {
        },
        LWLINK =>
        {
            DESCRIPTION => "Discovers LWLINKS and displays their information",
        },

        VPR =>
        {
            DESCRIPTION => "Video Protected Region on GP102+ chips",
        },
        RISCV =>
        {
            DESCRIPTION => "RISC-V extension",
            CHIPS_SUPPORTED      => [ Turing_and_later, ],
        },
        INTR =>
        {
            DESCRIPTION => "Interrupts extension",
            CHIPS_SUPPORTED     => [ Turing_and_later, ],
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
