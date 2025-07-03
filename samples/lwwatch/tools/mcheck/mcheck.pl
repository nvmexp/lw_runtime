

#!/usr/bin/perl
#
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
# File Name:   mcheck.pl
#
# Mcheck validates that the the Lwwatch HAL is not sharing
# manual defines among shared code.
#
# See the wiki for full documentation
#    https://wiki.lwpu.com/engwiki/index.php/Mcheck
#

#===========================================================
package LwWatchMcheck;

use File::Basename;

BEGIN
{
    # HACK : add rm mcheck dir @INC path
    # for using mcheck moduels without '-Ic:/long/path/to/mcheck' in command line
    my $pwdDir = dirname __FILE__;
    unshift @INC, $pwdDir.'/../../../../drivers/common/shared/mcheck';
}

use Mcheck;
our @ISA;
@ISA = qw( Mcheck );                 # Message class has error, warning, verbose methods


#
# Initialize client info for when running mcheck on the RM HAL
#
sub setup_client_info
{
    my $self = shift;

    my %client_info = (
        NAME            => 'LwWatch',
        CONFIG_FILE     => $self->{LW_ROOT}.'/apps/lwwatch/tools/mcheck/mcheck.config',

        RMCFG_PATH      => $self->{LW_ROOT}.'/drivers/common/chip-config/chip-config.pl',
        RMCFG_HEADER    => 'lwwatch-config.h',
        RMCFG_PROFILE   => 'default',
        RMCFG_SRC_PATH  => $self->{LW_ROOT}.'/apps/lwwatch',
        RMCFG_ARGS      => '--mode lwwatch-config '.
                           '--config '.$self->{LW_ROOT}.'/apps/lwwatch/config/lwwatch-config.cfg ',

        SCAN_CPP_IMACROS => '',     # set from build_drfheader_rm()
    );

    $self->{CLIENT}   = \%client_info;
}

sub generate_headers
{
    my $self = shift;

    $self->generate_headers_default();

    # build lwwatchdrf.h and appand it to $self->{CLIENT}->{SCAN_CPP_IMACROS}
    $self->build_drfheader_lwwatch();
}

sub build_drfheader_lwwatch
{
    my $self = shift;

    my $sdk_inc = $self->{LW_ROOT}."/sdk/lwpu/inc";
    my $drfheader_cpp_args;
    my $tmpHeaderDir = $self->get_tempfile_dir();
    my $kernel_inc = $self->{LW_ROOT}."/drivers/resman/kernel/inc";
    my $hwref_inc  = $self->{LW_ROOT}."/drivers/common/inc/hwref";

    open DRF_FILE, ">$tmpHeaderDir/lwwatchdrf.h";
    #
    # RM register access macros takes pGpu parameters while lwwatch doesn't. 
    #      RM usage : GPU_REG_RD32(pGpu, LW_PDISP_REGISTER);
    # Lwwatch usage : GPU_REG_RD32(LW_PDISP_REGISTER);
    #
    # undef those macros to avoid parsing error
    print DRF_FILE << "__EOL__";
#ifdef GPU_REG_RD32
#undef GPU_REG_RD32
#endif

#ifdef GPU_REG_WR32
#undef GPU_REG_WR32
#endif

#ifdef GPU_REG_RD16
#undef GPU_REG_RD16
#endif

#ifdef GPU_REG_WR16
#undef GPU_REG_WR16
#endif

#ifdef GPU_REG_RD08
#undef GPU_REG_RD08
#endif

#ifdef GPU_REG_WR08
#undef GPU_REG_WR08
#endif

__EOL__
    close DRF_FILE;

    $self->{CLIENT}->{SCAN_CPP_IMACROS} .= "-imacros $tmpHeaderDir/lwwatchdrf.h ";
}


#===========================================================
package main;

use Cwd qw(abs_path);
use File::Basename;

# get $LW_ROOT from the relative path of this file 'mcheck.pl'
my $lwroot_from_pwd  = abs_path((dirname __FILE__).'/../../../..');
my $srcroot_from_pwd = (dirname __FILE__).'/../..';

#
# Exelwtion starts here
#
our $MCHECK = LwWatchMcheck->new();

$MCHECK->init($lwroot_from_pwd,     # init hash {opt} and paths
              $srcroot_from_pwd);

$MCHECK->mcheck();

