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
# lwwatch-config file that specifies all known lwwatch-config features
#

package Features;
use warnings 'all';
no warnings qw(bareword);       # barewords makes this file easier to read
                                # and not so important for error checks here

use Carp;                       # for 'croak', 'cluck', etc.

use Groups;                     # Features is derived from 'Groups'

@ISA = qw(Groups);

# lwwatch-config does not manage Features, so this list is empty
my $featuresRef = [ 

      # platform lwwatch will run on.
      # These are enabled by the makefiles
      PLATFORM_UNKNOWN =>
       {
         DESCRIPTION         => "Running on an unknown platform",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_WINDOWS =>
       {
         DESCRIPTION         => "Running on Windows",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_WINDOWS_XP =>
       {
         DESCRIPTION         => "Running on Windows XP",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_WINDOWS_LDDM =>
       {
         DESCRIPTION         => "Running on Windows LDDM",
         DEFAULT_STATE       => DISABLED,
         ALIASES             => [ PLATFORM_WINDOWS_VISTA ],
       },
      PLATFORM_UNIX =>
       {
         DESCRIPTION         => "Running on Unix",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_DOS =>
       {
         DESCRIPTION         => "Running on DOS/DJGPP",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_SIM =>
       {
         DESCRIPTION         => "Running on Simulator",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_OSX =>
       {
         DESCRIPTION         => "Running on OSX",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_QNX =>
       {
         DESCRIPTION         => "Running on QNX",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_MODS =>
       {
         DESCRIPTION         => "Running as part of MODS",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_MODS_WINDOWS =>
       {
         DESCRIPTION         => "Running as part of MODS on Windows",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_MODS_UNIX =>
       {
         DESCRIPTION         => "Running as part of MODS on UNIX",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_MODS_OSX =>
       {
         DESCRIPTION         => "Running as part of MODS on OSX",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_MODS_DOS =>
       {
         DESCRIPTION         => "Running as part of MODS on DOS",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_MODS_DJGPP =>
       {
         DESCRIPTION         => "Running as part of MODS on DJGPP",
         DEFAULT_STATE       => DISABLED,
       },

      # Misc features 
      UNIX_USERMODE =>
       {
         DESCRIPTION         => "Unix usermode",
         DEFAULT_STATE       => DISABLED,
       },
      UNIX_MMAP =>
       {
         DESCRIPTION         => "Features specific to mmap",
         DEFAULT_STATE       => DISABLED,
       },
      UNIX_JTAG =>
       {
         DESCRIPTION         => "Features specific to jtag",
         DEFAULT_STATE       => DISABLED,
       },
      UNIX_HWSNOOP =>
       {
         DESCRIPTION         => "Features specific to hwsnoop",
         DEFAULT_STATE       => DISABLED,
       },
];

# Create the Features group
sub new {

    @_ == 1 or croak 'usage: obj->new()';

    my $type = shift;

    my $self = Groups->new("feature", $featuresRef);

    $self->{GROUP_PROPERTY_INHERITS} = 1;   # FEATUREs inherit based on name

    return bless $self, $type;
}

# end of the module
1;
