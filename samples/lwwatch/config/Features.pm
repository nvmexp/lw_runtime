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

my $featuresRef = [ 

      # platform lwwatch will run on.
      # These are enabled via profile setting in lwwatch-config.cfg 
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
      WINDOWS_STANDALONE =>
       {
         DESCRIPTION         => "Windows Standalone build",
         DEFAULT_STATE       => DISABLED,
       },

      PLATFORM_UNIX =>
       {
         DESCRIPTION         => "Running on Unix",
         DEFAULT_STATE       => DISABLED,
       },
      PLATFORM_MODS =>
       {
         DESCRIPTION         => "Running as part of MODS",
         DEFAULT_STATE       => DISABLED,
       },

      # UNIX builds
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
      UNIX_MOBILE =>
       {
         DESCRIPTION         => "Features specific to mobile",
         DEFAULT_STATE       => DISABLED,
       },
      # MODS builds
      MODS_UNIX =>
       {
         DESCRIPTION         => "MODS on Unix",
         DEFAULT_STATE       => DISABLED,
       },
      MODS_WINDOWS =>
       {
         DESCRIPTION         => "MODS on Windows",
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
