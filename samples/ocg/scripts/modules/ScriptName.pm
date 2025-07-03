package modules::ScriptName;  # assumes modules/ScriptName.pm

use strict;
use warnings;

require modules::PathUtils;
use modules::PathUtils;

BEGIN {

	require Exporter;
        require File::Basename;

	# set the version for version checking
	our $VERSION     = 1.00;

	# Inherit from Exporter to export functions and variables
	our @ISA         = qw(Exporter);

	# Functions and variables which are exported by default
	our @EXPORT      = qw($scriptName $scriptDir);

	# Functions and variables which can be optionally exported
	our @EXPORT_OK   = qw( );
}
# exported package globals go here
use File::Basename;
our $scriptName = basename($0);
our $scriptDir  = &AbsPath(dirname($0));

END { }       # module clean-up code here (global destructor)

1;  # don't forget to return a true value from the file
