package modules::Error;  # assumes modules/Error.pm

use strict;
use warnings;

BEGIN {

	require Exporter;

	# set the version for version checking
	our $VERSION     = 1.00;

	# Inherit from Exporter to export functions and variables
	our @ISA         = qw(Exporter);

	# Functions and variables which are exported by default
	our @EXPORT      = qw(Error);

	# Functions and variables which can be optionally exported
	our @EXPORT_OK   = qw();
}

# make all your functions, whether exported or not;
# remember to put something interesting in the {} stubs
sub Error
{
	my $msg = shift @_;
	print "ERROR: $msg\n";
	exit 1;
}

END { }       # module clean-up code here (global destructor)

1;  # don't forget to return a true value from the file
