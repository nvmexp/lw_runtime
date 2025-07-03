package modules::Utils;  # assumes modules/Utils.pm

use strict;
use warnings;

BEGIN {

	require Exporter;

	# set the version for version checking
	our $VERSION     = 1.00;

	# Inherit from Exporter to export functions and variables
	our @ISA         = qw(Exporter);

	# Functions and variables which are exported by default
	our @EXPORT      = qw(RunCmd RefString);

	# Functions and variables which can be optionally exported
	our @EXPORT_OK   = qw();
}

# make all your functions, whether exported or not;
# remember to put something interesting in the {} stubs
sub RunCmd($@)
{
    my ($cmd, @optionalArgs) = @_;

    my $optionsHash = @optionalArgs ? shift @optionalArgs : {};
    my $outnull = exists $optionsHash->{outnull} && $optionsHash->{outnull};

    # Echo the command being run inf requested by user.
    if (exists $optionsHash->{echo} && $optionsHash->{echo}) {
        print "$cmd\n";
    }

    # Suppress output if requested by the user.
    my $oldStdout;
    my $oldStderr;
    if ($outnull) {
        # Redirect STDOUT and STDERR file handles to null.
        my $null = $^O eq 'MSWin32' ? 'nul' : '/dev/null';
        open $oldStdout, ">&STDOUT"; # dup the stdout file handle.
        open $oldStderr, ">&STDERR"; # dup the stderr file handle.
        close STDOUT;
        close STDERR;
        open STDOUT, '>', $null;
        open STDERR, '>', $null;
    }

    system("$cmd");
    my $returnStatus = $?;

    if ($outnull) {
        # Restore original STDOUT & STDERR file handles.
        close STDOUT;
        close STDERR;
        open STDOUT, '>&', $oldStdout; # restore dup'd stdout file handle.
        open STDERR, '>&', $oldStderr; # restore dup'd stderr file handle.
    }

    # Return the status of the command exelwtion.
    return $returnStatus;
}

sub RefString($)
{
    my ($item) = @_;

    my $typeof = ref($item);

    if ($typeof eq 'ARRAY') { 
        my $ret = "( ";
        my $first = 1;
        for my $ii (@$item) {
            if (!$first) {
                $ret .= ", ";
            }
            $ret .= &RefString(\$ii);
            $first = 0;
        }
        $ret .= ")";
        return $ret;
    }

    elsif ($typeof eq 'SCALAR' || $typeof eq 'VSTRING') {
        return "$$item";
    }

    elsif ($typeof eq 'REF') {
        return &RefString($$item);
    }

    else {
        return "$item";
    }
}

END { }       # module clean-up code here (global destructor)

1;  # don't forget to return a true value from the file
