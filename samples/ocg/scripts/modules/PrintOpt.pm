package modules::PrintOpt;  # assumes modules/PrintOpt.pm

use strict;
use warnings;

BEGIN {
	require Exporter;

	# set the version for version checking
	our $VERSION     = 1.00;

	# Inherit from Exporter to export functions and variables
	our @ISA         = qw(Exporter);

	# Functions and variables which are exported by default
	our @EXPORT      = qw(PrintOpt PrintOptWrap PrintOptConfig);

	# Functions and variables which can be optionally exported
	our @EXPORT_OK   = qw( );
}

# exported package globals go here
my $limit  = 80;
my $indent = 24;
my $prefix = "  ";

# make all your functions, whether exported or not;
sub PrintOptConfig($)
{
    my ( $cfg ) = @_;

    if (exists $cfg->{limit}) {
        $limit = $cfg->{limit};
    }

    if (exists $cfg->{indent}) {
        $indent = $cfg->{indent};
    }

    if (exists $cfg->{prefix}) {
        $prefix = $cfg->{prefix};
    }
}

sub PrintOptWrap($$@)
{
    my ($line, $fmt, @args) = @_;

    my $str = @args ? (sprintf "$fmt", @args) : $fmt;
    my @words = split /\s+/, $str;

    my $justPrinted;
    for my $w (@words) {
        $justPrinted = 0;
        if (((length $line) + (length $w) + 1) > $limit) {
            print "$line\n";
            $justPrinted = 1;
            $line = "";
            if ((length $w) + $indent < $limit) {
                for (my $i=0; $i<$indent; $i++) {
                    $line .= ' ';
                }
            }
            $line .= $prefix;
        }

        if (length $line != 0 && !$justPrinted) {
            $line .= ' ';
        }
        $line .= "$w";
    }

    print "$line\n";
}

sub PrintOpt($$)
{
    my ( $opt, $descr ) = @_;

    my $line = "$prefix$opt";
    for (my $i=(length $line); $i<$indent; $i++) {
        $line .= ' ';
    }

    &PrintOptWrap($line, "%s", $descr);
    return;
}

END { }       # module clean-up code here (global destructor)

1;  # don't forget to return a true value from the file
