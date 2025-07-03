package modules::PathUtils;  # assumes modules/PathUtils.pm

use strict;
use warnings;

BEGIN {
	require Exporter;

	# set the version for version checking
	our $VERSION     = 1.00;

	# Inherit from Exporter to export functions and variables
	our @ISA         = qw(Exporter);

	# Functions and variables which are exported by default
	our @EXPORT      = qw(SlashPath WinPath DirName BaseName AbsPath RemovePathDots IsRelativePath);

	# Functions and variables which can be optionally exported
	our @EXPORT_OK   = qw( );
}

# exported package globals go here

# make all your functions, whether exported or not;

sub SlashPath($)
{
    my ($path) = @_;
    $path =~ s/\\/\//g;
    $path =~ s/\/\.\//\//g;
    $path =~ s/^\.\///;
    return $path;
}

sub WinPath($)
{
    my ($path) = @_;
    $path =~ s/\/cygdrive\/([a-zA-Z])\//$1:\//;
    $path =~ s/^([A-Za-z]):/\L$1:/;
    return $path;
}

sub DirName($)
{
    my ($path) = @_;
    $path =~ &RemovePathDots(&SlashPath(&WinPath($path)));
    $path =~ s/\/$//;
    $path =~ s/\/?[^\/]*$//;

    return $path;
}

sub BaseName($)
{
    my ($path) = @_;
    $path = &SlashPath($path);
    $path =~ s/\/$//;
    $path =~ s/.*\///g;

    return $path;
}

sub RemovePathDots($)
{
    my ($path) = @_;
    while ($path =~ /\/[^\s\/]+\/\.\.\//) {
        $path =~ s/\/[^\s\/]+\/\.\.\//\//g;
    }
    return $path;
}

sub AbsPath($@)
{
    my ($path, @optional) = @_;
    my $cwd;
    $cwd = shift @optional if (@optional);

    if (!&IsRelativePath($path)) {
        return &RemovePathDots(&WinPath(&SlashPath($path)));
    } else {
        {
            use Cwd;
            $cwd = getcwd() if (!defined $cwd);
        }
        $path = join '/', $cwd, $path;

        $path = &RemovePathDots(&WinPath(&SlashPath($path)));
        return $path;
    }
}

sub IsRelativePath($)
{
    my ($path) = @_;

    return ($path =~ /^(\/|[a-zA-Z]:[\/\\])/) ? 0 : 1;
}

END { }       # module clean-up code here (global destructor)

1;  # don't forget to return a true value from the file
