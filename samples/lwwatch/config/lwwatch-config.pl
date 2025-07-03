#!/usr/bin/perl -w

use strict;
use warnings "all";

use File::Basename;

# Adding the directory path of lwwatch-config.pl ( __FILE__ ) to @INC so that
# we can access default config file - lwwatch-config.cfg
BEGIN {
    unshift @INC, dirname __FILE__      if (__FILE__ && -f __FILE__);
}

my $dir = dirname __FILE__;
my $chipConfig = "$dir/../../../drivers/common/chip-config/chip-config.pl";

push @ARGV, '--mode', 'lwwatch-config';
push @ARGV, '--config', 'lwwatch-config.cfg';

if (!grep {$_ eq '--source-root'} @ARGV) {
    push @ARGV, '--source-root', "$dir/../";
}


# append --output-dir if not specified
if (!grep {$_ =~ m/--output-dir/} @ARGV) {
    # create deafult output directory
    my $outDir = '_out';
    if (not -e $outDir) {
        print "[lwwatch-config] creating output directory '$outDir'\n";
        mkdir($outDir)      or die "Unabled to create directory '$outDir'\n";
    }
    push @ARGV, '--output-dir', '_out';
}

require $chipConfig;
