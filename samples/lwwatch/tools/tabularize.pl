#!/usr/bin/perl
#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2010 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#

use strict;

use FileHandle;

#
# This perl script colwerts the flat function signal dumping program into
# a table driven signal dumping program.
#
# Based on transform.pl used for earlier versions of sigdump.(pre-fermi)
# Signal format in the input file now tells us how many bits to read for a particular signal
# belonging to an instance class.
#
my $numWrites;
my $regWriteDef;
my $sigDumpDef;
my $file_h;
my $chip;
my $lastStr;
my $reg;
my $regDef;
my $regName;
my $numFields;
my $fieldDef;
my $numSigs;

if (scalar @ARGV != 2)
{
    my $script;

    ($script = $0) =~ s#.*/##;
    print "Usage: \n  $script FILENAME.c --regsinfo  //For ouputing the register table\n";
    print "$script FILENAME.c --siginfo  //For outputing the signal table\n";
    print "Example: $script filename_gf100.c --siginfo > signal_gf100.sig\n";
    exit(1);
}

if ($ARGV[0] =~ /_([^_]*)\.c/)
{
    $chip = uc($1);
}
else
{
    die "Cannot figure out the chip for file $ARGV[0]\n";
}

$file_h = new FileHandle("<$ARGV[0]");
die "Cannot open $ARGV[0]: $!\n" unless defined $file_h;

$numWrites = 0;
$regWriteDef = "";
$sigDumpDef = "";
$lastStr = "";
$fieldDef = "";
$regDef = "";
$numSigs = 0;
while (<$file_h>)
{
    if (/^\s*RegWrite\(\s*([^,]+),\s*([^,]+),\s*([^\)]+)\)/)
    {
        $regWriteDef .= "$1 $2 $3 \n";
        $numWrites++;
    }
    elsif (/^\s*OutputSignal\(\s*fp,\s*\"([^\"]+:)\s*\",\s*RegBitRead\(\s*([^,]+),\s*([^\)]+)\,\s*([^\)]+)\)/)
    {
        #
        # Some of the original files produced exactly the same signal
        # multiple times in a row.
        #
        if ($1 eq $lastStr)
        {
            next;
        }
        $sigDumpDef .= "$1 $2 $3 $4 $numWrites \n";
        $numWrites = 0;
        $lastStr = $1;
        $numSigs++;
    }
    elsif (/^\s*(RegWrite|OutputSignal)/)
    {
        die "$ARGV[0]:$.: Misparsed line $_";
    }
    elsif (/^\s*r\s*=\s*REG_RD32\(\s*([^\)]+)\s*\)/)
    {
        if ($reg)
        {
            # Finish the last choice
            $regDef .= "    { \"$regName\", $reg, $numFields },\n";
        }
        $reg = $1;
    }
    elsif (/^\s*fprintf\(\s*fp\s*,\s*\"(\S+)\s/)
    {
        $regName = $1;
        $numFields = 0;
    }
    elsif (/^\s*fprintf\(\s*fp\s*,\s*\"\s+(\S+)\s.*\",\s*.*\>\>\s*(\S+)\)\s*\&\s*(\S+)\s*\)\;\s*$/)
    {
        $fieldDef .= "    { \"$1\", $2, $3 },\n";
        $numFields++;
    }
    elsif (/^\s*fprintf\(/)
    {
        die "$ARGV[0]:$.: Misparsed fprintf line $_";
    }
}
if ($reg)
{
    # Finish the last choice
    $regDef .= "    { \"$regName\", $reg, $numFields },\n";
}
$file_h->close();


if($ARGV[1] eq "--regsinfo")
{
print $regWriteDef;
}
elsif($ARGV[1] eq "--siginfo")
{
print $sigDumpDef;
}

