#!/bin/perl
#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.
#
# LWIDIA_COPYRIGHT_END

use strict;
use Digest::MD5 qw(md5_hex);
use Getopt::Long;

my $extins;
my $InstrDefs;
my %CmdLineOption;

GetOptions(\%CmdLineOption,
        "i=s",           # Input file containing PTX instruction templates
);

if ($CmdLineOption{i})           { $InstrDefs = $CmdLineOption{i};       }

print "${InstrDefs}\n";
open(IRdefsTable, $InstrDefs) or die "Couldn't open file\n $!";

# Identify the extended instructions, gather and clean there strings.
# Format of an instruction string in ptxInstructionDefs.table
# DEFINE(Instruction type(s), InstructionName(s), Argument(s), Feature(s), Type of instruction(s)(EXTENDED/STANDARD))
# This loop extracts Instruction type(s), InstructionName(s), Argument(s), Feature(s)
# to be used for callwlating the hash

while (<IRdefsTable>) {
    print $_;
    if (/\(.*EXTENDED.*\)/) {
        s/\s*//g;
        s/DEFINE//g;
        s/\(//g;
        s/\)//g;
        s/,//g;
        s/\|//g;
        $extins .= $_;
    }
}
close (IRdefsTable) || die "Couldn't close the file properly\n";

print "\nString for hash\n${extins}\n\n";

my $hash = md5_hex($extins);
my $token = substr $hash, 0, 8;

print "TOKEN 0x${token}\n";
