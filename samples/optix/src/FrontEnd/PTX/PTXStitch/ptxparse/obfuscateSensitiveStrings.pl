#!/bin/perl
#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.
#
# LWIDIA_COPYRIGHT_END

###############################################################################
###############################  MAIN PROGRAM  ################################
use strict;
use Getopt::Long;

my (%CmdLineOption, @strToHide, %encodeMap, $inputFile, $outputFile, $content);

init_elw();
extract_sensitive_strings();
populate_encode_map();
encode_sensitive_strings();

############################  END OF MAIN PROGRAM  ############################
###############################################################################

sub extract_sensitive_strings {
    open (INPUTFILE, "$inputFile") || die "Cannot open $inputFile.$!\n";
    local $/;
    $content = <INPUTFILE>;
    @strToHide = $content =~ m/__obfuscate\(\"(.+)\"\)/g;
    close (INPUTFILE);
}

sub populate_encode_map {
    for my $element (@strToHide) {
        encode_string($element);
    }
}

# The corresponding __deObfuscate is defined in the `ptxMacroUtils.c` file. 
# Changes to the definition of the following function needs to be
# propagated in the `__deObfuscate` function (defined in the above file).

sub encode_string {
    my ($origName) = @_;
    my $encodedName = "";
    my $char;
    foreach $char (split //, $origName) {
        # compute ROT13 of char
        if (($char ge 'A' and $char le 'M') or ($char ge 'a' and $char le 'm')) {
            $encodedName .= chr(ord($char) + 13);
        } elsif (($char ge 'N' and $char le 'Z') or ($char ge 'n' and $char le 'z')) {
            $encodedName .= chr(ord($char) - 13);
        } else {
            $encodedName .= $char;
        }
    }
    $encodeMap{$origName} = $encodedName;
}

sub encode_sensitive_strings {
    open (OUTPUTFILE, "> $outputFile") || die "Cannot open $outputFile.$!\n";
    foreach my $key (keys %encodeMap) {
        $content =~ s/__obfuscate\(\"$key\"\)/\"$encodeMap{$key}\"/g; 
    }
    print OUTPUTFILE $content;
    close (OUTPUTFILE);
}

sub init_elw {
    GetOptions(\%CmdLineOption,
        "i=s",        # Input file containing sensitive strings inside HIDE(...)
        "o=s",        # Output file name containing encoded strings
        "help",       # Show help message
    );
    if ($CmdLineOption{i})           { $inputFile  = $CmdLineOption{i};       }
    if ($CmdLineOption{o})           { $outputFile = $CmdLineOption{o};       }
    if ($CmdLineOption{help})        { print_readme(); exit 0;                }
}

sub print_readme {
    print<<'EndOfReadme';

===============================================================================
================================    README    =================================

This script is used to obfuscate sensitive strings. The user of the script needs
to annotate the sensitive strings with __obfuscate(..) 

-------------------------------------------------------------------------------
usage:
  perl obfuscateSensitiveStrings.pl [options]
       -i       <file>        : Input file containing sensitive strings annotated
                                with __obfuscate(...)
       -o       <file>        : Output file name which contains obfuscated version
                                of sensitive strings
       -help                  : Print this messege.

-------------------------------------------------------------------------------
Description:
------------

1. The overall mechanism works in following steps:
        (a) The user of the script annotates sensitive strings (eg modifier 
            string literals) with __obfuscate().
            For eg: Sparsity modifier from ptxIRdefs.h looks: __obfuscate(".sp")
        (b) The user also ensures that __deObfuscate() function is called when the 
            original sensitive string needs to be recovered. 
            For eg: In case of modifier strings, `get*AsString` function -- (just 
            before returning modifier string/s) -- ilwokes __deObfuscate().

2. The perl script transforms (user-annotated) input file so as to avoid 
   oclwrances of sensitive string literals. The script obfuscates sensitive 
   strings with ROT13 technique.

3. The __deObfuscate function defined in ptxMacroUtils.c simply recovers the 
   original sensitive string from it's obfuscated version.

4. The ROT13 algorithm can obfuscate only `character` strings. Therefore, the 
   other types of characters (like digits, special characters) are kept intact
   by the implementation of (de-)obfuscator.
   
===============================================================================
EndOfReadme
}

###############################################################################
###############################  END OF SCRIPT  ###############################
###############################################################################
