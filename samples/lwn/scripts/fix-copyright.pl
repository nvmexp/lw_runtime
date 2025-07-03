#!/usr/bin/elw perl
#
# fix-copyright.pl:  Scans all files in the current directory looking for
# LWPU copyrights, and replaces them with Hovi copyrights.  Used to
# simplify synchronizing files from LWPU internal repositories to the
# shared LWN repository. The Hovi copyright to use is passed as cmd line
# argument.
#
use strict;
use File::Find;

my $foundLWCopyrights = 0;

my $Hovi_Copyright;

# Utility code to check for LWPU copyrights and rewrite the file if found.
sub process_file()
{
    my $fn = $_;
    return if ! -f $fn;
    return if $fn !~ m/\.[ch](pp)?$/;

    # Slurp in the file.
    # Using utf-8 encoding will work for ASCII and utf-8 inputs
    # since the encoding of 1-byte characters are identical.
    if (! open FILE, "<:encoding(UTF-8)", $fn) {
        print STDERR "WARNING:  Could not open $fn.\n";
        return;
    }
    my @lines = <FILE>;
    close FILE;

    # Look for LWPU copyrights and bail if not found.
    my $hasLWCopyright = 0;
    for (@lines) {
        if (m/Copyright.*LWPU Corporation/i) {
            $hasLWCopyright = 1;
            last;
        } elsif (m/copyright.*LWN 3D API/i) {
            $hasLWCopyright = 1;
            last;
        }
    }
    return if !$hasLWCopyright;

    $foundLWCopyrights = 1;

    # Rewrite the file, replacing comments that look like:
    #
    #    /*
    #    ** Copyright 1998-2015, LWPU Corporation.
    #    ** All Rights Reserved.
    #    **
    #    ** THE INFORMATION CONTAINED HEREIN IS PROPRIETARY ...
    #    ** LWPU, CORPORATION.  USE, REPRODUCTION OR     ...
    #    ** IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU,  ...
    #    */
    #
    # with the Hovi copyright.
    #
    if (!open FILE, ">:encoding(UTF-8)", $fn) {
        print STDERR "ERROR:  Could not open $fn for write.\n";
        return;
    }

    my @hold;
    my $hasCopyright = 0;
    my $bom="";

    for (@lines) {

        # If we're in the middle of a C-style comment block, look for LWPU
        # copyrights or the closing line.
        if (scalar(@hold)) {

            push @hold, $_;

            if (m/Copyright.*LWPU Corporation/i) {
                # If we find a copyright, record that fact.
                $hasCopyright = 1;
            } elsif (m/copyright.*LWN 3D API/i) {
                # If we find a copyright partner notice, record that fact.
                $hasCopyright = 1;
            } elsif (m/\*\//) {
                # If we find a block closure, spit out the held lines (if no
                # copyright) or the Hovi copyright.
                if ($hasCopyright) {
                    print FILE $bom;
                    print FILE $Hovi_Copyright;
                } else {
                    for (@hold) { print FILE $_; }
                }
                @hold = ();
                $hasCopyright = 0;
            }
        } elsif (m/^\p{C}?\s*\/\*(?!.*\*\/)/)  {
            # We start a replacement candidate if we find a C++ style comment
            # opener at the beginning of the line that is NOT followed by a
            # C-style comment closer.
            @hold = ($_);
            if (m/^\p{C}/) {
               # If the copyright is at the beginning of an utf-8 file there might
               # be a leading BOM. The BOM needs to be stored in order to write it
               # out with the new copyright.
               $bom = $&;
            }
        } else {
            print FILE $_;
        }
    }
    close FILE;
}

my $num_args = $#ARGV + 1;
if ($num_args != 1) {
    print "\nUsage: fix-copyright.pl <path to copyright file>\n";
    exit;
}
my $filename = $ARGV[0];
# Read Hovi copyright
open my $fh, '<', $filename or die "Can't open copyright file $!";
read $fh, $Hovi_Copyright, -s $fh;

# Replace Lwpu copyright with Hovi copyright
find(\&process_file, ".");

print "No LWPU copyrights found.\n" if !$foundLWCopyrights;

