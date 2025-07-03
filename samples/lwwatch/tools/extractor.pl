#!/usr/bin/perl -w
# _LWRM_COPYRIGHT_BEGIN_
#
# Copyright 2013 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
# 
# _LWRM_COPYRIGHT_END_
# 

#
# @file  extractor.pl
# @brief Generates the symbol file for falcon ucode.
# 

use File::Spec;
use File::Find;
use Storable qw(dclone);

@PMU_SW_DIR = ("dev", "gpu_drv", "chips_a", "pmu_sw");
@FALCON_RESTRICTED_DIR = ("dev", "gpu_drv", "chips_a", "tools", "restricted");
@DPU_SW_DIR = ("dev", "gpu_drv", "chips_a", "uproc");
$m_wantedFile = "";
@m_foundFiles = ();

&main();

sub main
{
    #arg = debug_info, symb_info, gen_xml
    if (@ARGV != 7)
    {
            die "wrong args.\nShould be: <falcon_engine> <decoded_line> <symbol> <frames> 
                <output_file> <sw_tree_path> <family> <chip>\n";
    }

    ($falcon_engine, $dbg_info_file, $dbg_symbol, $lw_symbl_file, $tree, $family, $chip) = @ARGV;

    if ($falcon_engine eq "pmu")
    {
        @searchDir = (File::Spec->catdir(@PMU_SW_DIR), File::Spec->catdir(@FALCON_RESTRICTED_DIR));
    }
    else
    {
        @searchDir = (File::Spec->catdir(@DPU_SW_DIR), File::Spec->catdir(@FALCON_RESTRICTED_DIR));
    }

    # load symbol from symbol file
    my ($func_hash, $func_cnt);
    my ($obj_hash, $obj_cnt);
    $func_cnt = 0;
    $obj_cnt = 0;

    load_symb($dbg_symbol, \%func_hash, \$func_cnt, \%obj_hash, \$obj_cnt,);
    generatePcLineMatchingTable();

    # Start generating the debug info file for lwwatch
    open OUT, ">$lw_symbl_file";

    # Generate function symbol table (function name, starting address, ending address (included))
    print OUT "func_table $func_cnt:\n";

    foreach $func (sort keys %func_hash)
    {
        my $func_info = $func_hash{$func};
        print OUT "$func $$func_info[0] $$func_info[1]\n";
    }

    print OUT "\n\n\n";

    # Generate object symbol table (object name, starting address, ending drress (included))
    print OUT "obj_table $obj_cnt:\n";
    foreach $obj (sort keys %obj_hash)
    {
        my $obj_info = $obj_hash{$obj};
        print OUT "$obj $$obj_info[0] $$obj_info[1]\n";
    }

    print OUT "\n\n\n";
    print OUT "pc_line_matching_table $matchingCount:\n";
    foreach (@matchingArr)
    {
        my $matching_info = $_;
        print OUT "$$matching_info{filename} $$matching_info{line} $$matching_info{start} $$matching_info{end}\n";
    }

    print OUT "\n\n\n";
    close OUT;
}




sub functionNameFromPc
{
    my ($pc, $functionName, $lastPc) = @_;

    $$functionName = "Unknown";

    if(!defined($pc))
    {
        print "error\n";
    }

    foreach $func (keys %func_hash)
    {
        my $func_info = $func_hash{$func};
        my $start = $$func_info[0];
        my $end = $$func_info[1];
        if($start <= $pc and ($pc <= $end))
        {
            $$functionName = $func;
            $$lastPc = $end;
        }
    }
}

sub generatePcLineMatchingTable
{
    open IN, $dbg_info_file;
    if (!defined(IN))
    {
        die "file not exit $dbg_info_file\n"
    }

    while (<IN>)
    {
        if (/^(\S+)\s+(\d+)\s+((?:0x)?[a-f0-9]+)/)
        {
            # generate pc to line matching table
            #pmu_bigint.c                                 480             0x164e5
            #pmu_bigint.c                                 513             0x16390
            #<a>                                          <b>             <c>
            $fileName = $1;
            $pc = hex($3); # instruction address
            $line = $2;

            # Step 1: Make sure that we have no duplicate File:Line -> PC
            #         There are cases where a single line number reference multiple PCs
            #         In these cases we keep the entry that has the lowsest line number
            #

            #         So basically we go through the file and read the stuff out first
            $hash = $fileName;
            if(exists $matchingTable{$hash})
            {
                if(exists $matchingTable{$hash}{$line})
                {
                    # the file:line combo has already been maped to something
                    $storedPc = $matchingTable{$hash}{$line};
                    if($storedPc > $pc)
                    {
                        $matchingTable{$hash}{$line} = $pc;
                    }
                } else
                {
                    # we can add the PC into the array
                    $matchingTable{$hash}{$line} = $pc;
                }
            } else
            {
                # we need to add a new hash into the main hash (matchingTable)
                $matchingTable{$hash}{$line} = $pc;
            }
        }
    }

    close IN;

    # Step 2: Go through the matching table and find adjacent lines, fill in the PC gap
    #         Fill in the PC between two lines if only the current is in the same function 
    #         as in the next line,  Fill in the PC up to end of the current function if the 
    #         next line is not the same function

    foreach $fileName (keys %matchingTable)
    {
        $lines = $matchingTable{$fileName};
        my @keys = sort {$a <=> $b} keys %$lines;
        $keySize = @keys;

        # resolve filename to fullpath
        $dir = ();
        @m_foundFiles = ();
        $m_foundFile = "";

        # find matching
        &find_source($tree, \@searchDir, $fileName, \%dir);

        $aSize = @m_foundFiles;
        # TODO: clean this up. Everything from this point for file searching is temporary
        if($aSize eq 0)
        {
            print "File $fileName not found\n";
            $m_foundFile = $fileName;
        } 
        elsif($aSize gt 1)
        {
            foreach(@m_foundFiles)
            {
                my $str = $_;
                my $subs = $family;
                $a = index($str, $subs);
                $subs = $chip;
                $b = index($_, $subs);
                if(($a != -1) or ($b != -1))
                {
                     $m_foundFile = $_;
                }
            }
        } 
        else  # only 1 file
        {
            $m_foundFile = $m_foundFiles[0];
        }

        for($i = 0; $i < $keySize; $i++)
        {
            my $thisLine = $keys[$i];
            my $thisPc = $$lines{$thisLine};
            my $nextLine = 0;
            my $nextPc = 0;

            if($i != $keySize-1)
            {
                $nextLine = $keys[$i+1];
                $nextPc = $$lines{$nextLine};
            } else
            {
                $nextLine = $thisLine;
                $nextPc = $thisPc;
            }

            $matchingHash{$matchingCount}{start} = $thisPc;
            $matchingHash{$matchingCount}{filename} = $m_foundFile;
            $matchingHash{$matchingCount}{line} = $thisLine;
            my $end;
            # are the two lines in the same function?
            functionNameFromPc($thisPc, \$thisName, \$lastPcOfThisFunction);
            functionNameFromPc($nextPc, \$nextName, \$lastPcOfNextFunction);

            if(($thisPc == 0) || ($nextPc == 0) || ($thisName eq "Unknown") || ($nextName eq "Unknown"))
            {
                print "$fileName:$thisLine($thisPc) -> $nextLine($nextPc) ... ";
                print " ($thisName ?= $nextName) ";
                print "Skipping due to PC 0 or Unknown function\n";
                next;
            }

            if(($thisName eq $nextName) && ($i != $keySize - 1))
            {
                # both lines are in the same function, we give all PCs within the range
                # between this line and the next to this line

                # we need to check if the PCs are the same for the 2 lines, in which case we do nothing
                if($thisPc == $nextPc)
                {
                    $end = $nextPc;
                } else
                {
                    $end = $nextPc-1;
                }
            } else
            {
                # this line and the next are different, we assign the range
                # from this pc to the end of function to this line
                $end = $lastPcOfThisFunction;
            }

            push(@matchingArr, {start => $thisPc, filename => $m_foundFile, line => $thisLine, end => $end});

            $matchingCount++;
        }
    }

    close IN;
}



sub find_source
{
    my ($sw, $searchPaths, $wantedFile, $dirOut) = @_;

    $m_wantedFile = $wantedFile;
    @m_foundFiles = ();

    foreach (@searchDir)
    {
        @t = ($sw, $_);
        my $full_path = File::Spec->catdir(@t);

        find(\&wanted, $full_path);
    }
}

sub wanted
{
    if($_ eq $m_wantedFile)
    {
        $fullPath = File::Spec->catfile($File::Find::dir, $_);
        #print "Found file at $fullPath\n";
        unshift(@m_foundFiles, $fullPath);
    }
}

sub load_symb
{
    my ($file, $func_hash, $func_cnt, $obj_hash, $obj_cnt) = @_;
    my $func_info;
    my $obj_info;
    open IN, $file;

    if (!defined(IN))
    {
        die "file not exits\n";
    }

    foreach (<IN>)
    {
        if (/\s+\d+: ([a-f0-9]+)\s+(\d+) (\w+)\s+\w+\s+\w+\s+\S+ _(\S+)/)
        {
            if ($3 eq "FUNC")  # type: FUNC
            {
                if (exists $$func_hash{$4})
                {
                    next;
                }
                $func_info = [];
                push @$func_info, hex($1); # starting addr
                push @$func_info, hex($1) + int($2) - 1; # end address, included

                $$func_hash{$4} = $func_info; # hash{func_name}
                $$func_cnt++;
            }
            elsif($3 eq "OBJECT")
            {
                if (exists $$obj_hash{$4})
                {
                    next;
                }
                $obj_info = [];
                push @$obj_info, hex($1); # starting addr
                push @$obj_info, hex($1) + int($2) - 1; # end address, included
                $$obj_hash{$4} = $obj_info; # hash{obj_name}
                $$obj_cnt++;
            }
            elsif($3 eq "NOTYPE")
            {
                if (exists $$obj_hash{$4})
                {
                    next;
                }
                $obj_info = [];
                push @$obj_info, hex($1); # starting addr
                push @$obj_info, hex($1); # size as 1
                $$obj_hash{$4} = $obj_info; # hash{obj_name}
                $$obj_cnt++;
            }
        }
    }
    close IN;

}
