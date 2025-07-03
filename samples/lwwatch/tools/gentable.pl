# Usage:
#
# To generate msenc method tables and priv register tables for lwwatch extension
# Eg:  perl gentable.pl -hwfile=pascal/gp100/dev_lwenc_pri_sw.h -clfile=clc1b7.h -type=msenc -version=v06_00 -outputFile=msenc0600.h
#
# To generate lwdec method tables and priv register tables for lwwatch extension
# Eg:  perl gentable.pl -hwfile=pascal/gp100/dev_lwdec_pri.h -clfile=clc1b0.h -type=lwdec -version=v03_00 -outputFile=lwdec0300.h
# 

#!/usr/bin/elw perl

use Class::Struct;
use List::Util qw(max);
use Getopt::Long;

#
# Command line arguments
#

my $file, $clFile, $type, $version, $outputFile;

GetOptions ('hwfile=s' => \$file, 'clfile=s' => \$clFile, 'type=s' => \$type, 'version=s' => \$version, 'outputFile=s' => \$outputFile);

$file = "../../../drivers/common/inc/hwref/$file";

$clFile = "../../../sdk/lwpu/inc/class/$clFile";

$outputFile = "../inc/$outputFile";

open(my $fh, '>', $outputFile) or die "Could not open file '$outputFile' $!";

print $fh "/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */\n\n";

open(my $data, '<', $clFile) or die "Could not open '$clFile' $!\n";

#
# Parse each line of class file
#

print $fh "dbg_", $type, "_v01_01 ", $type, "MethodTable_", $version, "[] =\n{\n";

my $old = "_NOP";
my $k = 0;
my $num = 0;

while (my $line = <$data>)
{
    #
    # Start from LW****_NOP to LW****_PM_TRIGGER_END
    #
    if (($line !~ "_NOP") and ($k == 0))
    {
        next;
    }
    $k = 1;

    if ($line =~ "TRIGGER_END")
    {
        $k = 0;
    }

    my @wholeline = split("#define ", $line);

    my @regline = split("[ ]+", $wholeline[1]);

    if ($regline[0] =~ ":")
    {
        next;
    }

    # If methods are of the form LW**_**_**(b)
    if ($regline[0] =~ "b")
    {
        @regline2 = split(/\(/, $regline[0]);
        my $count = $regline[3];
        $count = substr($count, 4, 10);
        $num = eval $count;
        if ($regline2[0] =~ $old)
        {
            next;
        }
    }
    else
    {
        $num = 1;
        if (($regline[0] =~ $old) and (length($regline[0]) > 10))
        {
            next;
        }
    }

    if ($num > 1)
    {
        for (my $n=0; $n < $num; $n++)
        {
            print $fh "    privInfo_", $type, "_v01_01(", $regline2[0], "(", $n, ")),\n";
        }
        $old = $regline2[0];
    }
    else
    {
        print $fh "    privInfo_", $type, "_v01_01(", $regline[0], "),\n";
        $old = $regline[0];
    }
}

print $fh "    privInfo_", $type, "_v01_01(0)\n};\n\n";

close ($clFile);


#
# Parse each line of manual file
#

open(my $data, '<', $file) or die "Could not open '$file' $!\n";

print $fh "dbg_", $type, "_v01_01 ", $type, "PrivReg_", $version, "[] =\n{\n";

while (my $line = <$data>)
{
    #
    # Registers in manuals are defined with these comments
    #
    if (($line !~ "4R") and ($line !~ "4A"))
    {
        next;
    }

    #
    # Get function name and its depth
    #

    my @wholeline = split("#define ", $line);

    my @regline = split("[ ]+", $wholeline[1]);

    print $fh "    privInfo_", $type, "_v01_01(";

    my @dev;

    #
    # For registers with same name indexed with i
    # For msenc, we need to consider dev as well
    #
    if ($regline[0] =~ "i")
    {
        my @reg = split(/\(/, $regline[0]);
        $line = <$data>;
        if ($regline[0] =~ "dev")
        {
            if ($dev[2] == 0)
            {
                @dev = split("[ ]+", $line);
            }
            $line = <$data>;
        }

        my @size = split("[ ]+", $line);

        for (my $i=0; $i < $size[2] ; $i++)
        {
            if ($regline[0] =~ "dev")
            {
                @reg = split("dev", $reg[0]);
                print $fh $reg[0], "(0,", "$i", ")),\n";
            }
            else
            {
                if ($regline[0] =~ "j")
                {
                    print $fh $reg[0], "(", "$i", ",0)),\n";
                }
                else
                {
                    print $fh $reg[0], "(", "$i", ")),\n";
                }
            }
            if ($i < ($size[2]-1))
            {
                print $fh "    privInfo_", $type, "_v01_01(";
            }
        }
    }
    else
    {
        if ($regline[0] =~ "dev")
        {
            @regline = split("dev", $regline[0]);
            print $fh $regline[0], "0)),\n";
        }
        else
        {
            print $fh $regline[0], "),\n";
        }
    }
}

print $fh "    privInfo_", "$type", "_v01_01(0)\n};\n\n";

if ($type eq "msenc")
{
    print $fh "copy similar priv reg table upto $dev devices\n";
}

close ($fh);
close ($file);

print "--done--\n";
