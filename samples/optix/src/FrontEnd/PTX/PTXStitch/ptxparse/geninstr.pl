#!/bin/perl
#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright (c) 2018-2021, LWPU CORPORATION.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.
#
# LWIDIA_COPYRIGHT_END

use strict;
use Getopt::Long;

my $printEnums = '';
my $printInits = '';
my $printExtIncl = '';
my $printExtDesc = '';

use constant {
    STANDARD => 0,
    EXTENDED => 1
};

GetOptions('enums' => \$printEnums,
           'inits' => \$printInits,
           'ext-incl' => \$printExtIncl,
           'ext-desc' => \$printExtDesc);

($printEnums
 + $printInits
 + $printExtIncl
 + $printExtDesc) == 1 or die "exactly one output mode expected";

# Assumption: $selectedSet does not matter if $printEnums is set.
my $selectedSet = ($printInits ? STANDARD : EXTENDED);

# A set of all instruction names implemented as a hash. The key is the
# escaped name, and the value is its sequence number at time of
# insertion into the hash.
my %instructions = ();

# ptx_unknown_Instr is always the first enum
# "scalar keys": Perl idiom to retrieve the number of keys in the hash.
# It is assumed to be zero at this point, but we choose not to
# hard-code the zero. An assert in ptxas ensures that
# ptx_unknown_Instr is always assigned zero.
$instructions{"unknown"} = scalar keys %instructions;

# A single line in the table can describe multiple instructions, and
# the same instruction can appear in multiple lines. The count below
# keeps track of all combinations resulting from this cross product.
my $templateCount = 0;

my $stdCount = 0;
my $extCount = 0;

# For STANDARD instructions:
# $initBuffer: contains a list of calls to the template initializer,
#              one for each instruction.
#
# For EXTENDED instructions:
# $featureBuffer: defines one instance of ptxInstructionFeature for
#                 each instruction.
# $initBuffer   : defines an array containing the above instances.
my $initBuffer;
my $featureBuffer;

# String that describes extended instructions.
my $descBuffer;

# String to store token for extended descriptor
my $token = "";

# Arrays for extended, standard and all instructions.
my @extInstructions;
my @stdInstructions;
my @allInstructions;

LINE: while (<STDIN>) {

    if (m/^TOKEN (.*)$/) {
        !$token or die "multiple TOKEN lines encountered";
        $token = $1;
        next LINE;
    }

    # eliminate whitespace
    s/\s+//g;

    # skip uninteresting lines
    next LINE unless /^DEFINE\(/;

    # separate extended and standard instructions in arrays.
    if (/^.*EXTENDED.*$/) {
        push (@extInstructions, $_);
    } elsif (/^.*STANDARD.*$/) {
        push (@stdInstructions, $_);
    } else {
        die "unexpected flag: $_";
    }
}

# Combine extended and standard instructions array to create an array of all instructions
@allInstructions = (@extInstructions, @stdInstructions);

if ($printExtIncl) {
    $initBuffer .= "ptxInstructionFeature extFeatures[PTX_NUM_EXTENDED_TEMPLATES] = {\n";
}

# Process all instructions
foreach (@allInstructions) {
    # remove DEFINE(...)
    s/^DEFINE\((.*)\)$/$1/g;

    # split the main record
    my ($instrTypes, $instrList, $argTypes, $featureList, $flagList) = split /,/;

    if (!$printInits) {
        # We don't process the instruction types and argument types,
        # but if they are missing, we need a placeholder in the
        # descriptor string.
        if (!$instrTypes) { $instrTypes = '.'; }
        if (!$argTypes) { $argTypes = '.'; }
    }

    # get the list of instructions
    my @instrs = split /\|/, $instrList;
    @instrs != 0 or die "expected at least one instruction per record";

    # parse flags
    my $stdInstr = 0;
    my $extInstr = 0;
    my @flags = split /\|/, $flagList;
    FLAG: foreach (@flags) {
        next FLAG unless $_;
        SWITCH: for ($_) {
            if (/^STANDARD$/) { $stdInstr = 1; last SWITCH; }
            if (/^EXTENDED$/) { $extInstr = 1; last SWITCH; }
            die "unexpected flag: $_";
        }
    }

    $stdInstr + $extInstr != 0 or die "at least one flag required";

    # Process each instruction oclwring on current line as a separate template
    INSTR: foreach my $name (@instrs)
    {
        $name or die "every instruction needs a name";

        # create opcode by replacing dot with underscore
        (my $opcode = $name) =~ s/\./_/g;

        ++$templateCount;
        ++$extCount if $extInstr;
        ++$stdCount if $stdInstr;

        if (!exists($instructions{$opcode})) {
            $instructions{$opcode} = scalar keys %instructions;
        }

        next INSTR if $printEnums;

        next INSTR if ($selectedSet == EXTENDED && !$extInstr)
                   || ($selectedSet == STANDARD && !$stdInstr);

        my $featuresName = "${opcode}_${templateCount}_Features";

        if ($printExtIncl) {
            # Emit declaration for the template-specific features
            $featureBuffer .= "\n";
            $featureBuffer .= "ptxInstructionFeature ${featuresName} = { 0, };\n";
        } elsif ($printInits) {
            # initialize features
            $initBuffer .= "\n";
            $initBuffer .= "stdMEMCLEAR(&features);\n";
        }

        # Emit code to set the features declared for the current template
        my @features = split /\|/, $featureList;
        FEATURE: foreach (@features) {
            next FEATURE unless $_;
            if ($printExtIncl) {
                $featureBuffer .= "${featuresName}.${_} = 1;\n";
            } elsif ($printInits) {
                $initBuffer .= "features.${_} = 1;\n";
            }
        }

        if ($printInits) {
            $initBuffer .= "addInstructionTemplate(parseData, \"$instrTypes\",\"$name\",\"$argTypes\",features,ptx_${opcode}_Instr, False);\n";
        } else {
            if ($printExtIncl) {
                $initBuffer .= "    ${featuresName},\n";
            }
            $descBuffer .= " ${instrTypes}";
            $descBuffer .= " ${name}";
            $descBuffer .= " ${argTypes}";
            $descBuffer .= " ${instructions{${opcode}}}";
        }
    }
}

$descBuffer = $token . $descBuffer;

if ($printEnums) {
    print "#ifndef ptxInstructions_INCLUDED\n";
    print "#define ptxInstructions_INCLUDED\n";
    print "\n";
    print "typedef enum {\n";
    foreach (sort(keys %instructions)) {
        print "    ptx_${_}_Instr = ${instructions{$_}},\n";
    };
    print "} ptxInstructionCode;\n";
    print "\n";
    print "#define PTX_NUM_TEMPLATES ${templateCount}\n";
    print "#define PTX_NUM_STANDARD_TEMPLATES ${stdCount}\n";
    print "#define PTX_NUM_EXTENDED_TEMPLATES ${extCount}\n";
    print "\n";
    print "#endif\n";
} elsif ($printExtDesc) {
    print $descBuffer;
} else {
    if ($printExtIncl) {
        $featureBuffer .= "\n";
        $initBuffer .= "};\n";
    }

    print $featureBuffer;
    print $initBuffer;

    if ($printExtIncl) {
        print "\n";
        print "#define PTX_EXTENSION_TOKEN ${token}\n";
        print "#define PTX_EXT_DESC_STRING_LEN " . length($descBuffer) . "\n";
#if LWCFG(NOT_FOR_NDA_PARTNER_SOURCE_SHARING)
        print "#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)\n";
#endif
#if LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
        print "#if !defined(GPGPUCOMP_DRV_BUILD)\n";
        print "extDescBuffer = stdCOPYSTRING(\"${descBuffer}\");\n";
        print "stdASSERT(strlen(extDescBuffer) == PTX_EXT_DESC_STRING_LEN,"
            . " (\"bad descriptor buffer\"));\n";
        print "#endif\n";
#endif
#if LWCFG(NOT_FOR_NDA_PARTNER_SOURCE_SHARING)
        print "#endif\n";
#endif
    }
}
