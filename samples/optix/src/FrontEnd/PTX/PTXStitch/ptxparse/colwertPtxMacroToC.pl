#!/bin/perl
#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright (c) 2010-2021, LWPU CORPORATION.  All rights reserved.
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
use FindBin;
use lib "$FindBin::Bin/perlModules";
use Digest::Adler32 qw(:ALL);

my (%CmdLineOption, %unmangle, %multiop, %funcmap, @macros, %functions, @macrofunctions, %adlerHashForNames);
my ($ptxfile, $cfile, $newptxfile, $fmtfile, $irmap, $profile, $ptxfmt);
my ($tabsize, $margin, $content, $macrofuncs, $braces, $lineno, $offset);
my ($ismacrofunc, $insidemacrofunc);

init_elw();

$macrofuncs = parse_input_file();
$content    = get_functions_prototype();
$content   .= get_ptx_functions_code();
$content   .= get_macroname_maps();
$content   .= $macrofuncs;

write_c_file();

############################  END OF MAIN PROGRAM  ############################
###############################################################################

sub parse_input_file {
    my ($buffsize, $buffer, $ptxcode, $inmacro, $last_func_line, $version_target, $header, $mname, %macroargs, %macrofuncargs) = (50000);
    my ($parsing_inline_prototype, $proto_aclwmulator) = (0, "");

    open (IRDEFSMAP, "$irmap") || die "Cannot open $irmap.$!\n";
    foreach my $mapline (<IRDEFSMAP>) { $_ .= "$mapline "; }
    close(IRDEFSMAP);
    s/^\s*\(//; s/\)\s*$//; s/\),\s*\(/:/g; s/\)\s*\(/:/g; s/\"//g;
    foreach my $fmap (split (/:/, $_)) {
        $fmap =~ m/(\w+) (\w+) ((\w|\W)*)/;
        $funcmap{"$1:$3"} = $2;
    }

    open (PTXMACRO, "$ptxfile") || die "Cannot open $ptxfile.$!\n";
    while (<PTXMACRO>) {
        $lineno++;
        if (!$inmacro) {
            #         .MACRO   div <...>            macro name followed by space and arguments
            #         .MACRO   membar;              macro name followed by semi-colon and no arguments
            if (/^\s*\.MACRO\s+(\S+?)(\s+|;)/) {
                $ptxcode .= "$_";
                my ($argindex, $arglist) = (0, $_);
                %macroargs = ();
                $mname = mangle_macro_name($1);
                $arglist =~ s/^\s*\.MACRO\s+(\S+)\s*//;
                $arglist =~ s/\s*;\s*$//;
                foreach my $arg (split(/, /, $arglist)) {
                    $macroargs{$arg} = "Arg$argindex";
                    $argindex++;
                }
                $buffer .= sprintf("%s%s\n", ' 'x $margin, "String $mname (ptxParsingState parseState, char * macroBody)");
                push @macros, ($mname);
                $ismacrofunc = 0;
                $inmacro=1;
                $header =1;
                $braces =0;
            } elsif (/^\s*.FORCE_INLINE\s+/ || $parsing_inline_prototype) {
                if ($parsing_inline_prototype) {
                    $_ = "$proto_aclwmulator $_";
                    s/\s+/ /g;
                }
                if (m/.FORCE_INLINE .func\s+(\(((\w|\W)*)\))\s+((\%|\$|\w)+)\s+(\((\w|\W)*\))/) {
                    # Entire prototype of the .FORCE_INLINE is in the same line
                    $parsing_inline_prototype = 0;
                    $proto_aclwmulator = "";
                    $ptxcode .= "$_";
                    %macrofuncargs = ();
                    $mname = mangle_macro_name($4);
                    populate_adler32_hash($4);
                    my $iargs = $6;
                    my $oargs = $2;
                    $iargs =~ s/^\(\s*//; $iargs =~ s/\s*\)$//;
                    $oargs =~ s/^\(\s*//; $oargs =~ s/\s*\)$//;
                    my ($iargindex, $oargindex) = (0, 0);
                    my @iarglist = split(/, /, $iargs);
                    my @oarglist = split(/, /, $oargs);
                    foreach my $arg (@iarglist) {
                        if ($arg =~ /\s*.reg\s*\.(\S*)\s*(\S+)\s*/) {
                            my %argInfo;
                            $macrofuncargs{"I"}{$2}{"INDEX"}          = $iargindex;
                            $macrofuncargs{"I"}{$2}{"EMIT_STR"}       = "IArg$iargindex";
                            $macrofuncargs{"I"}{$2}{"TYPE"}           = "$1";
                            $macrofuncargs{"I"}{$2}{"SHADOW_VAR"}     = "__".$mname."_$2";
                            $iargindex++;
                        }
                    }
                    foreach my $arg (@oarglist) {
                        if ($arg =~ /\s*.reg\s*\.(\S*)\s*(\S+)\s*/) {
                            $macrofuncargs{"O"}{$2}{"INDEX"}          = $oargindex;
                            $macrofuncargs{"O"}{$2}{"EMIT_STR"}       = "OArg$oargindex";
                            $macrofuncargs{"O"}{$2}{"TYPE"}           = "$1";
                            $macrofuncargs{"O"}{$2}{"SHADOW_VAR"}     = "__".$mname."_$2";
                            $oargindex++;
                        }
                    }

                    $buffer .= sprintf("%s%s\n", ' 'x $margin, "String $mname (ptxParsingState parseState, char * macroBody)");
                    push @macrofunctions, ($mname);
                    $ismacrofunc = 1;
                    $inmacro=1;
                    $header =1;
                    $braces =0;
                } else {
                    # Partial prototype of the .FORCE_INLINE is seen in this line
                    $parsing_inline_prototype = 1;
                    chomp($_);
                    $proto_aclwmulator = "$_";
                }
            } elsif (m/\s*.DECL_FUNC\s+(.*)/) {
                $ptxcode .= $1;
            } elsif (/^\s*\.version\s*/ || /^\s*\.target\s*/) {
                $version_target .= $_;
                $ptxcode .= $_;
            } elsif (/\s*\.func\s*/) {
                # Start a new function definition. We could assert
                # that the variable is empty.
                $last_func_line = $_;
            } elsif ($last_func_line) {
                # Append if we are already consuming a function.
                $last_func_line .= $_;
            } else {
                # Not interesting to us.
                $ptxcode .= $_;
            }

            # Having consumed a line, check to see if we have a full
            # function definition. Parse it if we do.
            if ($last_func_line =~ m/\{/) {
                my $braces = $last_func_line =~ s/\{/\{/g;
                $braces -= $last_func_line =~ s/\}/\}/g;

                if ($braces == 0) {
                    parse_function_body($last_func_line, \$ptxcode, $version_target);
                    $last_func_line = "";
                }
            }
        }
        while($inmacro) {
            $_ = <PTXMACRO>; $lineno++;
            if (/^\s*\.ENDMACRO/) {
                $ptxcode .= "$_\n";
                $buffer .= sprintf("%s%s\n", ' 'x $margin, "// End of $mname\n");
                $inmacro = 0;
            } elsif (/^\s*\.IF/) {
                $buffer .= sprintf("%s%s\n", ' 'x $margin, expand_if($_, \%macroargs, \%macrofuncargs));
                $margin += $tabsize;
            } elsif (/^\s*\.ELIF/) {
                $margin -= $tabsize;
                $buffer .= sprintf("%s%s%s\n", ' 'x $margin, "} else ", expand_if($_, \%macroargs, \%macrofuncargs));
                $margin += $tabsize;
            } elsif (/^\s*\.ELSE/) {
                $margin -= $tabsize;
                $buffer .= sprintf("%s%s\n", ' 'x $margin, "} else {");
                $margin += $tabsize;
            } elsif (/^\s*\.ENDIF/) {
                $margin -= $tabsize;
                $buffer .= sprintf("%s%s\n", ' 'x $margin, "}");
            } elsif (/^\s*\{\s*$/) {
                if ($header) {
                    $buffer .= sprintf("%s%s\n",   ' 'x $margin, "{");
                    $margin += $tabsize;
                    $buffer .= sprintf("%s%s\n",   ' 'x $margin, "char *macroExpansion, *buffer = stdMALLOC($buffsize);");
                    $buffer .= sprintf("%s%s\n\n", ' 'x $margin, "int len = 0;");
                    $header  = 0;
                }
                if ($ismacrofunc) {
                    my $prologue = (' ' x $margin)."{\n";
                    $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify($prologue));
                    $margin += $tabsize;
                    $braces++;
                    $prologue    = (' ' x $margin)."// Defining the shadow variables\n";
                    $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify($prologue));

                    foreach my $inarg (keys %{$macrofuncargs{"I"}}) {
                        $prologue = (' ' x $margin).".reg .".$macrofuncargs{"I"}{$inarg}{"TYPE"}." ".$macrofuncargs{"I"}{$inarg}{"SHADOW_VAR"}.";\n";
                        $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify($prologue));
                    }

                    foreach my $outarg (keys %{$macrofuncargs{"O"}}) {
                        $prologue = (' ' x $margin).".reg .".$macrofuncargs{"O"}{$outarg}{"TYPE"}." ".$macrofuncargs{"O"}{$outarg}{"SHADOW_VAR"}.";\n";
                        $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify($prologue));
                    }

                    # branch out of the expansion if predicate is set
                    $buffer .= sprintf("\n%sif (get_PRED(parseState->parseData)) {", (' ' x $margin));
                    $prologue= sprintf("\n%s %{PRED_NEG} bra __skip_$unmangle{$mname};\n", (' ' x $margin));
                    $buffer .= sprintf("\n%s%s", ' 'x ($margin+$tabsize), expand_generic($prologue, \%macroargs, \%macrofuncargs));
                    $buffer .= sprintf("\n%s}\n\n", (' ' x $margin));

                    $prologue = "\n";
                    $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify($prologue));

                    $prologue = (' ' x $margin)."// Copying input arguments into shadow variables\n";
                    $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify($prologue));

                    foreach my $inarg (keys %{$macrofuncargs{"I"}}) {
                        $buffer .= sprintf("%sif (getExpressionKindForArg(parseState->parseData, %d,0) != ptxSinkExpression) {\n", ' ' x $margin, $macrofuncargs{"I"}{$inarg}{"INDEX"});
                        #                    mov.<type> <shadow-var>,  <inArg>
                        $prologue = sprintf("%smov.%s      %s ,        %{%s};\n", ' ' x $margin, $macrofuncargs{"I"}{$inarg}{"TYPE"}, $macrofuncargs{"I"}{$inarg}{"SHADOW_VAR"}, $inarg);
                        $buffer .= sprintf("%s%s\n", ' 'x ($margin+$tabsize), expand_generic($prologue, \%macroargs, \%macrofuncargs));
                        $buffer .= sprintf("%s}\n", ' ' x $margin);
                    }
                    $insidemacrofunc = 1;
                }
                $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify("{\n"));
                $margin += $tabsize;
                $braces++;
            } elsif (/^\s*\}\s*/) {
                $margin -= $tabsize;
                $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify($_));
                $braces--;
                if ($ismacrofunc && $braces == 1) {
                    $insidemacrofunc = 0;
                    my $epilogue = "\n";
                    $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify($epilogue));

                    $epilogue = (' ' x $margin)."// Copying shadow variables into return arguments\n";
                    $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify($epilogue));

                    foreach my $outarg (keys %{$macrofuncargs{"O"}}) {
                        $buffer .= sprintf("%sif (getExpressionKindForArg(parseState->parseData, %d,1) != ptxSinkExpression) {\n", ' ' x $margin, $macrofuncargs{"O"}{$outarg}{"INDEX"});
                        #                      mov.<type>  <outArg>, <shadow-var>
                        $epilogue = sprintf("%smov.%s      %{%s},     %s;\n", ' ' x $margin, $macrofuncargs{"O"}{$outarg}{"TYPE"}, $outarg, $macrofuncargs{"O"}{$outarg}{"SHADOW_VAR"});
                        $buffer .= sprintf("%s%s\n", ' 'x ($margin+$tabsize), expand_generic($epilogue, \%macroargs, \%macrofuncargs));
                        $buffer .= sprintf("%s}\n", ' ' x $margin);
                    }
                    # label to be branched to if the expansion is to be skipped
                    $buffer .= sprintf("\n%sif (get_PRED(parseState->parseData)) {", (' ' x $margin));
                    $epilogue= sprintf("%s__skip_$unmangle{$mname}:\n", ' ' x $margin);
                    $buffer .= sprintf("\n%s%s", ' 'x ($margin+$tabsize), expand_generic($epilogue, \%macroargs, \%macrofuncargs));
                    $buffer .= sprintf("\n%s}\n\n", (' ' x $margin));

                    $margin -= $tabsize;
                    $braces--;
                    $epilogue = (' ' x $margin)."}\n";
                    $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify($epilogue));

                    $inmacro = 0;
                    $ismacrofunc = 0;
                }
                if ($braces==0) {
                    $buffer .= sprintf("\n%smacroExpansion = stdCOPYSTRING(buffer);\n", ' ' x $margin);
                    $buffer .= sprintf("\n%sstdFREE(buffer);\n", ' ' x $margin);
                    $buffer .= sprintf("\n%sreturn macroExpansion;\n", ' ' x $margin);
                    $margin -= $tabsize;
                    $buffer .= sprintf("%s%s\n", ' 'x $margin, "}");
                }
            } elsif (/^\s*\.INLINE.*\s+call(\.uni)?\s+/) {
                my ($funcname, $inlined_body) = get_inlined_body_for_callee($_);

                open(NEWGROUP, "> $funcname.inlined.ptx") || die "Unable to dump inlined body for $funcname.$!\n";
                print NEWGROUP $inlined_body;
                close NEWGROUP;

                # FIXME: This code is mostly identical to the code
                # above that handles other statements.
                for (split(/^/, $inlined_body)) {
                    if (/^\s*\.IF/) {
                        $buffer .= sprintf("%s%s\n", ' 'x $margin, expand_if($_, \%macroargs, \%macrofuncargs));
                        $margin += $tabsize;
                    } elsif (/^\s*\.ELIF/) {
                        $margin -= $tabsize;
                        $buffer .= sprintf("%s%s%s\n", ' 'x $margin, "} else ", expand_if($_, \%macroargs, \%macrofuncargs));
                        $margin += $tabsize;
                    } elsif (/^\s*\.ELSE/) {
                        $margin -= $tabsize;
                        $buffer .= sprintf("%s%s\n", ' 'x $margin, "} else {");
                        $margin += $tabsize;
                    } elsif (/^\s*\.ENDIF/) {
                        $margin -= $tabsize;
                        $buffer .= sprintf("%s%s\n", ' 'x $margin, "}");
                    } elsif (/^\s*\{\s*$/) {
                        $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify($_));
                        $margin += $tabsize;
                        $braces++;
                    } elsif (/^\s*\}\s*/) {
                        $margin -= $tabsize;
                        $buffer .= sprintf("%slen += sprintf(buffer + len, %s, %s);\n", ' ' x $margin, '"%s"', unstringify($_));
                        $braces--;
                    } else {
                        if ($_) {
                            $buffer .= sprintf("%s%s\n", ' 'x $margin, expand_generic($_, \%macroargs, \%macrofuncargs));
                        }
                    }
                }
            } else {
                if ($_) {
                    $buffer .= sprintf("%s%s\n", ' 'x $margin, expand_generic($_, \%macroargs, \%macrofuncargs));
                }
            }
        }
    }
    close (PTXMACRO);

    open (NEWPTX, "> $newptxfile") || die "Cannot open $newptxfile.$!\n";
    print NEWPTX $ptxcode;
    close (NEWPTX);

    open (FMT, "> $fmtfile") || die "Cannot open $fmtfile.$!\n";
    binmode(FMT);
    print FMT $ptxfmt;
    close (FMT);

    # Print adler hash values so that it will help in debugging.
    write_adler_hash_file();

    return $buffer;
}

sub parse_function_body {
    my $buffer = shift;
    my $ptxcode = shift;
    my $version_target = shift;
    $buffer =~ m/((\w|\W)*?)(\{(\w|\W)*)/;
    my ($prototype, $funcbody, $funcname) = ($1, $3);
    $$ptxcode .= $prototype;
    $$ptxcode .= "\n" if ($prototype !~ /\n\s*$/);
    $prototype =~ m/.func\s+((\((\w|\W)*\))*)\s+((\%|\$|\w)+)\s+((\((\w|\W)*\))*)/;
    $funcname  = $4;
    $functions{$funcname} = analyze_function($funcbody, $prototype);
    open (FUNC, "> ${funcname}${profile}.bin2c") || die "Cannot open ${funcname}${profile}.bin2c.$!\n";
    print FUNC "${version_target}${prototype}${funcbody}";
    close (FUNC);
}

# Creates an inlined instruction group from a function body.
#
# Input is a single-line string containing a call instruction.
# Returns a multi-line string containing the body of the called
# function with changes that correspond to the .INLINE macro.
sub get_inlined_body_for_callee {
    my $line = shift;

    # Parse the call instruction
    #                               $ifclause   %{PRED}        call            (dst,...),   funcname,     (src,...); 
    $line =~ m/\s*\.INLINE(\_WHEN\s+)?(.*?)(\%\{PRED\})?\s+(call(\.uni)?)\s+\((.*?)\)\s*,\s*(.*?)\s*,\s*\((.*?)\)\s*;.*/;
    my ($ifkeyword, $ifclause, $pred, $call, $output_string, $funcname, $input_string) = ($1, $2, $3, $4, $6, $7, $8);
    $ifclause =~ s/\((.*)\)/\1/;

    my $func = $functions{$funcname};
    my @actual_outputs = split(/\s*,\s*/, $output_string);
    my @actual_inputs = split(/\s*,\s*/, $input_string);

    my $inlined_body;

    # Start the instruction group
    $inlined_body .= "{\n";

    # Declare formal inputs with mangled names.
    for my $tuple (@{$func->{INPUTS}}) {
        my ($ftype, $fname) = @{$tuple};
        $inlined_body .= ".reg $ftype $fname\_$funcname;\n";
    }

    $inlined_body .= "\n";
    # Declare formal outputs with mangled names.
    for my $tuple (@{$func->{OUTPUTS}}) {
        my ($ftype, $fname) = @{$tuple};
        $inlined_body .= ".reg $ftype $fname\_$funcname;\n";
    }

    # Assign actual inputs to formal inputs
    $inlined_body .= "\n";
    for my $index (0..@actual_inputs-1) {
        my $tuple = $func->{INPUTS}[$index];
        my ($ftype, $fname) = @{$tuple};
        my $actual = $actual_inputs[$index];
        # Mangle the formal argument.
        $inlined_body .= "mov$ftype $fname\_$funcname, $actual;\n";
    }

    # Print the body with suitable changes.
    for my $line (split(/\n/, $func->{BODY})) {
        # Remove return instructions.
        $line =~ s/^\s*ret\s*;\s*$//;

        # Mangle labels.
        $line =~ s/(\$[\w_]*)/$1\_$funcname/g;

        # Mangle names of formal arguments.
        for my $tuple (@{$func->{INPUTS}}) {
            my ($ftype, $fname) = @{$tuple};
            $line =~ s/([\s,;]*)$fname([\s,;]*)/$1$fname\_$funcname$2/g;
        }
        for my $tuple (@{$func->{OUTPUTS}}) {
            my ($ftype, $fname) = @{$tuple};
            $line =~ s/([\s,;]*)$fname([\s,;]*)/$1$fname\_$funcname$2/g;
        }

        $inlined_body .= $line . "\n";
    }

    # Assign formal outputs to actual outputs.
    $inlined_body .= "\n";
    for my $index (0..@actual_outputs-1) {
        my $tuple = $func->{OUTPUTS}[$index];
        my ($ftype, $fname) = @{$tuple};
        my $actual = $actual_outputs[$index];
        # Mangle the formal argument.
        $inlined_body .= "$pred mov$ftype $actual, $fname\_$funcname;\n";
    }

    $inlined_body .= "}\n";

    chomp $ifkeyword;
    if ($ifkeyword ne "") {
        $inlined_body = ".IF $ifclause\n$inlined_body";
        $inlined_body .= ".ELSE\n";
        $inlined_body .= "%{PRED} $call ($output_string), $funcname, ($input_string);\n";
        $inlined_body .= ".ENDIF\n";
    }

    return ($funcname, $inlined_body);
}

# Returns some information about the function as a hash with the
# following elements:
#
# BODY   : Multi-line string containing the function body
# INPUTS : List of tuples (type, name) for formal inputs
# OUTPUTS: List of tuples (type, name) for formal outputs
sub analyze_function {
    my ($funcbody, $prototype) = @_;

    my $func = {};
    $func->{BODY} = $funcbody;

    # Match the parentheses to locate arguments.
    $prototype =~ m/.*?\((.*?)\).*?\((.*?)\)/;
    my ($inputs, $outputs) = ($2, $1);

    # Comma-separated list of declarations with no terminator.
    # Example: ".reg .u32 arg0, .reg u32 arg1"
    for my $formal (split(/\s*,\s*/, $inputs)) {
        #               .reg    .u32      arg0
        $formal =~ m/\s*\.reg\s+(\..*?)\s+(.*)\s*/;
        my ($ftype, $fname) = ($1, $2);
        push @{$func->{INPUTS}}, [$ftype, $fname];
    }

    # Same for outputs.
    for my $formal (split(/\s*,\s*/, $outputs)) {
        $formal =~ m/\s*\.reg\s+(\..*?)\s+(.*)\s*/;
        my ($ftype, $fname) = ($1, $2);
        push @{$func->{OUTPUTS}}, [$ftype, $fname];
    }

    return $func;
}

sub expand_if {
    my ($macroargref, $macrofuncargref);
    ($_, $macroargref, $macrofuncargref) = @_;
    s/\s+$//;
    my ($eindex, $dollor, $thisline, $expression, %expr, $testexpr) = (0, '_DOLLOR', $_);
    s/^\s*\.(EL)?IF\s+//;
    s/\$/$dollor/g;
    while (/(\"\S*?\"\s+(==|!=|<=|<|>=|>|.in)\s+\"(\s|\S)*?\")/) {
        my $expr = $1;
        s/\Q$expr/EXPR$eindex/;
        $expr =~ s/$dollor/\$/g;
        $expr{"EXPR$eindex"} = $expr;
        $eindex++;
    }
    s/$dollor/\$/g;
    $expression = $testexpr = $_;
    $testexpr =~ s/(\s|\(|\)|\&\&|\|\||EXPR\d+)//g;
    die "Invalid PTX Macro Conditional Expression of the form '$expression'\n\tat ${ptxfile}, line no. $lineno : $thisline\n" if ($testexpr);
    
    foreach my $ekey (sort keys %expr) {
        my ($expr, $condition, $variable, $enumtype) = ($expr{$ekey});
        my ($lhs, $op, $rhs) = $expr =~ /\"(\S*)\"\s+(\S+)\s+\"((\s|\S)*)\"/;
        ($lhs, $rhs) = ($rhs, $lhs) if ($lhs !~ m/(\%|\$)\{(\S+)\}/);
        if ($lhs =~ m/(\%|\$)\{(\S+)\}/) {
            $enumtype = ( $1 eq "\$" ) ? "elw" : "instr";
            $variable = $2;
        }
        if ($variable) {
            if ($op eq ".in") {
                my @enums = split(/ /, $rhs);
                my $count = 0;
                $condition = "get_MacroElw(parseState->parseData, $variable) && (";
                foreach (@enums) {
                    $condition .= sprintf("%s%s", $count?" || ":"", "isMacroElwEqual(parseState->parseData, $variable, \"$_\")");
                    $count++;
                }
                $condition .= ")";
            } else {
                if ($rhs =~ /\./ && !$multiop{$op}) {
                    die "Invalid PTX Macro Relational Operator \"$op\" in expression \"$lhs\" $op \"$rhs\" at ${ptxfile}, line no. $lineno : $thisline\n";
                }                
                if ($lhs =~ /(TYPES)(\d?)/) {
                    my ($varname, $varindex) = ($1, $2);
                    $rhs =~ s/^\.//;
                    if ($lhs =~ /$varname\d/) {
                        $condition = sprintf("get_$varname(parseState->parseData, $varindex) $op  %s", $funcmap{"$varname:.$rhs"});
                    } else {
                        my $count = 0;
                        foreach (split(/\./, $rhs)) {
                            $_ = ".$_";
                            $condition .= sprintf("%sget_$varname(parseState->parseData, $count) $op %s", $count?" $multiop{$op} ":"", $funcmap{"$varname:$_"});
                            $count++;
                        }
                    }
                } elsif ($lhs =~ /VIDEOSELECTOR(0|1|2)(1|2|4)/) {
                    my ($varindex, $simdtype, $count, $vexpr) = ($1, $2, 0);
                    $rhs =~ s/^\.//;
                    my @rhs = split(/\./, $rhs);
                    foreach ((reverse(@rhs), ("", "", "", ""))) {
                        $_ = ".$_" if ($_);
                        $vexpr .= sprintf("%s(get_VIDEOSELECTOR(parseState->parseData, $varindex, $count, $simdtype) $op %s)", $count?" $multiop{$op} ":"", $funcmap{"VIDEOSELECTOR:$_"});
                        $count++;
                        last if ($count == $simdtype);
                    }
                    $condition = "( $vexpr )";
                } else {
                    if ($enumtype eq "elw") {
                        if ($rhs eq "true") {
                            $condition = sprintf ("get_MacroElw(parseState->parseData, $variable)");
                        } elsif ($rhs eq "false") {
                            $condition = sprintf ("!get_MacroElw(parseState->parseData, $variable)");
                        } else {
                            $condition = sprintf ("(get_MacroElw(parseState->parseData, $variable) $op $rhs)");
                        }
                    } else {
                        if ($funcmap{"$variable:$rhs"}) {
                            $condition = sprintf ("get_${variable}(parseState->parseData) $op %s", $funcmap{"$variable:$rhs"});
                        } elsif ($$macrofuncargref{"I"}{$variable} || $$macrofuncargref{"O"}{$variable}) {
                            my $argNum;
                            my $isRetArg = $$macrofuncargref{"I"}{$variable} ? 0 : 1;
                            if ($isRetArg) {
                                $argNum = $$macrofuncargref{"O"}{$variable}{"INDEX"};
                            } else {
                                $argNum = $$macrofuncargref{"I"}{$variable}{"INDEX"};
                            }
                            if (($rhs eq "_") && $multiop{$op}) {
                                $condition = "(getExpressionKindForArg(parseState->parseData, $argNum,$isRetArg) $op ptxSinkExpression)";
                            } elsif ($rhs =~ /\d+/) {
                                $condition = "(getExpressionKindForArg(parseState->parseData, $argNum, $isRetArg) == ptxIntConstantExpression && getConstantValueOfInputArg(parseState->parseData, $argNum) $op $rhs)";
                            } else {
                                die "Invalid PTX Macro Condition \"$lhs\" $op \"$rhs\" at ${ptxfile}, line no. $lineno : $thisline\n";
                            }
                        } else {
                            if ($rhs eq "") {
                                $condition = sprintf ("%sget_$variable(parseState->parseData)", $op eq "!="?"":"!");
                            } elsif ($rhs =~ /\d+/) {
                                $condition = sprintf ("get_$variable(parseState->parseData) $op $rhs");
                            } else {
                                # Can't be compared with non-null
                                die "Invalid PTX Macro Condition \"$lhs\" $op \"$rhs\" at ${ptxfile}, line no. $lineno : $thisline\n";
                            }
                        }
                    }
                }
            }
        } else {
            #Can LHS and RHS both be without any variable or empty???
            die "Invalid PTX Macro Condition at ${ptxfile}, line no. $lineno : $thisline\n";
        }
        $expression =~ s/$ekey/$condition/;
    }
    return "if ( $expression ) {";
}

sub expand_generic {
    my ($source, $macroargref, $macrofuncargref) = @_;
    my ($format, $unstr_format, $args) = ("", "", "");
    my @tokens = split(/\%\{(\w+)}/, $source);
    if (@tokens == 1) {
        $args   = unstringify($tokens[0]);
        $format = '"%s"';
        $unstr_format = $format;
    } else {
        my ($count, $argmargin) = (0, ' ' x ($margin + $tabsize));
        while (@tokens) {
            my ($inputfmt, $variable, $subst);
            $inputfmt = shift (@tokens);
            $inputfmt =~ s/\%/\%\%/g;
            if (@tokens) {
                $variable = shift (@tokens);
                # If we are properly inside of macro-function, then use shadow variables
                # in place of %{input-arg} OR %{return-arg}
                if ($insidemacrofunc && $$macrofuncargref{"O"}{$variable}) {
                    $subst      = "\"\" ";
                    $format    .= "${inputfmt}".$$macrofuncargref{"O"}{$variable}{"SHADOW_VAR"};
                } elsif ($insidemacrofunc && $$macrofuncargref{"I"}{$variable}) {
                    $subst      = "\"\" ";
                    $format    .= "${inputfmt}".$$macrofuncargref{"I"}{$variable}{"SHADOW_VAR"};
                } else {
                    $subst    = expand_variable($variable, $macroargref, $macrofuncargref);
                    $format  .= "${inputfmt}%s";
                    $args    .= $count ? ", $subst" : $subst;
                    $count++;
                }
            } else {
                $format  .= $inputfmt;
            }
        }
        $args   = "\n$argmargin$args" if ($count > 4);
        $format = unstringify($format);
    }
    if ($args) {
        return "len += sprintf(buffer + len, $format, $args);";
    } else {
        return "len += sprintf(buffer + len, $format);";
    }
}

sub expand_variable {
    my ($var, $macroargref, $macrofuncargref) = @_;
    if ($ismacrofunc) {
        $var = $$macrofuncargref{"I"}{$var}{"EMIT_STR"} if ($$macrofuncargref{"I"}{$var});
        $var = $$macrofuncargref{"O"}{$var}{"EMIT_STR"} if ($$macrofuncargref{"O"}{$var});
    } else {
        $var = $$macroargref{$var} if ($$macroargref{$var});
    }
    if ($var =~ m/(\d+)$/) {
        $var =~ s/(\d+)$//;
        return "get_str$var(parseState->parseData, $1)";
    } else {
        return "get_str$var(parseState->parseData)";
    }
}

sub populate_adler32_hash {
    my ($funcName) = @_;
    our $adler32 = Digest::Adler32->new;
    $adler32->add($funcName);
    my $adlerHash = hex $adler32->hexdigest;
    my %ilwertedAdlerHash = reverse %adlerHashForNames;
    if ( exists $ilwertedAdlerHash{$adlerHash} ) {
        die "Adler hash of $4 collides with $ilwertedAdlerHash{$adlerHash}\n"
    }
    $adlerHashForNames{$funcName} = $adlerHash;
}

sub write_adler_hash_file {
    open (ADLERHASHFILE, "> ptxForceInlineFuncAdlerHash.txt") || die "Cannot open ptxForceInlineFuncAdlerHash.txt\n";
    foreach my $key (keys %adlerHashForNames) {
        print ADLERHASHFILE "$adlerHashForNames{$key} $key\n";
    }
    close (ADLERHASHFILE);
}

sub mangle_macro_name {
    my ($name) = @_;
    $_ = "ptxMacroFuncs${profile}_${name}";
    s/\./DOT/g;
    $unmangle{$_} = $name;
    return $_;
}

sub unstringify {
    my $output = "\&macroBody[$offset]";
    $_ = $_[0]; s/\r\n/\n/g;
    $ptxfmt .= $_ . chr(0);
    $offset += length($_) + 1;
    return $output;
}

sub get_functions_prototype {
    my $proto;
    $proto      = "/*\n* Prototypes for all the functions defined in this file.\n*/\n\n";
    $proto     .= "void initMacroProfile${profile}(ptxParsingState ptxIR);\n";
    $proto     .= "void initMacroUtilFuncMap${profile}(ptxParsingState ptxIR);\n";
    foreach my $macro (@macros) {
        $proto .= sprintf ("String $macro (ptxParsingState parseState, char * macroBody);\n");
    }
    foreach my $macro (@macrofunctions) {
        $proto .= sprintf ("String $macro (ptxParsingState parseState, char * macroBody);\n");
    }
    return "$proto\n";
}

sub get_ptx_functions_code {
    my ($lmargin, $code, $count, $spaces) = (' ' x $tabsize);
    my @funcnames = sort keys %functions;
    $spaces    = (sort { $b <=> $a } (map(length, @funcnames)))[0];
    $code     .= "/*\n* Functions and declarations for processing .func bodies.\n*/\n\n";
    $code     .= sprintf("#define NO_FUNCTIONS %d\n",  0 + @funcnames);
    $code     .= sprintf("\nconst macroUtilFuncData_t utilFuncs%s[] = {", $profile);
    $code     .= sprintf("\n%s{NULL,%s NOT_PARSED}", $lmargin, ' ' x ($spaces - 4));
    foreach my $func (@funcnames) {
        $code .= sprintf(",\n%s{${func},%s NOT_PARSED}", $lmargin, ' 'x ($spaces - length($func)));
    }
    $code     .= "\n};\n";
    $code     .= "\lwoid initMacroUtilFuncMap${profile}(ptxParsingState ptxIR)\n{\n";
    $code     .= sprintf("%sstdMap_t macroUtilFuncMap${profile} = mapNEW(String, 128);\n\n", $lmargin);
    $code     .= sprintf("%sptxIR->utilFuncs        =  stdCOPY_S(utilFuncs${profile}, sizeof(utilFuncs${profile}));\n", $lmargin);
    $code     .= sprintf("%sptxIR->macroUtilFuncMap = macroUtilFuncMap${profile};\n", $lmargin);
    $code     .= sprintf("%sptxIR->numMacroUtilFunc = NO_FUNCTIONS+1;\n", $lmargin);
    foreach my $func (@funcnames) {
        $code .= sprintf("\n%smapDefine(macroUtilFuncMap${profile}, \"${func}\",%s (Pointer) %2d);", $lmargin, ' 'x ($spaces - length($func)), ++$count);
    }
    $code     .= "\n}\n";
    return $code;
}

sub get_macroname_maps {
    my ($spaces, $lmargin, $maps)   =  ((sort { $b <=> $a } (map(length, values(%unmangle))))[0], ' ' x $tabsize);
    $maps      = "\n/*\n* Functions and declarations for processing .MACRO bodies.\n*/\n\n";
    $maps     .= "char * fmtstr$profile = (char *) &ptxFmt$profile;\n";
    $maps     .= "int ptxFmt${profile}Size = sizeof(ptxFmt$profile);\n\n";
    $maps     .= "void initMacroProfile${profile}(ptxParsingState ptxIR) {\n";
    $maps     .= sprintf("%sinitMacroUtilFuncMap${profile}(ptxIR);\n", $lmargin);
    foreach my $macro (@macros) {
        $maps .= sprintf ("%smapDefine(ptxIR->macroMap, \"%s\",%s $macro);\n", $lmargin, $unmangle{$macro}, ' ' x ($spaces - length($unmangle{$macro})));
    }
    $spaces = (sort { $b <=> $a } (map(length, values(%adlerHashForNames))))[0];
    foreach my $macro (@macrofunctions) {
        my $adlerHashValue = $adlerHashForNames{$unmangle{$macro}};
        $maps .= sprintf ("%smapDefine(ptxIR->inlineFuncsMap, \"%llu\", %s $macro);\n", $lmargin, $adlerHashValue, ' ' x ($spaces - length($adlerHashValue)));
    }
    $maps     .= "}\n\n";
    return $maps;
}

sub write_c_file {
    open (CFILE, "> $cfile") || die "Cannot open $cfile.$!\n";
    print CFILE $content;
    close (CFILE);
}

sub init_elw {
    GetOptions(\%CmdLineOption,
        "profile=s",     # {Tesla|Fermi}.
        "i=s",           # Input file containing PTX macros
        "o=s",           # Output C file name
        "irmap=s",       # IR Definition Map file name
        "help",          # Show help message
    );
    if ($CmdLineOption{profile})     { $profile = $CmdLineOption{profile}; }
    if ($CmdLineOption{i})           { $ptxfile = $CmdLineOption{i};       }
    if ($CmdLineOption{o})           { $cfile   = $CmdLineOption{o};       }
    if ($CmdLineOption{irmap})       { $irmap   = $CmdLineOption{irmap};   }
    if ($CmdLineOption{help})        { print_readme(); exit 0;             }

    $newptxfile = "ptxInstructionMacros${profile}${profile}.bin2c";
    $fmtfile    = "ptxFmt${profile}${profile}.bin2c";

    %multiop    = ("==", "&&", "!=", "||");
    $tabsize    = 4;
    $offset     = 0;
}

sub print_readme {
    print<<'EndOfReadme';
===============================================================================
================================    README    =================================

This script is used to colwert ptx instruction macros to equivalent c code.
-------------------------------------------------------------------------------
usage:
  perl colwertPtxMacroToC.pl [options]
       -profile <profile>     : Profile {Fermi|Tesla}.
       -i       <file>        : Input file containing PTX macros
       -o       <file>        : Output C file name
       -irmap   <file>        : IR definitions map file name
       -help                  : Print this messege.

-------------------------------------------------------------------------------
Description:
------------

1. All substitutable variables(including % and $ types variables) allowed in
   macro language have a corresponding API of type get_<PARAMNAME>().
   These functions return a numeric value from corresponding enums.
   e.g. Value of %{ROUND} can be accessed with get_ROUND().
   Since TYPES may have multiple values, get_TYPES(int) takes index as parameter.

   Read 3. for details about colwersion of a .IF statement.

2. For the expansion in string, function get_str<PARAMNAME>() is used.
   e.g. to get string for ROUND, script should call a function get_strROUND().
   get_strTYPES() doesn't require any parameter.
   a. For macros -
      Macro arguments are acessed by get_strArg(i) where i is the index.
      e.g. %arg0 corresponds to get_strArg(0)
   b. For force-inline functions (refer (9)) -
      Input arguments are accessed by get_strIArg(i) where i is the index.
      Similarly, output arguments are accessed by get_strOArg(i) where i is the index.
      e.g. %inputArg0 corresponds to get_strArg(0) and
      %outputArg2 corresponds to get_strOArg(2) and

3. Environment variables (names that start with the "$" sigil)
   a. When checked as a string in a list of values:
      .IF ${VARIABLE} .in "value0 value1 ..."

      generated code is:
      if ( getMacroElw(VARIABLE) &&
           (isMacroElwEqual(VARIABLE, "value0") ||
            isMacroElwEqual(VARIABLE, "value1")...) )

   b. When checked as a Boolean against "true" or "false":
      .IF ${VARIABLE} == "true"

      generated code is:
      if ( getMacroElw(VARIABLE) )

      The condition is negated using "!" if compared with "false".

   b. When checked as a  number using a relational operator
      .IF ${VARIABLE} >= "50"

      generated code is:
      if ( getMacroElw(VARIABLE) >= 50 )

4. We use IRdef map file to choose mapping for % variable's value.
   a. If for a %VARIABLE, value pair we get mapping in map file
      generated code is: if ( get_VARIABLE() <operator> Mapping )
      e.g. Assuming the mapping ("ROUND"  ptxRN_MOD  ".rn")
           .IF "%{ROUND}" == ".rn" corresponds to
                                   if ( get_ROUND() == ptxRN_MOD )

   b. In force-inline functions (refer (9)), if the %VARIABLE is part of
      the input or the output arguments, then its use outside of the
      if conditionals would result in the generation of its corresponding
      shadow variable.

      Use of %{myInputArgument} would result in : <func-name>_myInputArgument

   c. In force-inline functions, if the input argument is used inside the
      if conditionals then RHS of the comparison must be an integer
      constant OR a sink. For the output argument, RHS can only be a sink.

      i) .IF "%{MyInputArg3}" >= 37
         would result in :
         if (getExpressionKindForArg(3, 0) == ptxIntConstantExpression &&
               getConstantValueOfInputArg(3) >= 37)

     ii) .IF "%{MyOutputArg2}" != "_"
          would result in :
          if (getExpressionKindForArg(2, 1) != ptxSinkExpression)

     Note : the second argument of getExpressionKindForArg() represents where
            the input under consideration is a return argument or an input argument

   d. If mapping is not found in the map file get_VARIABLE() is used.
      e.g. .IF "%{VARIABLE}" != "" corresponds to
                                   if ( get_VARIABLE() )

5. Generated c file contains functions for each macro defined with ".MACRO".
   In addition to #includes and extern declarations it also has a function
   "initMacroProfile<profile>" which contains mappping for each macro functions.
   e.g.
   In Tesla profile for sqrt macro it has ptxMacroFuncsTesla_sqrt function.
   initMacroProfileTesla function contains something like:
       mapDefine( macroMap, "sqrt", ptxMacroFuncsTesla_sqrt );

   Additionally, we need following line for each profile in the C code.
       int ptxFmt<profile>Size = sizeof(ptxFmt<profile>);

6. Generated C file also contains functions and mappings for processing ptx
   .func definitions found in the input ptx macro file.

7. This script also generates following additional files:
   a. Stripped ptx file w/o macro body or .func body
       ptxInstructionMacrosTeslaTesla.bin2c for Tesla profile
       ptxInstructionMacrosFermiFermi.bin2c for Fermi profile
   b. Format strings for macro bodies
       ptxFmtTeslaTesla.bin2c or ptxFmtFermiFermi.bin2c
   c. Files for each .func definition with body
       <funcname>Tesla.bin2c <funcname>Fermi.bin2c
   Naming convention for each of these files is such that we can get the name
   of the ull array generated by bin2c after stripping <profile>.bin2c

8. The macro ".INLINE" can be used to force the preprocessor to inline
   a function call. The macro ".INLINE_WHEN" can be used to
   conditionally inline a call. See the following file for more
   details, including limitations and restrictions:

   https://p4viewer/get///sw/compiler/gpgpu/doc/spec/ptxFE/INLINE%20macro.docx

   For every function call annotated with these macros, the inlined
   body of the callee is dumped in a corresponding *.inlined.ptx
   file. This file is not used by only tool, but made available only
   for manual inspection.

9. Functions with ".FORCE_INLINE" will inline the entire function body
   at the call-site. ".INLINE" is used at the call-site which MUST be present
   within this file and so the expansion happens within this file. Whereas
   ".FORCE_INLINE" attributed functions will be called from the user PTX.
   This is the main difference between the two variants. Refer the doc for details:

   https://p4viewer/get///sw/compiler/gpgpu/doc/spec/ptxFE/INLINE%20macro.docx

10.Mappings for functions with ".FORCE_INLINE" are created in "inlineFuncsMap".
   In order to avoid leaking names of such functions, mappings are created with
   adler32 hash of the function name appearing as a key. The script aborts in case
   of hash collisions. The adler32 hash keys are populated as strings in the map.
   This is a WAR to handle consistent results across 32bit and 64bit builds.
   stdMap API take Pointer as argument which will be 32bit on 32bit builds and hence
   implicitly values larger than 32bits will be truncated. The consumer needs to
   compute adler32 hash key of lookup name before searching for function bodies
   into the map. The adler32 perl implementation is taken from below repository:

   https://github.com/gisle/digest-adler32

===============================================================================
EndOfReadme
}

###############################################################################
###############################  END OF SCRIPT  ###############################
###############################################################################
