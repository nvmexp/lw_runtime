#!/usr/bin/elw perl -w
use warnings;
use strict;

# Add this script's directory to the modules search path.
use File::Basename qw( dirname );
use lib dirname($0);

use modules::Error;
use modules::Utils;
use modules::ScriptName;
use modules::PathUtils;

my @defaultParamPaths = (
    '../cop_parseasm/vc10/cop_parseasm.vcxproj.genparams',
    '../../../drivers/compiler/gpgpucomp/tools/ptxas/ptxas100.vcxproj.genparams',
    '../../../drivers/compiler/tools/lwvm/vc10/lwvmc100.vcxproj.genparams',
    '../../../drivers/compiler/gpgpucomp/tools/finalizer/finalizer100.vcxproj.genparams'
);

my $cwd = '';
my $runBuilds = 1;
my $verbose = 0;
my @paramFilePaths = ();
my $params;
my $capitalPaths = {};
my $filters = {};
my @order = ();

sub StripQuotes($)
{
    my ($str) = @_;
    $str =~ s/^['"]([^'"]*)['"]$/$1/;
    return $str;
}

sub SubstituteVars($$)
{
    my ($str, $varHash) = @_;

    foreach my $var (keys %{$varHash}) {
        $str =~ s/%%\{$var\}%%/$varHash->{$var}/g;
    }
    return $str;
}

sub GetParamArr
{
    my @paramPath = @_;
    my $ref = $params;

    while (@paramPath) {
        my $entry = shift @paramPath;

        if (! exists $ref->{$entry}) {
            return [];
        }
        $ref = $ref->{$entry};
    }

    return $ref;
}

sub PrintHelp($)
{
    my ($error) = @_;

    use modules::PrintOpt;

    &PrintOptConfig({limit=>80, indent=>16});

    print "\nUsage: $scriptName [options] [<ProjectParamFile> <ProjectParamFile> ...]\n\n";
    print "Options:\n";
    &PrintOpt('-nb|no-build', 'Skip running relevant build(s) before parsing build collateral (assumes user already performed builds).');
    &PrintOpt('-v|verbose',   'Display verbose build output.');
    &PrintOpt('-h|help',      'Print the help screen.');

    print "\n";

    &PrintOptConfig({indent=>1, prefix=>''});
    &PrintOptWrap('',
        'ProjectParamFile will be named with a ".genparams" suffix in the same directory 
          as the resulting Visual Studio project & filter files. If no ProjectParamFiles
          are specified, then the following param files are processed automatically:'
     );
     print "\n";
     foreach my $path (@defaultParamPaths) {
         printf "    %s\n", &AbsPath($path, $scriptDir);
     }
     print "\n";


    &Error("Problem processing the command line.") if $error;
}

sub ProcessCmdLine()
{
    use Getopt::Long;
    GetOptions (
        "nb|no-build"     => \my $noBuild,
        "v|verbose"       => \$verbose,
        "h|help"          => \my $help,
    ) or &PrintHelp(1);

    if (defined $help) {
        &PrintHelp(0);
        exit(0);
    }

    $runBuilds = defined $noBuild ? 0 : 1;

    my @genList;
    if (!@ARGV) {
        for my $p (@defaultParamPaths) {
            push @genList, join '/', $scriptDir, $p;
        }
    } else {
        @genList = @ARGV;
    }

    foreach my $f (@genList) {
        &Error ("Specified param file does not exist:\n  '$f'") unless -f $f;
        push @paramFilePaths, &AbsPath($f);
    }
}

sub CanonPathCapitalization($)
{
    my ($path) = @_;

    # strip trailing '/'s
    $path =~ s/\/*$//;

    my $lcPath = lc $path;

    if (exists $capitalPaths->{$lcPath}) {
        return $capitalPaths->{$lcPath};
    }

    my $ret = '';
    my @front = split /\//, $path;
    my @back  = (pop @front);

    # Move backwards through the path until we hit in the hash or we encounter a volume.
    while (@front) {
        $lcPath = lc join('/', @front);
        last if $lcPath =~ /^[a-z]:$/;

        if (exists $capitalPaths->{$lcPath}) {
            $ret = $capitalPaths->{$lcPath};
            @front = split '/', $ret;
            last;
        }

        unshift @back, pop @front;
    }

    while (@back) {
        my $tmp;
        my $lwr = shift @back;

        my $globPath = join '/', @front, "$lwr*";
        my @glob = glob $globPath;

        if (!@glob) {
            # no results... must be a capitalization mismatch.
            # Try again without specifying 'lwr' at the end.
            $globPath = join '/', @front, '*';
            @glob = glob $globPath;
        }

        my @matches = grep { /\/$lwr$/i } @glob;

        if (!@matches) {
            # path doesn't exist from here on.
            # Just cache and return the remainder of the original path.
            push @front, $lwr;
            $tmp = join '/', @front;
            $capitalPaths->{lc $tmp} = $tmp;

            while (@back) {
                push @front, shift @back;
                $tmp = join '/', @front;
                $capitalPaths->{lc $tmp} = $tmp;
            }
            last;
        }
            
        if (@matches > 1) {
            &Error("Multiple files match!");
        }

        my $match = $matches[0];
        $match =~ s/^.*\/+//g;

        push @front, $match;
        $tmp = join '/', @front;
        $capitalPaths->{lc $tmp} = $tmp;
    }
    
    return join '/', @front;
}

sub FindRelativePath($)
{
    my ($path) = @_;

    return $path unless &IsRelativePath($path);

    my @searchOrder = exists $params->{'RELATIVE_SEARCH_BASES'} ?
        @{$params->{'RELATIVE_SEARCH_BASES'}} :
        ('');

    foreach my $searchBase (@searchOrder) {
        my @result = ();
        &FindFiles($searchBase, "$path*", \@result, 1);
        foreach my $rslt (@result) {
            if ($rslt =~ /$path$/) {
                return $rslt;
            }
        }
    }
    return $path;
}

sub CanonPath($)
{
    my ($path) = @_;

    $path = &SlashPath($path);
    $path =~ s/\/\.\//\//g;
    $path =~ s/^\.\///;
    if (&IsRelativePath($path)) {
        $path = join '/', $cwd, &FindRelativePath($path);
    }

    $path = &WinPath($path);

    # Remove any '../' oclwrences after the begining of the path.
    while ($path =~ /[^\/\.]+\/\.\./) {
        $path =~ s/\/+([^\/\.]+)\/+\.\.//g;
    }

    return &CanonPathCapitalization($path);
}

sub CanonRelPath($)
{
    my ($path) = @_;

    $path = &CanonPath($path);

    my @pathArr = split /\//, $path;
    my @cwdArr  = split /\//, $cwd;
    my @dots    = ();
    while (@pathArr && @cwdArr && lc $pathArr[0] eq lc $cwdArr[0]) {
        shift @pathArr;
        shift @cwdArr;
    }

    while (@cwdArr) {
        push @dots, '..';
        shift @cwdArr;
    }

    my $ret = join '/', @dots, @pathArr;
    return $ret;
}

sub FindFiles($$$$)
{
    my ($path, $globStr, $outArr, $firstMatch) = @_;
    $path    =~ s/\n//g;
    $globStr =~ s/\n//g;

    my $prefix = ($path eq '' ? '' : $path . '/');
    my @result = glob "${prefix}${globStr}";
    push @$outArr, @result;
    return if @result && $firstMatch;

    opendir (my $DIR, $path) or return;
    for my $item (readdir($DIR)) {
        $item =~ s/\s*$//;
        next if ($item =~ /\.\.?[\/\\]?/ || $item eq '');
        if (-d "$prefix$item") {
            &FindFiles("$prefix$item", $globStr, $outArr, $firstMatch);
        }
    }
    closedir $DIR;
}

sub ReadConfigInfo($$$)
{
    my ($paramsPath, $paramsRef, $filterOrder) = @_;

    print "Parsing project generation config file...\n";

    $params = $paramsRef;

    open (my $in, "<", $paramsPath) or die "Can't open param file '$paramsPath'";

    my @state = ();
    my $hashRef = $params;

    while (<$in>) {
        my $line = $_;

        # Skip comment lines.
        next if ($line =~ /^\s*#/);

        if ($line =~ /^\s*(\w+)\s*=\s*'([^']*)'\s*$/) {
            my $varName = $1;
            my $val     = $2;
            if (! defined $hashRef) {
                $hashRef = $params;
                if (@state) {
                    for (my $ii=$#state; $ii > 0; --$ii) {
                        $hashRef = $hashRef->{$state[$ii]};
                    }
                    $hashRef->{$state[0]} = {};
                    $hashRef = $hashRef->{$state[0]};
                }
            }
            $hashRef->{$varName} = $val;
        }

        if ($line =~ /^\s*'([^']*)'\s*$/) {
            my $listVal = $1;
            if (! defined $hashRef) {
                $hashRef = $params;
                for (my $ii=$#state; $ii > 0; --$ii) {
                    $hashRef = $hashRef->{$state[$ii]};
                }
                $hashRef->{$state[0]} = [];
                $hashRef = $hashRef->{$state[0]};
            }
            push @$hashRef, $listVal;
        }

        if ($line =~ /^\s*'([^']*)'\s*=>\s*'([^']*)'/) {
            my $mapped = $1;
            my $val    = $2;

            if (! defined $hashRef) {
                $hashRef = $params;
                for (my $ii=$#state; $ii > 0; --$ii) {
                    $hashRef = $hashRef->{$state[$ii]};
                }
                $hashRef->{$state[0]} = {};
                $hashRef = $hashRef->{$state[0]};
            }

            $hashRef->{$mapped} = $val;

            if (join ('/', @state) eq 'FILTERS') {
                push @$filterOrder, $mapped;
            }
        }

        if ($line =~ /\s*<(\/)?\s*([\w]+)\s*>/) {
            my $stateArg = $2;
            if (!defined $1) {
                if (!defined $hashRef) {
                    $hashRef = $params;
                    for (my $ii=$#state; $ii > 0; --$ii) {
                        $hashRef = $hashRef->{$state[$ii]};
                    }
                    $hashRef->{$state[0]} = {};
                }
                unshift @state, $stateArg;
                undef $hashRef;
            } else {
                if (shift @state ne $stateArg) {
                    &Error("unmatched end token in param file '</$2>'");
                }

                # Path in the params hash must already exist since we
                # are popping up to a parent state.
                $hashRef = $params;
                for (my $ii=$#state; $ii >= 0; --$ii) {
                    $hashRef = $hashRef->{$state[$ii]};
                }
            }
        }
    }

    close ($in);

    $filters = $params->{'FILTERS'};

    print "done!\n\n";
}

sub DoClobber($$)
{
    my ($projHash, $config) = @_;

    return 0 if (!exists $params->{'BUILD_INFO'}->{$config}->{'CLOBBER_CMD'});

    my $projVars = $projHash->{'VARIABLES'};

    print "Clobbering $config...\n";

    my $cmd = &SubstituteVars($params->{'BUILD_INFO'}->{$config}->{'CLOBBER_CMD'}, $projVars);
    my $ret = &RunCmd($cmd, {echo=>1, outnull=>($verbose ? 0 : 1)});

    print "done!\n\n";
    return $ret;
}

sub DoBuild($$)
{
    my ($projHash, $config) = @_;

    my $projVars = $projHash->{'VARIABLES'};

    print "Building $config...\n";

    my $cmd = &SubstituteVars($params->{'BUILD_INFO'}->{$config}->{'BUILD_CMD'}, $projVars);
    my $ret = &RunCmd($cmd, {echo=>1, outnull=>($verbose ? 0 : 1)});

    print "done!\n\n";
    return $ret;
}

sub DoBuilds($)
{
    my ($projHash) = @_;

    return if (!$runBuilds);
    
    my @configs = sort { $a cmp $b } keys %{$params->{'BUILD_INFO'}};
    my $error = 0;

    # error during clobber can be ignored.
    foreach my $config (@configs) {
        &DoClobber($projHash, $config);
    }

    foreach my $config (@configs) {
        $error |= &DoBuild($projHash, $config);
    }

    &Error("Build failure.") if ($error);
}


sub GetFilter($$)
{
    my ($projHashRef, $relPath) = @_;

    foreach my $prefix (@order) {
        my $substPrefix = &SubstituteVars($prefix, $projHashRef->{'VARIABLES'});
        my @prefixArr   = split /\//, $substPrefix;
        my @relPathArr  = split /\//, $relPath;

        while (@prefixArr) {
            last if lc $prefixArr[0] ne lc $relPathArr[0];
            shift @prefixArr;
            shift @relPathArr;
        }

        # Matched filter!
        if ($prefix eq '' || (!@prefixArr && $relPathArr[0] ne '..')) {
            # Remove the file name from the relPath.
            pop @relPathArr;
            # Remove '..' entries from the front of path (catch-all case).
            while (@relPathArr && $relPathArr[0] eq '..') {
                shift @relPathArr;
            }
            my $ret = join '\\', $filters->{$prefix}, @relPathArr;
            return $ret;
        }
    }

    print "Error: No filter found for '$relPath'\n";
    exit 1;
    return '';
}

sub EmitGenFileWarning($)
{
    my ($out) = @_;

    my $genFileWarning =
"<!--
  |
  | WARNING: This file is generated.  Do not modify this file directly!
  |
  |  You probably intend to modify contents of the corresponding
  |   <name>.vcxproj.template or <name>.vcxproj.genparams.
  |
  |  Once modifications are made, run the update script at the following path:
  |   %s
  |
  |  NOTE: If you are adding a file to the compiler, you should modify copfiles.inc
  |   and then simply run the updateProjects script mentioned above.  Project files
  |   will be automatically generated based off builds results.
  |
  -->";

    printf $out "$genFileWarning\n", &CanonRelPath("$scriptDir/$scriptName");
}

sub EmitFilterDecl($$$)
{
    my ($out, $linePrefix, $filter) = @_;

    use Digest::MD5  qw(md5_hex);

    my $md5 = md5_hex($filter);

    $md5 = substr($md5, 0, 8) . '-'.
           substr($md5, 8, 4) . '-'.
           substr($md5,12, 4) . '-'.
           substr($md5,16, 4) . '-'.
           substr($md5,20,12);

    print $out "$linePrefix<Filter Include=\"$filter\">\n";
    print $out "$linePrefix  <UniqueIdentifier>{$md5}</UniqueIdentifier>\n";
    print $out "$linePrefix</Filter>\n";
}

sub EmitFilterFile($$)
{
    my ($projHashRef, $buildHashRef) = @_;

    my $outPath = $projHashRef->{'FILTERS_PATH'};

    print "Emitting filter file...\n";

    # Check out the filter file.
    $outPath = &CanonPath($outPath);
    &RunCmd("p4 edit $outPath", {echo=>1, outnull=>($verbose ? 0 : 1)});

    my %filterHash = ();
    my %headers = ();
    my %sources = ();
    my $extraSources = &GetParamArr('EXTRA_SOURCES');
    my $visualizers  = &GetParamArr('VISUALIZERS');

    # Collect filters across all build configs.
    foreach my $config (keys %$buildHashRef) {
        my $srcHashRef = $buildHashRef->{$config}->{'SOURCE_FILES'};
        my $hdrHashRef = $buildHashRef->{$config}->{'HEADER_FILES'};

        foreach my $src (keys %$srcHashRef) {
            $sources{$src} = 1;
        }

        foreach my $hdr (keys %$hdrHashRef) {
            $headers{$hdr} = 1;
        }
    }

    # Create filters for all files being included in the project.
    foreach my $f (keys %sources, keys %headers, @$extraSources, @$visualizers) {
        my $filter = &GetFilter($projHashRef, $f);
        while ($filter ne '') {
            $filterHash{$filter} = 1;
            $filter =~ s/\\?[^\\]*$//;
        }
    }

    open (my $out, ">", $outPath) or die "Can't open output file $outPath.";

    print $out '<?xml version="1.0" encoding="utf-8"?>
';
    &EmitGenFileWarning($out);

    print $out '<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
';
    print $out "  <ItemGroup>\n";
    foreach my $filter (sort { $a cmp $b } keys %filterHash) {
        &EmitFilterDecl($out, "    ", $filter);
    }
    print $out "  </ItemGroup>\n";

    print $out "  <ItemGroup>\n";
    foreach my $src (sort { $a cmp $b } @$extraSources) {
        my $srcBackSlash = $src;
        $srcBackSlash =~ s/\//\\/g;
        printf $out "    <None Include=\"$srcBackSlash\">\n";
        printf $out "      <Filter>%s</Filter>\n", &GetFilter($projHashRef, $src);
        printf $out "    </None>\n";
    }
    foreach my $src (sort { $a cmp $b } @$visualizers) {
        my $srcBackSlash = $src;
        $srcBackSlash =~ s/\//\\/g;
        printf $out "    <Natvis Include=\"$srcBackSlash\">\n";
        printf $out "      <Filter>%s</Filter>\n", &GetFilter($projHashRef, $src);
        printf $out "    </Natvis>\n";
    }
    print $out "  </ItemGroup>\n";

    print $out "  <ItemGroup>\n";
    foreach my $src (sort { $a cmp $b } keys %sources) {
        my $srcBackSlash = $src;
        $srcBackSlash =~ s/\//\\/g;
        printf $out "    <ClCompile Include=\"$srcBackSlash\">\n";
        printf $out "      <Filter>%s</Filter>\n", &GetFilter($projHashRef, $src);
        printf $out "    </ClCompile>\n";
    }
    print $out "  </ItemGroup>\n";

    print $out "  <ItemGroup>\n";
    foreach my $hdr (sort { $a cmp $b } keys %headers) {
        my $hdrBackSlash = $hdr;
        $hdrBackSlash =~ s/\//\\/g;
        printf $out "    <ClInclude Include=\"$hdrBackSlash\">\n";
        printf $out "      <Filter>%s</Filter>\n", &GetFilter($projHashRef, $hdr);
        printf $out "    </ClInclude>\n";
    }
    print $out "  </ItemGroup>\n";

    print $out '</Project>';
    close $out;

    print "done!\n\n";
}

sub EmitProjectFile($$)
{
    my ($projHashRef, $buildHashRef) = @_;

    my $projVars     = $projHashRef->{'VARIABLES'};
    my $templatePath = $projHashRef->{'TEMPLATE_PATH'};
    my $outPath      = $projHashRef->{'PROJECT_PATH'};

    print "Emitting project file...\n";

    # Check out the project file.
    $outPath = &CanonPath($outPath);
    &RunCmd("p4 edit $outPath", {echo=>1, outnull=>($verbose ? 0 : 1)});
    
    my %defines        = ();
    my %includes       = ();
    my %forcedIncludes = ();
    my %sources        = ();
    my %headers        = ();
    my $extraSources   = &GetParamArr('EXTRA_SOURCES');
    my $visualizers    = &GetParamArr('VISUALIZERS');

    foreach my $config (keys %$buildHashRef) {
        foreach my $def (sort { $a cmp $b } keys %{$buildHashRef->{$config}->{'DEFINES'}}) {
            if (defined $defines{$config}) {
                $defines{$config} .= ";" . $def . $buildHashRef->{$config}->{'DEFINES'}->{$def};
            } else {
                $defines{$config} = $def . $buildHashRef->{$config}->{'DEFINES'}->{$def};
            }
        }

        foreach my $inc (sort { $a cmp $b } keys %{$buildHashRef->{$config}->{'INCLUDE_PATHS'}}) {
            if (defined $includes{$config}) {
                $includes{$config} .= ';' . $inc;
            } else {
                $includes{$config} = $inc;
            }
        }

        foreach my $fInc (sort { $a cmp $b } keys %{$buildHashRef->{$config}->{'FORCED_INCLUDES'}}) {
            next if defined $forcedIncludes{$fInc};
            if (defined $forcedIncludes{$config}) {
                $forcedIncludes{$config} .= ';' . $fInc;
            } else {
                $forcedIncludes{$config} = $fInc;
            }
        }

        foreach my $src (keys %{$buildHashRef->{$config}->{'SOURCE_FILES'}}) {
            next if defined $sources{$src};

            $sources{$src} = 1;
        }

        foreach my $hdr (keys %{$buildHashRef->{$config}->{'HEADER_FILES'}}) {
            next if defined $headers{$hdr};

            $headers{$hdr} = 1;
        }

        if (exists $params->{'EXTRA_INCLUDES'}) {
            foreach my $inc (@{$params->{'EXTRA_INCLUDES'}}) {
                if (defined $includes{$config}) {
                    $includes{$config} .= ';' . $inc;
                } else {
                    $includes{$config} = $inc;
                }
            }
        }
    }

    open (my $in,  "<", $templatePath) or die "Can't open input file $templatePath.";
    open (my $out, ">", $outPath)      or die "Can't open output file $outPath.";

    # First line contains the XML directive... needs to be top line of output.
    my $line = <$in>;
    $line =~ s/\s*$//;
    print $out "$line\n";

    # Print the warning that the project file is generated.
    &EmitGenFileWarning($out);

    my $assumedCfg = 'debug';
    while (<$in>) {
        $line = &SubstituteVars($_, $projVars);
        $line =~ s/\s*$//;

        while ($line =~ /%%\{DEFINES_([A-Z]+)\}%%/) {
            my $txt = $1;
            my $cfg = lc $txt;
            $cfg = exists $defines{$cfg} ? $cfg : $assumedCfg;
            my $subst = exists $defines{$cfg} ? $defines{$cfg} : '';
            $line =~ s/%%\{DEFINES_$txt\}%%/$subst/g;
        }

        while ($line =~ /%%\{INCLUDES_([A-Z]+)\}%%/) {
            my $txt = $1;
            my $cfg = lc $txt;
            $cfg = exists $includes{$cfg} ? $cfg : $assumedCfg;
            my $subst = exists $includes{$cfg} ? $includes{$cfg} : '';
            $line =~ s/%%\{INCLUDES_$txt\}%%/$subst/g;
        }

        while ($line =~ /%%\{FORCED_INCLUDES_([A-Z]+)\}%%/) {
            my $txt = $1;
            my $cfg = lc $txt;
            $cfg = exists $forcedIncludes{$cfg} ? $cfg : $assumedCfg;
            my $subst = exists $forcedIncludes{$cfg} ? $forcedIncludes{$cfg} : '';
            $line =~ s/%%\{FORCED_INCLUDES_$txt\}%%/$subst/g;
        }

        if ($line =~ /^([^%]*)%%\{SOURCES_GROUP\}%%(.*)/) {
            $line  = $2;
            print $out "$1\n";
            foreach my $src (sort { $a cmp $b } keys %sources) {
                $src =~ s/\//\\/g;
                print $out "    <ClCompile Include=\"$src\" />\n";
            }
        }
        if ($line =~ /^([^%]*)%%\{EXTRA_SOURCES_GROUP\}%%(.*)/) {
            $line  = $2;
            print $out "$1\n";
            foreach my $src (sort { $a cmp $b } @$extraSources) {
                $src =~ s/\//\\/g;
                print $out "    <None Include=\"$src\" />\n";
            }
            foreach my $src (sort { $a cmp $b } @$visualizers) {
                $src =~ s/\//\\/g;
                print $out "    <Natvis Include=\"$src\" />\n";
            }
        }

        if ($line =~ /([^%]*)%%\{HEADERS_GROUP\}%%(.*)/) {
            $line  = $2;
            print $out "$1\n";
            foreach my $hdr (sort { $a cmp $b } keys %headers) {
                $hdr =~ s/\//\\/g;
                print $out "    <ClInclude Include=\"$hdr\" />\n";
            }
        }
        print $out "$line\n";
    }

    print "done!\n\n";
}


sub Init($)
{
    my ($paramPath) = @_;

    print "Init...\n";

    $capitalPaths = {};
    $filters = {};
    @order = ();

    {
        # Script assumes working directory is the same as the params file.
        my $dir = &DirName($paramPath);
        chdir "$dir";
        $paramPath = &BaseName($paramPath);
    }

    {
        use Cwd;
        $cwd = getcwd();
        $cwd = &CanonPath($cwd);
    }
    print "done!\n\n";
}

sub ParseBuildFiles($$)
{
    my ( $projHashRef, $config, $outHashRef ) = @_;

    print "Parsing build files ($config)...\n";

    my $objDir = &SubstituteVars($params->{'BUILD_INFO'}->{$config}->{'OBJ_PATH'}, $projHashRef->{'VARIABLES'});

    if (!defined $outHashRef->{$config}) {
        $outHashRef->{$config} = {};
    }

    my $configHashRef = $outHashRef->{$config};

    my $defHashRef = $configHashRef->{'DEFINES'}       = {};
    my $incHashRef = $configHashRef->{'INCLUDE_PATHS'} = {};
    my $srcHashRef = $configHashRef->{'SOURCE_FILES'}  = {};
    my $hdrHashRef = $configHashRef->{'HEADER_FILES'}  = {};
    my $fIncHashRef = $configHashRef->{'FORCED_INCLUDES'} = {};

    # Get the names of all the .obj_cl_params files.
    my @files = ();
    &FindFiles($objDir, "*$params->{'CL_PARAMS_SUFFIX'}", \@files, 0);


    print "  snooping cl_param files...\n";

    # Traverse the cl_params files to extract preprocessing defines, include paths and source paths.
    foreach my $f (@files) {
        open (my $inputHandle, "<", $f) or die "Can't open input file $f.";

        my $sourcePath;
        while (<$inputHandle>) {
            my $line = $_;
            my @lineArr = split /\s+/, $line;
            while (@lineArr) {
                my $token = &StripQuotes(shift @lineArr);
                if ($token =~ /^-D/) {
                    $token =~ s/^-D//;

                    if ($token eq '') {
                        # whitespace between -D and the define.
                        $token = shift @lineArr;
                    }

                    # Strip quotes from the define string.
                    $token = &StripQuotes($token);

                    if ($token =~ /([^=]+)=([^=]+)/) {
                        $defHashRef->{$1} = "=$2";
                    } else {
                        $defHashRef->{$token} = "";
                    }
                }

                elsif ($token =~ /^-I/) {
                    $token =~ s/^-I//;

                    if ($token eq '') {
                        # whitespace between -I and the path.
                        $token = shift @lineArr;
                    }

                    # Strip quotes from the include string.
                    $token = &StripQuotes($token);

                    my $incPath = &CanonRelPath($token);
                    $incHashRef->{$incPath} = 1;
                }

                elsif ($token =~ /^-FI/) {
                    $token =~ s/^-FI//;

                    if ($token eq '') {
                        # whitespace between -FI and the path.
                        $token = shift @lineArr;
                    }

                    # Strip quotes from the include string.
                    $token = &StripQuotes($token);

                    my $fIncPath = &FindRelativePath($token);
                    $fIncHashRef->{$fIncPath} = 1;
                }

                elsif ($token =~ /^[^-]/) {
                    $sourcePath = &CanonRelPath($token);
                }
            }
        }
        close $inputHandle;

        if (!defined $sourcePath) {
            &Error("source path undefined.");
        }
        $srcHashRef->{$sourcePath} = 1;
    }

    @files = ();
    &FindFiles($objDir, "*$params->{'MAKE_DEP_SUFFIX'}", \@files, 0);


    print "  snooping make depend files...\n";

    # Traverse the make dependence files to extract relevant header paths.
    foreach my $f (@files) {
        open (my $inputHandle, "<", $f) or die "Can't open input file $f.";

        my $state = "INIT";
        while (<$inputHandle>) {
            my $line = $_;
            $line =~ s/\s*$//;
            
            if ($state eq 'INIT' && $line =~ /[^\s:]\.obj:.*\\/) {
                $state = 'HEADERS';
            }

            elsif ($state eq 'HEADERS' && $line =~ /^\s*([^\s]+)\s*\\?/) {
                my $hdrPath = &CanonRelPath($1);
                $hdrHashRef->{$hdrPath} = 1;
            }

            else {
                last;
            }
        }
        close $inputHandle;
    }

    # Move source pattern files from Headers to Sources.
    # Also remove any Headers which don't exist.
    foreach my $hdr (keys %$hdrHashRef) {
        if (! -f $hdr) {
            &Error("Header file does not exist:\n  $hdr");
        }

        if ($hdr =~ /\.c(pp)?$/) {
            if (!defined $srcHashRef->{$hdr}) {
                $srcHashRef->{$hdr} = 1;
            }
            delete $hdrHashRef->{$hdr};
        }
    }

    # Remove any Sources which don't exist.
    foreach my $src (keys %$srcHashRef) {
        if (! -f $src) {
            &Error("Source file does not exist:\n  $src");
        }
    }

    # Remove any Include search paths which don't exist.
    foreach my $inc (keys %$incHashRef) {
        if (! -d $inc) {
            delete $incHashRef->{$inc};
        }
    }

    print "done!\n\n";
}

sub main
{
    &ProcessCmdLine();

    for my $paramPath (@paramFilePaths) {

        &Init($paramPath);

        my %paramInfo = ();

        &ReadConfigInfo($paramPath, \%paramInfo, \@order);

        foreach my $proj (sort keys %{$paramInfo{'PROJECT_INFO'}}) {
            print "======== Project = $proj ========\n";

            my $projHash = $paramInfo{'PROJECT_INFO'}{$proj};

            &DoBuilds($projHash);

            my %buildInfo = ();
            foreach my $config (sort keys %{$paramInfo{'BUILD_INFO'}}) {
                &ParseBuildFiles($projHash, $config, \%buildInfo);
            }

            &EmitFilterFile ($projHash, \%buildInfo);
            &EmitProjectFile($projHash, \%buildInfo);
        }
    }

    return 0;
}

exit(&main());
