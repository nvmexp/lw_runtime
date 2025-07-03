#!/usr/bin/elw perl

# mkvcproj.pl - A script to make the visual studio project containing all the sources.
# Inputs: ARG[0] - A file containing a space-separated list of sources and headers.  An individual
#                  entry of "--" will be used by the script to switch from processing source files
#                  to header files.

use strict;

my @sources;                    # list of source files
my @headers;                    # list of header files

# List of project configurations.
my @configs = ("Debug|NX32", "Debug|Win32",
               "Release|NX32", "Release|Win32",
               "Debug|x64", "Release|x64");

# Get the file name containing all of the sources (a list of space-separated sources and headers).
# The sources and headers are divided by a separate "--" argument: this is used later in the script.
my $input_sources_file_name = $ARGV[0];

# Opens the source list  file for reading.
open my $input_sources_file_handle, '<', $input_sources_file_name or die "Can't open $input_sources_file_name";

# Read in the entirety of the input source list file into a single string.
my $input_sources_file_content = do { local $/; <$input_sources_file_handle> };

# Split the source list by space to create the arguments array.
my @args = split(' ', $input_sources_file_content);

# Build up source and header files from the argument list.  We start with
# source files and jump to headers when we see "--".
my $isheader = 0;
for my $file (@args) {
    if ($file =~ m/^--$/) {
        $isheader = 1;
        next;
    }
    $file =~ s#/#\\#g;
    push @sources, $file if !$isheader;
    push @headers, $file if $isheader;
}

# Add CheetAh-specific files that will normally not be on the source file list
# when building for Windows.
push @sources, "elw\\tegra_main.cpp";
push @sources, "elw\\tegra_utils.cpp";

########################################################################

print "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
print "<Project DefaultTargets=\"Build\" ToolsVersion=\"14.1\" ".
  "xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">\n";

print "  <ItemGroup Label=\"ProjectConfigurations\">\n";
for (@configs) {
    my ($cfg, $plat) = split(/\|/, $_);
    print "    <ProjectConfiguration Include=\"$_\">\n";
    print "      <Configuration>$cfg</Configuration>\n";
    print "      <Platform>$plat</Platform>\n";
    print "    </ProjectConfiguration>\n";
}
print "  </ItemGroup>\n";

print "  <PropertyGroup Label=\"Globals\">\n";
print "    <ProjectGuid>{7E3123C2-8592-40E0-B855-96DD9F2709EB}</ProjectGuid>\n";
print "    <RootNamespace>lwntest</RootNamespace>\n";
print "    <WindowsTargetPlatformVersion>8.1</WindowsTargetPlatformVersion>\n";
print "  </PropertyGroup>\n";
print "  <Import Project=\"\$(VCTargetsPath)\\Microsoft.Cpp.Default.props\" />\n";

for (@configs) {
    next if m/NX32$/;
    my $isdebug = m/^Debug/ ? "true" : "false";
    my $isx86 = m/Win32$/ ? "true" : "false";
    print "  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='$_'\" ".
      "Label=\"Configuration\">\n";
    print "    <ConfigurationType>Application</ConfigurationType>\n";
    print "    <UseDebugLibraries>$isdebug</UseDebugLibraries>\n";
    print "    <PlatformToolset>v141</PlatformToolset>\n";
    if ($isdebug eq "false") {
        print "    <WholeProgramOptimization>true</WholeProgramOptimization>\n";
    }
    print "    <CharacterSet>NotSet</CharacterSet>\n";
    if ($isx86 eq "false") {
        print "    <UseOfMfc>false</UseOfMfc>\n";
    }
    print "  </PropertyGroup>\n";
}

for (@configs) {
    next if !m/NX32$/;
    print "  <PropertyGroup Label=\"Configuration\" Condition=\"'\$(Configuration)|\$(Platform)'=='$_'\">\n";
    print "    <PlatformToolset>v141</PlatformToolset>\n";
    print "  </PropertyGroup>\n";
}

print "  <Import Project=\"\$(VCTargetsPath)\\Microsoft.Cpp.props\" />\n";
print "  <ImportGroup Label=\"ExtensionSettings\">\n";
print "  </ImportGroup>\n";
print "  <ImportGroup Label=\"Shared\">\n";
print "  </ImportGroup>\n";
  
for (@configs) {
    next if m/NX32$/;
    print "  <ImportGroup Label=\"PropertySheets\" Condition=\"'\$(Configuration)|\$(Platform)'=='$_'\">\n";
    print "    <Import Project=\"\$(UserRootDir)\\Microsoft.Cpp.\$(Platform).user.props\" ".
        "Condition=\"exists('\$(UserRootDir)\\Microsoft.Cpp.\$(Platform).user.props')\" Label=\"LocalAppDataPlatform\" />\n";
    print "  </ImportGroup>\n";
}

print "  <PropertyGroup Label=\"UserMacros\" />\n";

for (@configs) {
    next if m/NX32$/;
    print "  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='$_'\">\n";
    print "    <OutDir>\$(SolutionDir)..\\_out\\\$(Configuration)_\$(Platform)\\</OutDir>\n";
    print "    <IntDir>\$(SolutionDir)..\\_out\\\$(Configuration)_\$(Platform)\\</IntDir>\n";
    print "    <TargetName>lwntest</TargetName>\n";
    print "  </PropertyGroup>\n";
}

for (@configs) {
    next if !m/NX32$/;
    my $dbgrel = m/^Debug/ ? "debug" : "release";
    print "  <PropertyGroup Condition=\"'\$(Configuration)|\$(Platform)'=='$_'\">\n";
    print "    <OutDir>..\\..\\..\\..\\..\\..\\out\\hos-t210ref-$dbgrel\\intermediates</OutDir>\n" if $dbgrel eq "debug";
    print "    <TargetName>lwntest</TargetName>\n";
    print "  </PropertyGroup>\n";
}

# The double loop here effectively overrides the sorting of @configs.
for my $loop (0..1) {
for (@configs) {
    next if m/NX32$/;
    next if !m/^Debug/ && $loop == 0;
    next if !m/^Release/ && $loop == 1;
    my $isdebug = m/^Debug/ ? "true" : "false";
    print "  <ItemDefinitionGroup Condition=\"'\$(Configuration)|\$(Platform)'=='$_'\">\n";
    print "    <ClCompile>\n";
    print "      <WarningLevel>Level3</WarningLevel>\n";
    if ($isdebug eq "true") {
        print "      <Optimization>Disabled</Optimization>\n";
    } else {
        print "      <Optimization>Full</Optimization>\n";
        print "      <FunctionLevelLinking>true</FunctionLevelLinking>\n";
        print "      <IntrinsicFunctions>true</IntrinsicFunctions>\n";
    }
    print "      <SDLCheck>true</SDLCheck>\n";

    # Write out any include directories.
    print "      <AdditionalIncludeDirectories>..\\tests\\lwn;..\\elw;..\\include;";
    print "..\\..\\..\\..\\drivers\\lwn\\public;..\\..\\..\\..\\drivers\\lwn\\interface;";
    print "..\\..\\common;..\\..\\samples\\common;";
    print "..\\shaderc\\libshaderc\\include;..\\shaderc\\libshaderc_util\\include;..\\shaderc\\third_party\\glslang;";
    print "..\\shaderc\\third_party\\spirv-tools;..\\shaderc\\third_party\\spirv-tools\\include;..\\shaderc\\third_party\\spirv-tools\\source;";
    print "..\\shaderc\\third_party\\spirv-tools\\external\\spirv-headers\\include;..\\shaderc\\third_party\\spirv-tools\\external\\spirv-headers\\include\\spirv\\unified1;";
    print "\%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>\n";

    print "      <PreprocessorDefinitions>_CRT_NONSTDC_NO_DEPRECATE;_CRT_SELWRE_NO_WARNINGS;GLSLC_LIB_DYNAMIC_LOADING;SPIRV_ENABLED;ENABLE_HLSL\%(PreprocessorDefinitions)</PreprocessorDefinitions>\n";
    print "      <DisableSpecificWarnings>4090;4838</DisableSpecificWarnings>\n";
    if ($isdebug eq "false") {
        print "      <InlineFunctionExpansion>AnySuitable</InlineFunctionExpansion>\n";
        print "      <FavorSizeOrSpeed>Speed</FavorSizeOrSpeed>\n";
        print "      <OmitFramePointers>true</OmitFramePointers>\n";
        print "      <EnableFiberSafeOptimizations>true</EnableFiberSafeOptimizations>\n";
    }
    print "      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>\n";
    print "      <MultiProcessorCompilation>true</MultiProcessorCompilation>\n";
    print "    </ClCompile>\n";
    print "    <Link>\n";
    print "      <GenerateDebugInformation>true</GenerateDebugInformation>\n";
    if ($isdebug eq "false") {
        print "      <EnableCOMDATFolding>true</EnableCOMDATFolding>\n";
        print "      <OptimizeReferences>true</OptimizeReferences>\n";
    }
    my $lwnlib = m/Win32$/ ? "lwn32.lib" : "lwn.lib";
    print "      <AdditionalDependencies>user32.lib;gdi32.lib;opengl32.lib;$lwnlib</AdditionalDependencies>\n";
    print "      <AdditionalLibraryDirectories>..\\..\\win32\\lib;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>\n";
    if ($isdebug eq "false") {
        print "      <LinkTimeCodeGeneration>UseLinkTimeCodeGeneration</LinkTimeCodeGeneration>\n";
    }
    print "    </Link>\n";
    print "  </ItemDefinitionGroup>\n";
}
}

# Build the list of source files.  Configure CheetAh-specific files to be
# excluded from Windows platform builds.
# Suppress C4005 warnings from specific shaderc files (Pp* files) that
# want to redefine preprocessor macros that we've already defined.
print "  <ItemGroup>\n";
for my $file (sort @sources) {
    if ($file =~ m/tegra_(main|utils).cpp$/) {
        print "    <ClCompile Include=\"..\\", $file, "\">\n";
        for (@configs) {
            next if !m/(Win32|x64)$/;
            print "      <ExcludedFromBuild Condition=\"'\$(Configuration)|\$(Platform)'=='$_'\">true</ExcludedFromBuild>\n";
        }
        print "    </ClCompile>\n";
        next;
    }
    if ($file =~ m/shaderc.*Pp.*\.cpp$/) {
        print "    <ClCompile Include=\"..\\", $file, "\">\n";
        print "      <DisableSpecificWarnings>4005</DisableSpecificWarnings>\n";
        print "    </ClCompile>\n";
        next;
    }

    print "    <ClCompile Include=\"..\\", $file, "\" />\n";
}

print "  </ItemGroup>\n";

# Build the list of header files.
print "  <ItemGroup>\n";
for my $file (sort @headers) {
    print "    <ClInclude Include=\"..\\", $file, "\" />\n";
}
print "  </ItemGroup>\n";
print "  <Import Project=\"\$(VCTargetsPath)\\Microsoft.Cpp.targets\" />\n";
print "  <ImportGroup Label=\"ExtensionTargets\">\n";
print "  </ImportGroup>\n";
print "</Project>\n";

exit(0);
