#!/usr/bin/elw perl

# mkvcprojflt.pl - A script to make the visual studio project filters containing all the sources.
# Inputs: ARG[0] - A file containing a space-separated list of sources and headers.  An individual
#                  entry of "--" will be used by the script to switch from processing source files
#                  to header files.

use strict;
use File::Basename;

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

# Build up a list of directories containing source files and header files.
my %srcdirs;
my %headerdirs;
for my $file (@sources) {
    my $dir = dirname($file);
    while ($dir ne ".") {
        $srcdirs{$dir} = 1;
        $dir = dirname($dir);
    }
}
for my $file (@headers) {
    my $dir = dirname($file);
    while ($dir ne ".") {
        $headerdirs{$dir} = 1;
        $dir = dirname($dir);
    }
}

########################################################################

print "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
print "<Project ToolsVersion=\"4.0\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">\n";

print "  <ItemGroup>\n";
print "    <Filter Include=\"Source Files\">\n";
print "      <UniqueIdentifier>{4FC737F1-C7A5-4376-A066-2A32D752A2FF}</UniqueIdentifier>\n";
print "      <Extensions>cpp;c;cc;cxx;def;odl;idl;hpj;bat;asm;asmx</Extensions>\n";
print "    </Filter>\n";
print "    <Filter Include=\"Header Files\">\n";
print "      <UniqueIdentifier>{93995380-89BD-4b04-88EB-625FBE52EBFB}</UniqueIdentifier>\n";
print "      <Extensions>h;hh;hpp;hxx;hm;inl;inc;xsd</Extensions>\n";
print "    </Filter>\n";

# Generate "folders" for each directory containing source or header files,
# assigning a "unique ID" for each.
my $id = 0;
for (sort keys %srcdirs) {
    print "    <Filter Include=\"Source Files\\$_\">\n";
    printf "      <UniqueIdentifier>{ed2441c8-144d-4582-9f53-423f1e%06x}</UniqueIdentifier>\n", $id++;
    print "    </Filter>\n";
}
for (sort keys %headerdirs) {
    print "    <Filter Include=\"Header Files\\$_\">\n";
    printf "      <UniqueIdentifier>{ed2441c8-144d-4582-9f53-423f1e%06x}</UniqueIdentifier>\n", $id++;
    print "    </Filter>\n";
}

print "  </ItemGroup>\n";

print "  <ItemGroup>\n";
for my $file (sort @sources) {
    print "    <ClCompile Include=\"..\\", $file, "\">\n";
    print "      <Filter>Source Files\\", dirname($file), "</Filter>\n";
    print "    </ClCompile>\n";
}
print "  </ItemGroup>\n";

print "  <ItemGroup>\n";
for my $file (sort @headers) {
    print "    <ClInclude Include=\"..\\", $file, "\">\n";
    print "      <Filter>Header Files\\", dirname($file), "</Filter>\n";
    print "    </ClInclude>\n";
}
print "  </ItemGroup>\n";

print "</Project>\n";

exit(0);
