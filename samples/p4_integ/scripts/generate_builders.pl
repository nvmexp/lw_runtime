#!/usr/bin/perl

use strict;

use XML::Simple;
use Getopt::Long;
use File::Basename;
use File::Compare qw (compare_text);
use File::Path;
use File::Copy;
use File::Find;
use File::Temp qw (tempfile);
use Cwd;
use Data::Dumper;

BEGIN {
    unshift @INC, dirname(__FILE__);
    unshift @INC, dirname(__FILE__) . "/dvs/";
}
use InfoHelper;
use dvsLwdaSamples;

my $info;
my $clean;
my $verify;
my $debug = 0;
my $quiet = 0;
my $p4_edit;
my $p4_cl;
my $builder_types;
my $generate_eris_vlct = 0;
my @supported_vs_years = ("2017", "2019", "2022");
my @os_list = ("Linux", "Windows", "QNX", "Android");
my @arch_list = ("x86_64", "ppc64le", "armv7l", "aarch64");
my $supported_sms;
my $available_dependencies;
my $target_os;
my $target_arch;
my $lwda_version;
my $generate_readme_md;
my %samples_by_os;

my $retval = GetOptions(
    "info=s" => \$info,
    "clean" => \$clean,
    "verify=s" => \$verify,
    "debug" => \$debug,
    "quiet" => \$quiet,
    "p4-edit" => \$p4_edit,
    "p4-cl=s" => \$p4_cl,
    "type=s" => \$builder_types,
    "generate-eris-vlct" => \$generate_eris_vlct,
    "supported-sms=s" => \$supported_sms,
    "available-dependencies=s" => \$available_dependencies,
    "target-arch=s" => \$target_arch,
    "target-os=s" => \$target_os,
    "lwca-version=s" => \$lwda_version,
    "generate-readme-md" => \$generate_readme_md,
    "help|h" => sub { &Usage(); exit 0; },
);

exit 1 if ($retval != 1);

my $generate_makefile;
my $name;
my $title;
my $description;
my $whitepaper;
my $concept_text;
my @sample_supported_sms;
my @driver_api;
my @runtime_api;
my $makefile;
my $vscode;
my $solution;
my $src_ext;
my $ptx;
my $fatbin;
my $cpp11;
my $cpp14;
my $glibc;
my $gccMilwersion;
my $driver;
my $openGL;
my $openGLES;
my $freeimage;
my $directX;
my $openMP;
my $MPI;
my $uvm;
my $pthreads;
my $x11;
my $egloutput;
my $EGL;
my $lwsci;
my $lwmedia;
my $vulkan;
my $screen;
my $libLWRTC;
my $lwsolver;
my $fallback_min_ptx;
my $conditional_exec;
my $linker = "LWCC";
my $platform_specific_added = 0;
my $sample_dir;
my $output_dir;
my $p4_root;
my $samples_vlct_text;
my @desktop_sms;
my @tegra_sms;
my @libs;
my @so_libs;
my @elw_vars;
my @sms;
my @cflags;
my @lwstom_objects;
my @source_ignore;
my @source_extra_compilations;
my @source_extra_headers;
my @source_extra_nones;
my @statics;
my @postbuild_events;
my @postbuild_eventsclean;
my @library_paths;
my @include_paths;
my @additional_preprocessor;
my @extra_cleans;
my @build_vs_years;
my @required_dependencies;
my @supported_platforms;
my @qnx_available_dependencies;
my @qatest_parameters;
my %vs_year_format_map = (
    "2017", "12",
    "2019", "12",
    "2022", "12",
);

END
{
    if ($p4_edit && $p4_cl && !$verify)
    {
        CleanupP4();
    }
}

HandleCommandLineOptions();

my @info_list;

if (!$generate_eris_vlct && !($target_os =~ m/windows/i))
{
    my $qnx_deps = `make -f qnx.mk DONOTHING 2> /dev/null | grep -i QNX_DEPS | head -1`;
    $qnx_deps =~ s/QNX_DEPS//;
    $qnx_deps =~ s/\s+//g;
    @qnx_available_dependencies = split ',', $qnx_deps;
    print "QNX deps=>@qnx_available_dependencies\n";
}

if ( -d $info)
{
    find(\&FindInfoXML, $info);
}
else
{
    push (@info_list, $info);
}

my $script_dir = dirname($0);
my $samples_root = "$script_dir/..";

if ($generate_eris_vlct)
{
    InitializeSamplesVLCT();
}

if ($generate_readme_md)
{
    # Initialize the samples by os list
    @samples_by_os{@os_list} = ();
}

for my $lwr_info (@info_list)
{
    print "Processing $lwr_info\n" unless $quiet;

    $sample_dir = dirname($lwr_info);
    $sample_dir =~ s|\\|/|g;
    $sample_dir =~ s/^\.\///;

    if ($verify || $p4_edit)
    {
        $verify = dirname($verify) if ( -f $verify );
        
        if (defined $ELW{'TMPDIR'})
        {
            $output_dir = $ELW{'TMPDIR'}."/samples_verify";
        }
        else
        {
            $output_dir = "/tmp/samples_verify";
        }
        
        mkpath("$output_dir");
    }
    else
    {
        $output_dir = $sample_dir;
    }

    if ($clean)
    {
        Clean();
        next;
    }

    next if ParseXMLInfo($lwr_info);

    if ($generate_readme_md)
    {
        GenerateReadmeMD();
    }

    my @available_dependencies = split(',', $available_dependencies);

    my $skip = 0;
    for my $required_dependency (@required_dependencies)
    {
        my $dependency_satisfied = 0;

        for my $dependency_available (@available_dependencies)
        {
            if (lc($dependency_available) eq lc($required_dependency))
            {
                $dependency_satisfied = 1;
                last;
            }
        }

        if ($target_os =~ m/windows/i || $target_os =~ m/mac/i)
        {
            # if the required dependency for the sample is X11 allowing it for Windows & Mac
            if ($required_dependency =~ m/x11/i)
            {
                $dependency_satisfied = 1;
            }
        }

        if ($target_os =~ m/mac/i)
        {
            if ($required_dependency =~ m/openmp/i && $pthreads)
            {
                $dependency_satisfied = 1;
            }
        }

        if (!$dependency_satisfied)
        {
            $skip = 1;
            print "Skipping $lwr_info: config does not have $required_dependency support.\n";
        }
    }
    next if $skip;

    if ($target_os ne "" && $target_arch ne "")
    {
        my $found_match = 0;

        for (@supported_platforms)
        {
            my $platform = @{$_}[0];
            my $arch = @{$_}[1];

            if ("$platform-$arch" eq "$target_os-$target_arch")
            {
                $found_match = 1;
                last;
            }

            if ("$platform" eq "$target_os")
            {
                $found_match = 1;
                last;
            }

            # This is to skip the samples which are supported only on aarch64 but not on ARMv7
            # Bug 200372491
            if ($platform =~ m/aarch64/i &&  $target_arch eq 1 &&  $target_os =~ m/arm/i)
            {
                $found_match = 1;
                last;
            }

            # TO-DO : move target-gen-os to be linux in v4l_dGPU.mk and qnx in qnx.mk
            # TO-DO : thereafter add corresponding tags in info.xml wherever required.
            if ($platform =~ m/linux/i &&  $arch =~ m/aarch64/i &&  $target_os =~ m/arm/i)
            {
                $found_match = 1;
                last;
            }

            if (($platform =~ m/windows/i) || $platform =~ m/mac/i)
            {
                if ($platform =~ m/$target_os/i)
                {
                    $found_match = 1;
                    last;
                }
            }
        }

        if (!$found_match)
        {
            print "Skipping $lwr_info: sample not supported on target platform.\n";
            next;
        }
    }

    if ($makefile && $generate_makefile)
    {
        GenerateMakefile();
    }

    if ($solution && not $makefile)
    {
        GenerateDummyMakefile();
    }

    if ($vscode)
    {
        GenerateVSCode($_);
    }

    if ($solution)
    {
        for (@build_vs_years)
        {
            GenerateSolution($_);
            GenerateProject($_);
        }
    }

    if ($generate_eris_vlct)
    {
        for (@qatest_parameters)
        {
            AddTestCase($_);
        }

        if (!@qatest_parameters)
        {
            AddTestCase();
        }
    }

    if ($verify)
    {
        my $ret;
        $verify =~ s|/$||;

        if ( -f $info )
        {
            $ret = Verify($verify);
        }
        else
        {
            my $verify_dir = dirname($lwr_info);
            $verify_dir = "./$verify_dir";
            $ret = Verify($verify_dir);
        }

        rmtree "$output_dir";
        exit $ret if $ret;
    }
    
    if ($p4_edit)
    {
        rmtree "$output_dir";
    }
}

if ($generate_readme_md)
{
    GenerateMasterReadmeMD();
}

if ($generate_eris_vlct)
{
    my $temp = "
<<RUN_TEST_CASE>>";
    $samples_vlct_text =~ s/$temp//;
    if ($generate_eris_vlct == 2)
    {
        open VLCTFILEOUT, ">samples_tests_ppc64le.vlct" or die "Couldn't open output VLCT file to write: $!";
        print VLCTFILEOUT $samples_vlct_text;
        close VLCTFILEOUT;
    }
    else
    {
        open VLCTFILEOUT, ">samples_tests.vlct" or die "Couldn't open output VLCT file to write: $!";
        print VLCTFILEOUT $samples_vlct_text;
        close VLCTFILEOUT;
    }
}

exit 0;

sub FindInfoXML
{
    if ($_ eq "info.xml")
    {
        push (@info_list, $File::Find::name);
    }
}

sub InitializeSamplesVLCT
{
    my $samples_vlct_in = "$script_dir/samples.vlct.in";
    $samples_vlct_in = "$script_dir/samples_ppc64le.vlct.in" if ($generate_eris_vlct == 2);
    local $/ = undef;

    open SAMPLES_VLCT_IN, "$samples_vlct_in" or die "Couldn't open template samples.vlct.in: $!";
    $samples_vlct_text = <SAMPLES_VLCT_IN>;
    close SAMPLES_VLCT_IN;
}

sub AddTestCase
{
    my $qatest_parameter = shift;

    my %sample_supported_platforms;
    $sample_supported_platforms{"Linux"}   = 0;
    $sample_supported_platforms{"Mac"}     = 0;
    $sample_supported_platforms{"Windows"} = 0;
# Disabling ARM as supported platform temporarily till we have Linux, Mac, Windows x86_64 working fine on Eris.
#    $sample_supported_platforms{"arm"}     = 0;

    for (@supported_platforms)
    {
        my $platform = @{$_}[0];
        my $arch = @{$_}[1];

        my %blacklisted_samples;
        my $black_list_file = dirname(__FILE__) . "/../../scripts/exclude_list.txt";

        if ($platform =~ m/linux/i)
        {
            if ($arch =~ m/x86_64/i)
            {
                build_samples_blacklist(
                    BLACKLIST_REF => \%blacklisted_samples, 
                    BLACK_LIST_FILE => $black_list_file,
                    MODE => "manual",
                    BUILD => "release",
                    OS => "Linux",
                    ARCH => "x86_64",
                );

                $sample_supported_platforms{"Linux"} = 1 if (!$blacklisted_samples{$sample_dir});
            }
        }
        elsif ($platform =~ m/mac/i)
        {
            if ($arch =~ m/x86_64/i)
            {
                build_samples_blacklist(
                    BLACKLIST_REF => \%blacklisted_samples, 
                    BLACK_LIST_FILE => $black_list_file,
                    MODE => "manual",
                    BUILD => "release",
                    OS => "Darwin",
                    ARCH => "x86_64",
                );

                $sample_supported_platforms{"Mac"} = 1 if (!$blacklisted_samples{$sample_dir});
            }
        }
        elsif ($platform =~ m/windows/i)
        {
            build_samples_blacklist(
                BLACKLIST_REF => \%blacklisted_samples, 
                BLACK_LIST_FILE => $black_list_file,
                MODE => "manual",
                BUILD => "release",
                OS => "win32",
                ARCH => "x86_64",
            );

            $sample_supported_platforms{"Windows"} = 1 if (!$blacklisted_samples{$sample_dir});
        }
        elsif ($platform =~ m/arm/i)
        {
            #$sample_supported_platforms{"arm"} = 1;
        }
    }

    my $filter_os;
    my $valid_os = 0;

    for my $key (keys(%sample_supported_platforms))
    {
        if (!$sample_supported_platforms{$key})
        {
            $filter_os .= "$key, ";
        }
        else
        {
            $valid_os = 1;
        }
    }
    $filter_os =~ s|, $||;

# If no valid OS is supported we do not add the test-case for it.
    if (!$valid_os)
    {
        return;
    }

    my $test_name;
    my $test_case;

    $test_name = "$name $qatest_parameter";
    $test_name =~ s/^\s+|\s+$//g;

    if (@elw_vars)
    {
        my @temp_elw_vars = @elw_vars;

        foreach (@temp_elw_vars)
        {
            $_ = "\"$_\"";
        }

        my $all_elw_vars = join(',', @temp_elw_vars);

        $test_case  = "              { \"elw\" : [$all_elw_vars], 
                \"exe\" : \"$test_name\"";
    }
    else
    {
        $test_case = "              { \"exe\" : \"$test_name\" ";
    }

    if ($filter_os)
    {
        $test_case .=  ",
                \"attributes\" : [
                  { \"filter\" : { \"os\" : \"$filter_os\" }},
                  \"result=waive\"
                  ] 
              },
<<RUN_TEST_CASE>>";
    }
    else
    {
        $test_case .= "},
<<RUN_TEST_CASE>>";
    }

    $samples_vlct_text =~ s/<<RUN_TEST_CASE>>/$test_case/;
}

sub GenerateMasterReadmeMD
{
    my $readme_md_in = "$script_dir/Master_README.md.in";
    #
    # The local delimitter is usually set to '\n'.
    # Set it to undef to pull the entire file as a string.
    #
    local $/ = undef;
    open READMEMDIN, "$readme_md_in" or die "Couldn't open template Master_README.md.in: $!";
    my $readme_md_text = <READMEMDIN>;
    close READMEMDIN;

    $readme_md_text = Tag($readme_md_text, "LWDA_VERSION", $lwda_version);

    # my $samples_list_tmp = "";
    # my $columns_creator = "---|---|---|---|";
    # my $i = 0;
    # my $no_of_cols = 4;
    # for (@{$samples_by_os{Linux}})
    # {
    #     $i++;
    #     $samples_list_tmp .= "**[$_](./Samples/$_)** | ";
    #     $samples_list_tmp .= "\n" if ($i % $no_of_cols == 0);
    #     $samples_list_tmp .= "$columns_creator\n" if ($i == $no_of_cols);
    # }
    # if ($i < 4)
    # {
    #     $samples_list_tmp .= "$columns_creator\n";
    # }

    # $readme_md_text = Tag($readme_md_text, "LINUX_SAMPLES", $samples_list_tmp);

    # $samples_list_tmp = "";
    # $i = 0;
    # for (@{$samples_by_os{Windows}})
    # {
    #     $i++;
    #     $samples_list_tmp .= "**[$_](./Samples/$_)** | ";
    #     $samples_list_tmp .= "\n" if ($i % $no_of_cols == 0);
    #     $samples_list_tmp .= "$columns_creator\n" if ($i == $no_of_cols);
    # }
    # if ($i < 4)
    # {
    #     $samples_list_tmp .= "\n$columns_creator\n";
    # }

    # $readme_md_text = Tag($readme_md_text, "WINDOWS_SAMPLES", $samples_list_tmp);

    print "Writing Master readme file : $samples_root/README.md \n";
    #
    # Remove any unused tags.
    #
    $readme_md_text = Replace($readme_md_text, "\n<<.*?>>\n", "\n");
    $readme_md_text = Replace($readme_md_text, " <<.*?>> ", " ");
    $readme_md_text = Replace($readme_md_text, "<<.*?>>", "");

    #
    # Smooth out the Makefile.
    # Any spaces or tabs before newlines are removed.
    # Two or more newlines in a row become two newlines.
    #
    $readme_md_text = Replace($readme_md_text, "( *|\t*)*\n", "\n");
    $readme_md_text = Replace($readme_md_text, "\n\n+", "\n\n");

    open READMEMDOUT, ">$samples_root/README.md" or die "Couldn't open output README.md to write: $!";
    print READMEMDOUT $readme_md_text;
    close READMEMDOUT;
}

sub GenerateReadmeMD
{
    my $readme_md_in = "$script_dir/README.md.in";

    #
    # The local delimitter is usually set to '\n'.
    # Set it to undef to pull the entire file as a string.
    #
    local $/ = undef;
    open READMEMDIN, "$readme_md_in" or die "Couldn't open template README.md.in: $!";
    my $readme_md_text = <READMEMDIN>;
    close READMEMDIN;

    my $sample_name = $name;
    $sample_name .= " - $title" if $title;

    $readme_md_text = Tag($readme_md_text, "SAMPLE_NAME", $sample_name);
    $readme_md_text = Tag($readme_md_text, "DESCRIPTION", $description);

    if ($whitepaper)
    {
        $whitepaper =~ s/\\/\//ig;
        my $whitepaper_text = "[whitepaper](./";
        $whitepaper_text .= $whitepaper;
        $whitepaper_text .= ")";

        $readme_md_text = Tag($readme_md_text, "REFERENCES", $whitepaper_text);
    }

    # This is to replace SM string for instance, "20 30 32 35 37 50 52 53" to be
    # "2.0 3.0 3.2 3.5 3.7 5.0 5.2 5.3"
    my @treated_sms_arr;
    foreach (@sample_supported_sms)
    {
        substr($_, -1, 0, ".");
        push(@treated_sms_arr, $_);
    }

    my $sms_string = "";
    foreach(@treated_sms_arr)
    {
        $sms_string .= "[SM $_ ](https://developer.lwpu.com/lwca-gpus)  ";
    }

    $readme_md_text = Tag($readme_md_text, "SUPPORTED_SMS", $sms_string);
    $readme_md_text = Tag($readme_md_text, "KEY_CONCEPTS", $concept_text);
    
    my $supported_os = "";
    my $supported_cpu_arch = "";

    my %os_exist;
    @os_exist{@os_list} = ();

    my %arch_exist;
    @arch_exist{@arch_list} = ();

    for (@supported_platforms)
    {
        my $platform = @{$_}[0];
        my $arch = @{$_}[1];

        if ($platform =~ m/linux/i)
        {
            $os_exist{Linux} = '1';
            if ($arch =~ m/x86_64/i)
            {
                $arch_exist{x86_64} = '1';
            }
            elsif ($arch =~ m/ppc64le/i)
            {
                $arch_exist{ppc64le} = '1';
            }
            elsif ($arch =~ m/arm/i)
            {
                $arch_exist{armv7l} = '1';
            }
            elsif ($arch =~ m/aarch64/i)
            {
                $arch_exist{aarch64} = '1';
            }
        }
        elsif ($platform =~ m/mac/i)
        {
            $os_exist{MacOSX} = '1';
            $arch_exist{x86_64} = '1';
        }
        elsif ($platform =~ m/windows/i)
        {
            $os_exist{Windows} = '1';
            $arch_exist{x86_64} .= '1';
        }
        elsif ($platform =~ m/arm/i)
        {
            # TODO - Update Info XML for embedded OSes
            $os_exist{Linux} = '1';
            $arch_exist{armv7l} = '1';
        }
        elsif ($platform =~ m/aarch64/i)
        {
            # TODO - Update Info XML for embedded OSes
            $os_exist{Linux} = '1';
            $arch_exist{aarch64} = '1';
        }
        elsif ($platform =~ m/qnx/i)
        {
            # TODO - Update Info XML for embedded OSes
            $os_exist{QNX} = '1';
            $arch_exist{aarch64} = '1';
        }
    }

    for (@os_list)
    {
        if ($os_exist{$_})
        {
            $supported_os .= "$_ " ;
        }
    }
    $supported_os =~ s/ /, /g;
    $supported_os =~ s/, $//;

    my $make_cmd_arch = "";
    for (@arch_list)
    {
        if ($arch_exist{$_})
        {
            $supported_cpu_arch .= "$_ " ;
            $make_cmd_arch .= "`\$ make TARGET_ARCH=$_` <br/> ";
        }
    }
    $supported_cpu_arch =~ s/ /, /g;
    $supported_cpu_arch =~ s/, $//;

    $readme_md_text = Tag($readme_md_text, "SUPPORTED_OS", $supported_os);
    $readme_md_text = Tag($readme_md_text, "SUPPORTED_ARCH", $supported_cpu_arch);

    my $lwda_apis = "";

    if (@driver_api or @runtime_api)
    {
        if (@driver_api)
        {
            $lwda_apis .= "### [LWCA Driver API](http://docs.lwpu.com/lwca/lwca-driver-api/index.html)\n";
            my $lwda_apis_tmp = "";
            for (@driver_api)
            {
                $lwda_apis_tmp .= "$_ ";
            }
            $lwda_apis_tmp =~ s/ /, /g;
            $lwda_apis_tmp =~ s/, $//;
            $lwda_apis .=  $lwda_apis_tmp;
        }

        if (@runtime_api)
        {
            $lwda_apis .= "\n\n### [LWCA Runtime API](http://docs.lwpu.com/lwca/lwca-runtime-api/index.html)\n";
            my $lwda_apis_tmp = "";
            for (@runtime_api)
            {
                $lwda_apis_tmp .= "$_ ";
            }
            $lwda_apis_tmp =~ s/ /, /g;
            $lwda_apis_tmp =~ s/, $//;
            $lwda_apis .=  $lwda_apis_tmp;
        }
    }

    $readme_md_text = Tag($readme_md_text, "LWDA_API", $lwda_apis);

    my $required_deps = "";

    if (@required_dependencies)
    {
        $required_deps .= "## Dependencies needed to build/run\n";
        my $tmp_deps_list = "";
        for (@required_dependencies)
        {
            my $lowercase_depname = lc $_;
            $tmp_deps_list .= "[$_](../../../README.md#$lowercase_depname) ";
        }
        $tmp_deps_list =~ s/ /, /g;
        $tmp_deps_list =~ s/, $//;
        $required_deps .= $tmp_deps_list;
    }

    $readme_md_text = Tag($readme_md_text, "DEPENDENCIES", $required_deps);

    $readme_md_text = Tag($readme_md_text, "LWDA_VERSION", $lwda_version);
    if (@required_dependencies)
    {
        my $dependency_blurb = "Make sure the dependencies mentioned in [Dependencies]() section above are installed.\n";
        $readme_md_text = Tag($readme_md_text, "DOES_DEPENDENCY_EXIST", $dependency_blurb);
    }

    my $build_details = "";

    if ($os_exist{Windows})
    {
        push (@{$samples_by_os{Windows}}, $name);

        $build_details .= <<END 

### Windows
The Windows samples are built using the Visual Studio IDE. Solution files (.sln) are provided for each supported version of Visual Studio, using the format:
```
*_vs<version>.sln - for Visual Studio <version>
```
Each individual sample has its own set of solution files in its directory:

To build/examine all the samples at once, the complete solution files should be used. To build/examine a single sample, the individual sample solution files should be used.
> **Note:** Some samples require that the Microsoft DirectX SDK (June 2010 or newer) be installed and that the VC++ directory paths are properly set up (**Tools > Options...**). Check DirectX Dependencies section for details."
END
    }
    if ($os_exist{Linux} || $os_exist{QNX})
    {
        push (@{$samples_by_os{Linux}}, $name);

        $build_details .= <<END

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
\$ cd <sample_dir>
\$ make
```
The samples makefiles can take advantage of certain options:
*  **TARGET_ARCH=<arch>** - cross-compile targeting a specific architecture. Allowed architectures are $supported_cpu_arch.
    By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.<br/>
$make_cmd_arch
    See [here](http://docs.lwpu.com/lwca/lwca-samples/index.html#cross-samples) for more details.
*   **dbg=1** - build with debug symbols
    ```
    \$ make dbg=1
    ```
*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`.
    ```
    \$ make SMS="50 60"
    ```

*  **HOST_COMPILER=<host_compiler>** - override the default g++ host compiler. See the [Linux Installation Guide](http://docs.lwpu.com/lwca/lwca-installation-guide-linux/index.html#system-requirements) for a list of supported host compilers.
```
    \$ make HOST_COMPILER=g++
```
END
    }

    $readme_md_text = Tag($readme_md_text, "BUILD_RUN_DETAILS", $build_details);



    $readme_md_text = Tag($readme_md_text, "BUILD_RUN_DETAILS", $build_details);



    print "Writing readme file: $name\n";
    #
    # Remove any unused tags.
    #
    $readme_md_text = Replace($readme_md_text, "\n<<.*?>>\n", "\n");
    $readme_md_text = Replace($readme_md_text, " <<.*?>> ", " ");
    $readme_md_text = Replace($readme_md_text, "<<.*?>>", "");

    #
    # Smooth out the Makefile.
    # Any spaces or tabs before newlines are removed.
    # Two or more newlines in a row become two newlines.
    #
    $readme_md_text = Replace($readme_md_text, "( *|\t*)*\n", "\n");
    $readme_md_text = Replace($readme_md_text, "\n\n+", "\n\n");

    open READMEMDOUT, ">$output_dir/README.md" or die "Couldn't open output README.md to write: $!";
    print READMEMDOUT $readme_md_text;
    close READMEMDOUT;
}

sub GenerateVSCode
{
    my @vscode_items =
    (
        { template => "$script_dir/vscode.cpp-props.in",  filename => "c_cpp_properties.json" },
        { template => "$script_dir/vscode.extensions.in", filename => "extensions.json" },
        { template => "$script_dir/vscode.launch.in",     filename => "launch.json" },
        { template => "$script_dir/vscode.tasks.in",      filename => "tasks.json" }
    );

    my $vscode_sample_dir = "$sample_dir/.vscode";

    my $vscode_output_dir = "$output_dir/.vscode";
    mkdir $vscode_output_dir unless -d $vscode_output_dir;

    for my $vscode_ref (@vscode_items)
    {
        my $template_in = $vscode_ref->{template};
        my $filename = $vscode_ref->{filename};

        local $/ = undef;
        open ITEMIN, "$template_in" or die "Couldn't open template $template_in: $!";
        my $template_text = <ITEMIN>;
        close ITEMIN;

        my $item_text = Tag($template_text, "PROJECT_NAME", $name);

        #
        # Remove any unused tags.
        #
        $item_text = Replace($item_text, "\n<<.*?>>\n", "\n");
        $item_text = Replace($item_text, "<<.*?>>", "");

        open ITEMOUT, ">$vscode_output_dir/$filename"  or die "Couldn't open output file $vscode_output_dir/$filename: $!";
        print ITEMOUT $item_text;
        close ITEMOUT;

        if ($p4_edit && !$verify && compare_text("$vscode_output_dir/$filename", "$vscode_sample_dir/$filename", \&verify_comparator))
        {
            HandleP4("$vscode_sample_dir/$filename");
            copy("$vscode_output_dir/$filename", "$vscode_sample_dir") or die "$filename copy failed.";
        }
    }
}

sub GenerateSolution
{
    my $solution_in = "$script_dir/solution.sln.in";
    my $vs_year = shift;
    
    local $/ = undef;
    open SLNFILEIN, "$solution_in" or die "Couldn't open template solution.sln.in: $!";
    my $solution_text = <SLNFILEIN>;
    close SLNFILEIN;
    
    my $sln_format_version = $vs_year_format_map{"$vs_year"};
    
    $solution_text = Tag($solution_text, "VS_YEAR", $vs_year);
    $solution_text = Tag($solution_text, "SLN_FORMAT_VERSION", $sln_format_version);
    $solution_text = Tag($solution_text, "PROJECT_NAME", $name);

    #
    # Remove any unused tags.
    #
    $solution_text = Replace($solution_text, "\n<<.*?>>\n", "\n");
    $solution_text = Replace($solution_text, "<<.*?>>", "");

    my $filename = "${name}_vs${vs_year}.sln";

    open SLNFILEOUT, ">$output_dir/$filename" or die "Couldn't open output solution file to write: $!";
    print SLNFILEOUT $solution_text;
    close SLNFILEOUT;

    if ($p4_edit && !$verify && compare_text("$output_dir/$filename", "$sample_dir/$filename", \&verify_comparator))
    {
        HandleP4("$sample_dir/$filename");
        copy("$output_dir/$filename", "$sample_dir") or die "$filename copy failed.";
    }
}

sub GenerateProject
{
    my $project_in = "$script_dir/project.vcxproj.in";
    my $vs_year = shift;
    
    local $/ = undef;
    open PROJFILEIN, "$project_in" or die "Couldn't open template project.vcxproj.in: $!";
    my $project_text = <PROJFILEIN>;
    close PROJFILEIN;
    
    $project_text = Tag($project_text, "PROJECT_NAME", $name);
    
    $project_text = AddLibs($project_text, "vcxproj");
    
    if ($openGL)
    {
        $project_text = AddOpenGL($project_text, "vcxproj");
    }
    
    if ($directX)
    {
        $project_text = AddDirectX($project_text);
    }

    if ($vulkan)
    {
		$project_text = AddVulkan($project_text, "vcxproj");
    }
    
    $project_text = AddGencode($project_text, "vcxproj");
    $project_text = AddSubsystemType($project_text, "vcxproj");

    for (@additional_preprocessor)
    {
        $project_text = Tag($project_text, "ADDITIONAL_PREPROCESSOR", "$_;");
    }
    my @source_nones;

    my ($compiler_files_ref, $header_files_ref) = GetSrcFiles($sample_dir);
    my @compiler_files = @$compiler_files_ref;
    my @header_files = @$header_files_ref;
    @compiler_files = sort @compiler_files;
    my $compile_string;
    for (@compiler_files)
    {
        my $lwstomCmd = 0;
        my $lwstomGencode;
        for my $lwrLwstom (@lwstom_objects)
        {
            if ($lwrLwstom->{SRC} =~ m/$_/)
            {
                $lwstomCmd = 1;
                
                if ($lwrLwstom->{GENCODE}->{sm})
                {
                    for my $lwrSM (@desktop_sms)
                    {
                        push(@{$lwrLwstom->{GENCODE}->{sm}}, $lwrSM);
                    }

                    $lwstomGencode = "<CodeGeneration>";
                    for my $lwstom_sm(@{$lwrLwstom->{GENCODE}->{sm}})
                    {
                        $lwstomGencode .= "compute_$lwstom_sm,sm_$lwstom_sm;";
                    }
                    $lwstomGencode .= "</CodeGeneration>";
                }
            }
        }

        if ($_ =~ m/\.lw$/ && $libLWRTC)
        {
            push (@source_nones, $_);
            next;
        }

        if ($_ =~ m/\.lw$/)
        {
            if ($lwstomCmd)
            {
                $compile_string .= <<END;
    <LwdaCompile Include="$_">
        $lwstomGencode
    </LwdaCompile>
END
            }
            else
            {
                $compile_string .= <<END;
    <LwdaCompile Include="$_" />
END
            }
        }
        else
        {
            $compile_string .= <<END;
    <ClCompile Include="$_" />
END
        }
    }
    
    if ($ptx)
    {
        $compile_string .= <<END;
    <LwdaCompile Include="$ptx.lw">
      <CompileOut Condition="'\$(Platform)'=='x64'">data/%(Filename)64.ptx</CompileOut>
      <LwccCompilation>ptx</LwccCompilation>
    </LwdaCompile>
END
    }

    if ($fatbin)
    {
        $compile_string .= <<END;
    <LwdaCompile Include="$fatbin.lw">
      <CompileOut Condition="'\$(Platform)'=='x64'">data/%(Filename)64.fatbin</CompileOut>
      <LwccCompilation>fatbin</LwccCompilation>
    </LwdaCompile>
END
    }

    for (@source_extra_compilations)
    {
        $compile_string .= <<END;
    <ClCompile Include="$_" />
END
    }
    
    $project_text = Tag($project_text, "COMPILE_FILES", $compile_string) if ($compile_string);
    
    @header_files = sort @header_files;
    my $header_string;
    for (@header_files)
    {
        if ($_ =~ m/\.lwh/)
        {
            push (@source_nones, $_);
            next;
        }
        
        $header_string .= <<END;
    <ClInclude Include="$_" />
END
    }
    
    for (@source_extra_headers)
    {
        $header_string .= <<END;
    <ClInclude Include="$_" />
END
    }
    
    $project_text = Tag($project_text, "INCLUDE_FILES", $header_string) if ($header_string);
    
    my $none_string;
    for ((@source_nones, @source_extra_nones))
    {
        $none_string .= <<END;
    <None Include="$_" />
END
    }
    
    $project_text = Tag($project_text, "NONE_FILES", $none_string) if ($none_string);
    
    $project_text = AddCFlags($project_text, "vcxproj");

    $project_text = AddOpenMP_VS($project_text, "vcxproj");

    $project_text = AddPostBuildEvents($project_text, "vcxproj");

    if ($MPI)
    {
        $project_text = AddMPI($project_text, "vcxproj");
    }
    
    $project_text = AddIncludePaths($project_text, "vcxproj");
    
    if ($vs_year eq "2017")
    {
        $project_text = Tag($project_text, "ADDITIONAL_GLOBAL_PROPERTIES", "    <PlatformToolset>v141</PlatformToolset>\n<<ADDITIONAL_GLOBAL_PROPERTIES>>");
        $project_text = AddWin10SDK($project_text, "vcxproj");
    }

    if ($vs_year eq "2019")
    {
        $project_text = Tag($project_text, "ADDITIONAL_GLOBAL_PROPERTIES", "    <PlatformToolset>v142</PlatformToolset>\n<<ADDITIONAL_GLOBAL_PROPERTIES>>");
        $project_text = Tag($project_text, "ADDITIONAL_GLOBAL_PROPERTIES", "    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>\n<<ADDITIONAL_GLOBAL_PROPERTIES>>");
    }

    if ($vs_year eq "2022")
    {
        $project_text = Tag($project_text, "ADDITIONAL_GLOBAL_PROPERTIES", "    <PlatformToolset>v143</PlatformToolset>\n<<ADDITIONAL_GLOBAL_PROPERTIES>>");
        $project_text = Tag($project_text, "ADDITIONAL_GLOBAL_PROPERTIES", "    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>\n<<ADDITIONAL_GLOBAL_PROPERTIES>>");
    }
    
    $project_text = Tag($project_text, "VS_YEAR", $vs_year);

    #
    # Remove any unused tags.
    #
    $project_text = Replace($project_text, "\n<<.*?>>\n", "\n");
    $project_text = Replace($project_text, "<<.*?>>", "");
    
    #
    # Smooth out the project file. There should be no lines that are just whitespace.
    #
    $project_text = Replace($project_text, "\n\s*\n", "\n");

    my $filename = "${name}_vs${vs_year}.vcxproj";

    open PROJFILEOUT, ">$output_dir/$filename" or die "Couldn't open output project file to write: $!";
    print PROJFILEOUT $project_text;
    close PROJFILEOUT;

    if ($p4_edit && !$verify && compare_text("$output_dir/$filename", "$sample_dir/$filename", \&verify_comparator))
    {
        HandleP4("$sample_dir/$filename");
        copy("$output_dir/$filename", "$sample_dir") or die "$filename copy failed.";
    }
}

sub AddPostBuildEvents
{
    my $text = shift;
    my $builder = shift;

    if ($builder eq "vcxproj")
    {
        my $commands_windows;
        for my $event_data (@postbuild_events)
        {
            if ($event_data->{OS} =~ m/windows/i)
            {
                $commands_windows .= "$event_data->{EVENT}\n";
            }
        }
        if ($commands_windows)
        {
            $text = Tag($text, "POSTBUILD_EVENTS", "    <PostBuildEvent>\n        <Command>$commands_windows        </Command>\n    </PostBuildEvent>\n<<POSTBUILD_EVENTS>>");
        }
        return $text;
    }
    elsif ($builder eq "makefile")
    {
        my $commands_linux;
        for my $event_data (@postbuild_events)
        {
            if (($event_data->{OS} =~ m/linux/i) || ($event_data->{OS} =~ m/mac/i))
            {
                $commands_linux .= "	\$(EXEC) $event_data->{EVENT}\n";
            }
        }

        if ($commands_linux)
        {
            $text = Tag($text, "POSTBUILD_EVENTS", $commands_linux);
        }

        my $clean_commands_linux;
        for my $event_data (@postbuild_eventsclean)
        {
            if (($event_data->{OS} =~ m/linux/i) || ($event_data->{OS} =~ m/mac/i))
            {
                $clean_commands_linux .= "	$event_data->{EVENT}\n";
            }
        }

        if ($clean_commands_linux)
        {
            $text = Tag($text, "POSTBUILD_EVENTS_CLEAN", $clean_commands_linux);
        }
        return $text;
    }
}

sub AddPlatformSpecific
{
    my $project_text = shift;
    
    if (!$platform_specific_added)
    {
        my $platform_specific_text = <<END;
  <ItemDefinitionGroup Condition="'\$(Platform)'=='Win32'">
    <Link>
      <<PLATFORM_SPECIFIC_LINKER_32>>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'\$(Platform)'=='x64'">
    <Link>
      <<PLATFORM_SPECIFIC_LINKER_64>>
    </Link>
  </ItemDefinitionGroup>
END

        $project_text = Tag($project_text, "PLATFORM_SPECIFIC", "$platform_specific_text");
    }
    
    return $project_text;
}

sub GenerateDummyMakefile
{
    my $makefile_in = "$script_dir/MakefileDummy.in";

    #
    # The local delimitter is usually set to '\n'.
    # Set it to undef to pull the entire file as a string.
    #
    local $/ = undef;
    open MAKEFILEIN, "$makefile_in" or die "Couldn't open template MakefileDummy.in: $!";
    my $makefile_text = <MAKEFILEIN>;
    close MAKEFILEIN;

    $makefile_text = Tag($makefile_text, "NAME", $name);

    #
    # Remove any unused tags.
    #
    $makefile_text = Replace($makefile_text, "\n<<.*?>>\n", "\n");
    $makefile_text = Replace($makefile_text, " <<.*?>> ", " ");
    $makefile_text = Replace($makefile_text, "<<.*?>>", "");

    #
    # Smooth out the Makefile.
    # Any spaces or tabs before newlines are removed.
    # Two or more newlines in a row become two newlines.
    #
    $makefile_text = Replace($makefile_text, "( *|\t*)*\n", "\n");
    $makefile_text = Replace($makefile_text, "\n\n+", "\n\n");

    open MAKEFILEOUT, ">$output_dir/Makefile" or die "Couldn't open output Makefile to write: $!";
    print MAKEFILEOUT $makefile_text;
    close MAKEFILEOUT;

}

sub GenerateMakefile
{
    my $makefile_in = "$script_dir/Makefile.in";

    #
    # The local delimitter is usually set to '\n'.
    # Set it to undef to pull the entire file as a string.
    #
    local $/ = undef;
    open MAKEFILEIN, "$makefile_in" or die "Couldn't open template Makefile.in: $!";
    my $makefile_text = <MAKEFILEIN>;
    close MAKEFILEIN;

    if (@qatest_parameters)
    {
        my $testcase_template = <<END;
	<<CONDITIONAL>><<ELW_VARS>>./<<NAME>> <<TEST_PARAMS>>
	<<TEST_CASE>>
END

        # add test case for each qatest param
        for (@qatest_parameters)
        {
            $makefile_text = Tag($makefile_text, "TEST_CASE", $testcase_template);
            $makefile_text = TagOne($makefile_text, "TEST_PARAMS", "$_")
        }
    }

    $makefile_text = AddCompilations($makefile_text);

    if ($driver)
    {
        if ($p4_edit && !$verify)
        {
            HandleP4("$sample_dir/findlwdalib.mk");
        }
        copy("$samples_root/common/findlwdalib.mk", "$sample_dir") or die "findlwdalib.mk copy failed.";
        $makefile_text = Tag($makefile_text, "DRIVER", "include ./findlwdalib.mk");
    }

    if ($openGL)
    {
        $makefile_text = AddOpenGL($makefile_text, "makefile");
    }

    if ($openGLES)
    {
        $makefile_text = AddOpenGLES($makefile_text, "makefile");
    }

    if ($vulkan)
    {
        $makefile_text = AddVulkan($makefile_text, "makefile");
    }
    if ($EGL)
    {
        $makefile_text = AddEGL($makefile_text, "makefile");
    }

    if ($lwsci)
    {
        $makefile_text = AddLwSci($makefile_text, "makefile");
    }

    if ($lwmedia)
    {
        $makefile_text = AddLwMedia($makefile_text, "makefile");
    }

    if ($freeimage)
    {
        $makefile_text = AddFreeimageCheck($makefile_text);
    }

    if ($openMP)
    {
        $makefile_text = AddOpenMP($makefile_text);
    }

    if ($MPI)
    {
        $makefile_text = AddMPI($makefile_text, "makefile");
    }

    if ($libLWRTC)
    {
        $makefile_text = AddLibLWRTC($makefile_text);
    }

    if ($lwsolver)
    {
        my $lwsolver_text = <<END;
ifeq (\$(TARGET_OS),linux)
ALL_CCFLAGS += -Xcompiler \\"-Wl,--no-as-needed\\"
endif
END

        $makefile_text = Tag($makefile_text, "LWSOLVER", $lwsolver_text);
    }

    if ($cpp11)
    {
        if ($gccMilwersion)
        {
            $makefile_text = AddCPPVersionCheck($makefile_text, $gccMilwersion, "C++11");
        }
        else
        {
            # else pass minimum GCC version required for C++11 as 4.7.0
            $makefile_text = AddCPPVersionCheck($makefile_text, "4.7.0", , "C++11");
        }
    }
    if ($cpp14)
    {
        # pass minimum GCC version required for C++14 as 5.0.0
        $makefile_text = AddCPPVersionCheck($makefile_text, "5.0.0", "C++14");
    }

    if ($glibc)
    {
        $makefile_text = AddGLibCVersionCheck($makefile_text, "2.33");
    }

    if ($ptx)
    {
        $makefile_text = AddPtx($makefile_text);
    }

    if ($fatbin)
    {
        $makefile_text = AddFatbin($makefile_text);
    }

    $makefile_text = AddPlatformChecks($makefile_text);

    if (@libs)
    {
        $makefile_text = AddLibs($makefile_text, "makefile");
    }

    if (@so_libs)
    {
        $makefile_text = AddSOLibs($makefile_text);
    }

    if (@elw_vars)
    {
        $makefile_text = AddElwVars($makefile_text);
    }

    if (@include_paths)
    {
        $makefile_text = AddIncludePaths($makefile_text, "makefile");
    }

    $makefile_text = AddCFlags($makefile_text, "makefile");

    if (@extra_cleans)
    {
        $makefile_text = AddExtraCleans($makefile_text);
    }

    $makefile_text = AddGencode($makefile_text, "makefile");

    $makefile_text = AddDebugVars($makefile_text);

    $makefile_text = AddLinks($makefile_text);

    $makefile_text = AddPostBuildEvents($makefile_text, "makefile");

    $makefile_text = Tag($makefile_text, "NAME", $name);

    if ($conditional_exec)
    {
        $makefile_text = Tag($makefile_text, "DEFINE_DEPS", "SAMPLE_ENABLED := 1");
        $makefile_text = Tag($makefile_text, "CONDITIONAL", "\$(EXEC) ");
        $makefile_text = InsertCheckDeps($makefile_text);
    }

    #
    # Remove any unused tags.
    #
    $makefile_text = Replace($makefile_text, "\n<<.*?>>\n", "\n");
    $makefile_text = Replace($makefile_text, " <<.*?>> ", " ");
    $makefile_text = Replace($makefile_text, "<<.*?>>", "");

    #
    # Smooth out the Makefile.
    # Any spaces or tabs before newlines are removed.
    # Two or more newlines in a row become two newlines.
    #
    $makefile_text = Replace($makefile_text, "( *|\t*)*\n", "\n");
    $makefile_text = Replace($makefile_text, "\n\n+", "\n\n");

    open MAKEFILEOUT, ">$output_dir/Makefile" or die "Couldn't open output Makefile to write: $!";
    print MAKEFILEOUT $makefile_text;
    close MAKEFILEOUT;

    if ($p4_edit && !$verify && compare_text("$output_dir/Makefile", "$sample_dir/Makefile", \&verify_comparator))
    {
        HandleP4("$sample_dir/Makefile");
        copy("$output_dir/Makefile", "$sample_dir") or die "Makefile copy failed.";
    }
}

sub CreateP4CL
{
    my $p4_desc = `p4 change -o`;
    $p4_desc =~ s|<enter description here>|Autogenerated CL from generate_builders.pl.|;
    my @desc_split = split(/(\r\n|\n\r|\n|\r)/, $p4_desc);
    undef $p4_desc;

    #
    # We don't want to bring in any files in the default CL
    #
    for my $line (@desc_split)
    {
        $p4_desc .= "$line\n";
        last if ($line =~ m/^Files:/);
    }

    (my $TMPCL, my $cl_filename) = tempfile();
    print {$TMPCL} $p4_desc;

    my $p4_cl_string;
    if ($cl_filename =~ m|:\\|)
    {
        $p4_cl_string = `type $cl_filename | p4 change -i`;
    }
    else
    {
        $p4_cl_string = `cat $cl_filename | p4 change -i`;
    }

    return (split(' ', $p4_cl_string))[1];
}

sub GetP4Root
{
    my $p4_client = `p4 client -o`;
    $p4_client =~ m/(\r|\n)Root:\s+(\S+)/;
    return $2;
}

sub HandleP4
{
    if (!$p4_cl)
    {
        $p4_cl = CreateP4CL();
    }

    if (!$p4_root)
    {
        $p4_root = GetP4Root();
    }

    my $file = shift;
    my $p4_file = getcwd;
    $p4_file =~ s|\\|/|g;
    $p4_file =~ s|.*sw|//sw|;
    $p4_file .= "/$file";

    `p4 edit -c $p4_cl $p4_file`;
}

sub CleanupP4
{
    `p4 revert -a -c $p4_cl`;

    my $files_opened = `p4 opened -c $p4_cl 2>&1`;

    if ($files_opened =~ m/not opened on this client/)
    {
        `p4 change -d $p4_cl`;
    }
}

sub AddExtraCleans
{
    my $text = shift;
    my $clean_text;

    for $clean (@extra_cleans)
    {
        $clean_text .= " $clean";
    }

    $text = Tag($text, "EXTRA_CLEANS", $clean_text);

    return $text;
}

sub AddDirectX
{
    my $text = shift;
    
    if ($directX eq "12")
    {
        $text = Tag($text, "ADDITIONAL_LIBS", "d3d$directX.lib;d3dcompiler.lib;dxgi.lib;<<ADDITIONAL_LIBS>>");
        $text = AddWin10SDK($text, "vcxproj");
    }
    else
    {
        $text = Tag($text, "ADDITIONAL_LIBS", "d3d$directX.lib;d3dcompiler.lib;<<ADDITIONAL_LIBS>>");
    }
    
    return $text;
}

sub AddEGL
{
# routine to add EGL libraries info to makefile
    my $text = shift;

    copy ("$samples_root/scripts/findegl.mk", "$output_dir") or die "findegl.mk copy failed";
    if ($p4_edit && !$verify && compare_text("$output_dir/findegl.mk", "$sample_dir/findegl.mk", \&verify_comparator))
    {
        HandleP4("$sample_dir/findegl.mk");
        copy ("$output_dir/findegl.mk", "$sample_dir/findegl.mk") or die "findegl.mk copy failed";
    }
    my $egl_text = <<END;
# Makefile include to help find EGL Libraries
include ./findegl.mk

# EGL specific libraries
ifneq (\$(TARGET_OS),darwin)
 LIBRARIES += -lEGL
endif
END

    $text = Tag($text, "EGL", $egl_text);

    $conditional_exec = 1;

    return $text;
}

sub AddOpenGLES
{
    my $text = shift;
    
    my $windowing_system = "";
    $x11 = 1 if (!$x11 && !$screen && !$egloutput); # default to X11 if not selected

    $windowing_system .= "-lX11 " if ($x11);
    $windowing_system .= "-ldrm " if ($egloutput);
    $windowing_system .= "-lscreen " if ($screen);

    copy ("$samples_root/scripts/findgleslib.mk", "$output_dir") or die "findgleslib.mk copy failed";
    if ($p4_edit && !$verify && compare_text("$output_dir/findgleslib.mk", "$sample_dir/findgleslib.mk", \&verify_comparator))
    {
        HandleP4("$sample_dir/findgleslib.mk");
        copy ("$output_dir/findgleslib.mk", "$sample_dir/findgleslib.mk") or die "findgleslib.mk copy failed";
    }
    my $gles_text = <<END;
# Makefile include to help find GLES Libraries
include ./findgleslib.mk

# OpenGLES specific libraries
ifneq (\$(TARGET_OS),darwin)
 LIBRARIES += \$(GLESLINK) -lGLESv2 -lEGL $windowing_system
endif
END

    $text = Tag($text, "OPENGLES", $gles_text);

    $conditional_exec = 1;

    return $text;
}

sub AddOpenGL
{
    my $text = shift;
    my $builder = shift;

    if ($builder eq "makefile")
    {
        copy ("$samples_root/scripts/findgllib.mk", "$output_dir") or die "findgllib.mk copy failed";
        if ($p4_edit && !$verify && compare_text("$output_dir/findgllib.mk", "$sample_dir/findgllib.mk", \&verify_comparator))
        {
            HandleP4("$sample_dir/findgllib.mk");
            copy ("$output_dir/findgllib.mk", "$sample_dir/findgllib.mk") or die "findgllib.mk copy failed";
        }
        my $gl_text = '# Makefile include to help find GL Libraries
include ./findgllib.mk

# OpenGL specific libraries 
ifeq ($(TARGET_OS),darwin)
 # Mac OSX specific libraries and paths to include
 LIBRARIES += -L/System/Library/Frameworks/OpenGL.framework/Libraries 
 LIBRARIES += -lGL -lGLU
 ALL_LDFLAGS += -Xlinker -framework -Xlinker GLUT
else
 LIBRARIES += $(GLLINK)
 LIBRARIES += -lGL -lGLU -lglut
endif';

        $text = Tag($text, "OPENGL", $gl_text);

        $conditional_exec = 1;
    }
    elsif ($builder eq "vcxproj")
    {
        $text = AddPlatformSpecific($text);
        $text = Tag($text, "ADDITIONAL_LIBS", "freeglut.lib;<<ADDITIONAL_LIBS>>");
        $text = Tag($text, "PLATFORM_SPECIFIC_LINKER_64", "<AdditionalDependencies>glew64.lib;%(AdditionalDependencies)</AdditionalDependencies>\n<<PLATFORM_SPECIFIC_LINKER_64>>");
        $text = Tag($text, "ADDITIONAL_LIB_DIRS", "../../../Common/lib/\$(PlatformName);<<ADDITIONAL_LIB_DIRS>>");
    }

    return $text;
}

sub AddVulkan
{
    my $text = shift;
    my $builder = shift;

    if ($builder eq "makefile")
    {
        copy ("$samples_root/scripts/findvulkan.mk", "$output_dir") or die "findvulkan.mk copy failed";
        if ($p4_edit && !$verify && compare_text("$output_dir/findvulkan.mk", "$sample_dir/findvulkan.mk", \&verify_comparator))
        {
            HandleP4("$sample_dir/findvulkan.mk");
            copy ("$output_dir/findvulkan.mk", "$sample_dir/findvulkan.mk") or die "findvulkan.mk copy failed";
        }
        my $gl_text = '# Makefile include to help find Vulkan SDK and dependencies
include ./findvulkan.mk

# Vulkan specific libraries
ifeq ($(TARGET_OS),linux)
 ifneq ($(TARGET_ARCH),$(HOST_ARCH))
  LIBRARIES += -L$(VULKAN_SDK_LIB) -lvulkan
  LIBRARIES += -lglfw
  INCLUDES  += -I$(VULKAN_HEADER)
 else
 LIBRARIES += -L$(VULKAN_SDK_LIB)
 LIBRARIES += `pkg-config --static --libs glfw3` -lvulkan
 INCLUDES  += `pkg-config --static --cflags glfw3` -I$(VULKAN_HEADER)
 endif
endif';

        $text = Tag($text, "VULKAN", $gl_text);

        $conditional_exec = 1;
    }
    elsif ($builder eq "vcxproj")
    {
        $text = AddPlatformSpecific($text);
        $text = Tag($text, "ADDITIONAL_LIBS", "vulkan-1.lib;glfw3dll.lib;<<ADDITIONAL_LIBS>>");
        $text = Tag($text, "ADDITIONAL_LIB_DIRS", "../../../common/lib/\$(PlatformName);\$(VULKAN_SDK)/Lib;<<ADDITIONAL_LIB_DIRS>>");
        $text = Tag($text, "ADDITIONAL_INCLUDE_DIRS", "\$(VULKAN_SDK)/include;<<ADDITIONAL_INCLUDE_DIRS>>");
    }

    return $text;
}


sub AddLibLWRTC
{
    my $text = shift;

    my $lwrtc_text = <<END;
# libLWRTC specific libraries
ifeq (\$(TARGET_OS),darwin)
 LDFLAGS += -L\$(LWDA_PATH)/lib -F/Library/Frameworks -framework LWCA
endif
END

    $text = Tag($text, "LIBLWRTC", $lwrtc_text);

    return $text;
}

sub AddLwSci
{
    my $text = shift;

    copy ("$samples_root/scripts/findlwsci.mk", "$output_dir") or die "findlwsci.mk copy failed";
    if ($p4_edit && !$verify && compare_text("$output_dir/findlwsci.mk", "$sample_dir/findlwsci.mk", \&verify_comparator))
    {
        HandleP4("$sample_dir/findlwsci.mk");
        copy ("$output_dir/findlwsci.mk", "$sample_dir/findlwsci.mk") or die "findlwsci.mk copy failed";
    }
    my $lwsci_text = <<END;
# Makefile include to help find LWSCI Libraries
include ./findlwsci.mk
END

    $text = Tag($text, "LWSCI", $lwsci_text);

    $conditional_exec = 1;

    return $text;
}

sub AddLwMedia
{
    my $text = shift;

    copy ("$samples_root/scripts/findlwmedia.mk", "$output_dir") or die "findlwmedia.mk copy failed";

    my $lwmedia_text = <<END;
# Makefile include to help find LwMedia Libraries
include ./findlwmedia.mk
END

    $text = Tag($text, "LWMEDIA", $lwmedia_text);

    $conditional_exec = 1;

    return $text;
}

sub InsertCheckDeps
{
    my $text = shift;

    my $check_deps = <<END;
check.deps:
ifeq (\$(SAMPLE_ENABLED),0)
	\@echo "Sample will be waived due to the above missing dependencies"
else
	\@echo "Sample is ready - all dependencies have been met"
endif
END

    $text = Tag($text, "CHECK_DEPS", $check_deps);
    $text = Tag($text, "DEPS_RULE", "check.deps");

    my $exec_def = <<END;
ifeq (\$(SAMPLE_ENABLED),0)
EXEC ?= \@echo "[@]"
endif
END

    $text = Tag($text, "DEFINE_EXEC", $exec_def);

    return $text;
}

sub AddPlatformChecks
{
    my $text = shift;

    my $linux64 = 0;
    my $mac64 = 0;
    my $windows = 0;
    my $arm = 0;
    my $aarch64 = 0;
    my $sbsa = 0;
    my $qnx = 0;

    for (@supported_platforms)
    {
        my $platform = @{$_}[0];
        my $arch = @{$_}[1];

        if ($platform =~ m/linux/i)
        {
            if ($arch =~ m/x86_64/i)
            {
                $linux64 = 1;
            }
        }
        elsif ($platform =~ m/mac/i)
        {
            if ($arch =~ m/x86_64/i)
            {
                $mac64 = 1;
            }
        }
        elsif ($platform =~ m/windows/i)
        {
            $windows = 1;
        }
        elsif ($platform =~ m/arm/i)
        {
            $arm = 1;
        }
        elsif ($platform =~ m/aarch64/i)
        {
            $aarch64 = 1;
        }
        elsif ($platform =~ m/sbsa/i)
        {
            $sbsa = 1;
        }
        elsif ($platform =~ m/qnx/i)
        {
            $qnx = 1;
            if ($arch =~ m/aarch64/i)
            {
                $aarch64 = 1;
            }
        }
    }

    if (!$linux64)
    {
        $conditional_exec = 1;

        my $no_linux_64 = <<END;
# This sample is not supported on Linux x86_64
ifeq (\$(TARGET_OS),linux)
  ifeq (\$(TARGET_ARCH),x86_64)
    \$(info >>> WARNING - <<NAME>> is not supported on Linux x86_64 - waiving sample <<<)
    SAMPLE_ENABLED := 0
  endif
endif

<<PLATFORM_SUPPORTED>>
END

        $text = Tag($text, "PLATFORM_SUPPORTED", $no_linux_64);
    }

    if (!$mac64)
    {
        $conditional_exec = 1;

        my $no_mac = <<END;
# This sample is not supported on Mac OSX
ifeq (\$(TARGET_OS),darwin)
  \$(info >>> WARNING - <<NAME>> is not supported on Mac OSX - waiving sample <<<)
  SAMPLE_ENABLED := 0
endif

<<PLATFORM_SUPPORTED>>
END

        $text = Tag($text, "PLATFORM_SUPPORTED", $no_mac);
    }

    if (!$windows)
    {
    }

    if (!$arm)
    {
        $conditional_exec = 1;

        my $no_arm = <<END;
# This sample is not supported on ARMv7
ifeq (\$(TARGET_ARCH),armv7l)
  \$(info >>> WARNING - <<NAME>> is not supported on ARMv7 - waiving sample <<<)
  SAMPLE_ENABLED := 0
endif

<<PLATFORM_SUPPORTED>>
END

        $text = Tag($text, "PLATFORM_SUPPORTED", $no_arm);
    }

    if (!$arm && !$aarch64)
    {
        $conditional_exec = 1;
        my $no_aarch64;
        if ($qnx)
        {
            $no_aarch64 = <<END;
# This sample is not supported on aarch64
ifeq (\$(TARGET_ARCH),aarch64)
  ifneq (\$(TARGET_OS),qnx)
      \$(info >>> WARNING - <<NAME>> is not supported on aarch64-\$(TARGET_OS) - waiving sample <<<)
      SAMPLE_ENABLED := 0
  endif
endif
<<PLATFORM_SUPPORTED>>
END
        }
        else
        {
            $no_aarch64 = <<END;
# This sample is not supported on aarch64
ifeq (\$(TARGET_ARCH),aarch64)
  \$(info >>> WARNING - <<NAME>> is not supported on aarch64 - waiving sample <<<)
  SAMPLE_ENABLED := 0
endif

<<PLATFORM_SUPPORTED>>
END
        }

        $text = Tag($text, "PLATFORM_SUPPORTED", $no_aarch64);
    }

    if (!$sbsa)
    {
        $conditional_exec = 1;

        my $no_sbsa = <<END;
# This sample is not supported on sbsa
ifeq (\$(TARGET_ARCH),sbsa)
  \$(info >>> WARNING - <<NAME>> is not supported on sbsa - waiving sample <<<)
  SAMPLE_ENABLED := 0
endif

<<PLATFORM_SUPPORTED>>
END

        $text = Tag($text, "PLATFORM_SUPPORTED", $no_sbsa);
    }

    if ($EGL)
    {
        $conditional_exec = 1;

        my $no_android = <<END;
# This sample is not supported on android
ifeq (\$(TARGET_OS),android)
  \$(info >>> WARNING - <<NAME>> is not supported on android - waiving sample <<<)
  SAMPLE_ENABLED := 0
endif

<<PLATFORM_SUPPORTED>>
END
        $text = Tag($text, "PLATFORM_SUPPORTED", $no_android);
    }

    if (scalar(@required_dependencies) && ($arm || $aarch64))
    {
        my $num_of_deps_needed = scalar(@required_dependencies);
        my $num_of_deps_available = 0;
        my $is_lwmedia = 0;

        # because LwMedia sample is a special case (only supporting cross builds) so we
        # allow it to build on QNX but not on ship in native qnx package.
        foreach (@required_dependencies)
        {
            if ($_ =~ m/LwMedia/i)
            {
                $is_lwmedia = 1;
            }
        }

        if (!$is_lwmedia)
        {
            foreach my $qnx_dep (@qnx_available_dependencies)
            {
                foreach (@required_dependencies)
                {
                    if ($qnx_dep =~ m/^$_$/i)
                    {
                        $num_of_deps_available++;
                    }
                }
            }
            if ($num_of_deps_available != $num_of_deps_needed)
            {
                $conditional_exec = 1;

                my $no_qnx = <<END;
# This sample is not supported on QNX
ifeq (\$(TARGET_OS),qnx)
  \$(info >>> WARNING - <<NAME>> is not supported on QNX - waiving sample <<<)
  SAMPLE_ENABLED := 0
endif

<<PLATFORM_SUPPORTED>>
END
                $text = Tag($text, "PLATFORM_SUPPORTED", $no_qnx);
            }
        }
    }

    return $text;
}

sub AddOpenMP
{
    my $text = shift;
    my $use_pthread = 'endif';
    if ($uvm)
    {
        $use_pthread = 'else
LIBRARIES += -lpthread
ALL_CCFLAGS += -DUSE_PTHREADS
endif
';
    }

    my $check_text = '
# Attempt to compile a minimal OpenMP application. If a.out exists, OpenMP is properly set up.
ifneq (,$(filter $(TARGET_OS),linux android))

ifneq (,$(filter $(TARGET_OS), android))
     LIBRARIES += -lomp
else
     LIBRARIES += -lgomp
endif

ALL_CCFLAGS += -Xcompiler -fopenmp
$(shell echo "#include <omp.h>" > test.c ; echo "int main() { omp_get_num_threads(); return 0; }" >> test.c ; $(HOST_COMPILER) -fopenmp test.c)
OPENMP ?= $(shell find a.out 2>/dev/null)

ifeq ($(OPENMP),)
      $(info -----------------------------------------------------------------------------------------------)
      $(info WARNING - OpenMP is unable to compile)
      $(info -----------------------------------------------------------------------------------------------)
      $(info   This LWCA Sample cannot be built if the OpenMP compiler is not set up correctly.)
      $(info   This will be a dry-run of the Makefile.)
      $(info   For more information on how to set up your environment to build and run this )
      $(info   sample, please refer the LWCA Samples documentation and release notes)
      $(info -----------------------------------------------------------------------------------------------)
      SAMPLE_ENABLED := 0
endif

$(shell rm a.out test.c 2>/dev/null)
'.$use_pthread.'<<OPENMP>>';

    $text = AddUbuntu($text);
    $text = Tag($text, "OPENMP", $check_text);

    $conditional_exec = 1;

    return $text;
}

sub AddOpenMP_VS
{
    my $text = shift;
    if ($openMP)
    {
        $text = Tag($text, "OPENMP_FLAG", "      <AdditionalCompilerOptions>/openmp</AdditionalCompilerOptions>\n");
    }
    else
    {
        $text = Tag($text, "OPENMP_FLAG", "");
    }

    $conditional_exec = 1;

    return $text;
}

sub AddUbuntu
{
    my $text = shift;

    $text = Tag($text, "IS_UBUNTU", 'UBUNTU = $(shell lsb_release -i -s 2>/dev/null | grep -i ubuntu)');

    return $text;
}

sub AddDebugVars
{
    my $text = shift;

    my $debug_vars;

    #
    # If MPI is used, we need to define the debug variables slightly different
    # in order to ensure MPICXX doesn't get the -G parameter.
    #
    if ($MPI)
    {
        $debug_vars = "LWCCFLAGS += -G
      CCFLAGS += -g";
    }
    else
    {
        $debug_vars = "LWCCFLAGS += -g -G";
    }

    $text = Tag($text, "DEBUG_VARS", $debug_vars);

    return $text;
}

sub AddFreeimageCheck
{
    my $text = shift;

    my $check_freeimage_text = <<END;
# Attempt to compile a minimal application linked against FreeImage. If a.out exists, FreeImage is properly set up.
\$(shell echo "#include \\"FreeImage.h\\"" > test.c; echo "int main() { return 0; }" >> test.c ; \$(LWCC) \$(ALL_CCFLAGS) \$(INCLUDES) \$(ALL_LDFLAGS) \$(LIBRARIES) -l freeimage test.c)
FREEIMAGE := \$(shell find a.out 2>/dev/null)
\$(shell rm a.out test.c 2>/dev/null)

ifeq ("\$(FREEIMAGE)","")
\$(info >>> WARNING - FreeImage is not set up correctly. Please ensure FreeImage is set up correctly. <<<)
SAMPLE_ENABLED := 0
endif
END

    $text = Tag($text, "DETECT_FREEIMAGE", $check_freeimage_text);

    $conditional_exec = 1;

    return $text;
}

sub AddMPI
{
   my $text = shift;
   my $builder = shift;

  if ($builder eq "makefile")
  {
    my $mpi_text = '# MPI check and binaries
MPICXX ?= $(shell which mpicxx 2>/dev/null)

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
      $(info -----------------------------------------------------------------------------------------------)
      $(info WARNING - Cross Compilation not supported for MPI Samples.)
      $(info -----------------------------------------------------------------------------------------------)
      $(info   Waiving the build )
      $(info   This will be a dry-run of the Makefile.)
      $(info   For more information on how to set up your environment to build and run this )
      $(info   sample, please refer the LWCA Samples documentation and release notes)
      $(info -----------------------------------------------------------------------------------------------)
      MPICXX=mpicxx
      SAMPLE_ENABLED := 0
endif

ifeq ($(MPICXX),)
      $(info -----------------------------------------------------------------------------------------------)
      $(info WARNING - No MPI compiler found.)
      $(info -----------------------------------------------------------------------------------------------)
      $(info   LWCA Sample "simpleMPI" cannot be built without an MPI Compiler.)
      $(info   This will be a dry-run of the Makefile.)
      $(info   For more information on how to set up your environment to build and run this )
      $(info   sample, please refer the LWCA Samples documentation and release notes)
      $(info -----------------------------------------------------------------------------------------------)
      MPICXX=mpicxx
      SAMPLE_ENABLED := 0
endif
';

    $text = Tag($text, "MPI_COMPILER", $mpi_text);

    my $mpi_var_defines = '
MPI_CCFLAGS :=
MPI_CCFLAGS += $(CCFLAGS)
MPI_CCFLAGS += $(EXTRA_CCFLAGS)

MPI_LDFLAGS :=
MPI_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
MPI_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))
';

    $text = Tag($text, "MPI_VAR_DEFINES", $mpi_var_defines);

    $linker = "MPICXX";
    $conditional_exec = 1;

    return $text;
  }
  elsif ($builder eq "vcxproj")
  {
    $text = Tag($text, "ADDITIONAL_LIB_DIRS", "\$(MSMPI_LIB64);<<ADDITIONAL_LIB_DIRS>>");

    my $library_text .= "msmpi.lib;";
    $text = Tag($text, "ADDITIONAL_LIBS", "$library_text<<ADDITIONAL_LIBS>>") if ($builder eq "vcxproj");
    $text = Tag($text, "ADDITIONAL_INCLUDE_DIRS", "\$(MSMPI_INC);<<ADDITIONAL_INCLUDE_DIRS>>");
    return $text;
  }
}

sub AddCPPVersionCheck
{
    my $text = shift;
    my $gccVersion = shift;
    my $cppStd = shift;
    my $gccDotVersion = $gccVersion;
    # keep only the version number remove alphabets & dot
    $gccVersion =~ s/\D//g;
    $gccVersion .= "00";
    $gccDotVersion =~ s/gcc//ig;

    my $cpp11_text = "#Detect if installed version of GCC supports required $cppStd
ifeq (\$(TARGET_OS),linux)
    empty :=
    space := \$(empty) \$(empty)
    GCCVERSIONSTRING := \$(shell expr `\$(HOST_COMPILER) -dumpversion`)
#Create version number without \".\"
    GCCVERSION := \$(shell expr `echo \$(GCCVERSIONSTRING)` | cut -f1 -d.)
    GCCVERSION += \$(shell expr `echo \$(GCCVERSIONSTRING)` | cut -f2 -d.)
    GCCVERSION += \$(shell expr `echo \$(GCCVERSIONSTRING)` | cut -f3 -d.)
# Make sure the version number has at least 3 decimals
    GCCVERSION += 00
# Remove spaces from the version number
    GCCVERSION := \$(subst \$(space),\$(empty),\$(GCCVERSION))
#\$(warning \$(GCCVERSION))

    IS_MIN_VERSION := \$(shell expr `echo \$(GCCVERSION)` \\>= $gccVersion)

    ifeq (\$(IS_MIN_VERSION), 1)
        \$(info >>> GCC Version is greater or equal to $gccDotVersion <<<)
    else 
        \$(info >>> Waiving build. Minimum GCC version required is $gccDotVersion<<<)
        SAMPLE_ENABLED := 0
    endif
endif

";

    $conditional_exec = 1;
    $text = Tag($text, "CPP11_CHECK", $cpp11_text);
    return $text;
}

sub AddGLibCVersionCheck
{
    my $text = shift;
    my $glibcVersion = shift;
    my $glibcDotVersion = $glibcVersion;
    # keep only the version number remove alphabets & dot
    $glibcVersion =~ s/\D//g;
    $glibcVersion .= "00";

    my $glibc_text = "#check glibc version is <= 2.33
ifeq (\$(TARGET_OS),linux)
    empty :=
    space := \$(empty) \$(empty)
    GLIBCVERSIONSTRING := \$(shell ldd --version | head -1 | rev | cut -f1 -d' ' | rev)
#Create version number without \".\"
    GLIBCVERSION := \$(shell expr `echo \$(GLIBCVERSIONSTRING)` | cut -f1 -d.)
    GLIBCVERSION += \$(shell expr `echo \$(GLIBCVERSIONSTRING)` | cut -f2 -d.)
# Make sure the version number has at least 3 decimals
    GLIBCVERSION += 00
# Remove spaces from the version number
    GLIBCVERSION := \$(subst \$(space),\$(empty),\$(GLIBCVERSION))
#\$(warning \$(GLIBCVERSION))

    IS_MIN_VERSION := \$(shell expr `echo \$(GLIBCVERSION)` \\<= $glibcVersion)

    ifeq (\$(IS_MIN_VERSION), 1)
        \$(info >>> GLIBC Version is less or equal to $glibcDotVersion <<<)
    else 
        \$(info >>> Waiving build. GLIBC > $glibcDotVersion is not supported<<<)
        SAMPLE_ENABLED := 0
    endif
endif

";

    $conditional_exec = 1;
    $text = Tag($text, "GLIBC_CHECK", $glibc_text);
    return $text;
}

sub GetSrcFiles
{
    my $dir = shift;
    my @compiler_files;
    my @header_files;

    local(*DIR);
    opendir(DIR, $dir) or die "Unable to open dir $dir: $!";

    while (my $file = readdir(DIR))
    {
        $file =~ s/^\.\///;
        my $filepath = "$dir/$file";
        $file = $filepath;
        $file = substr $file, (length $sample_dir) + 1;

        next if ($ptx && $file =~ m/$ptx.lw$/);
        next if ($fatbin && $file =~ m/$fatbin.lw$/);

        if (-d $filepath && $file !~ m/\.$/ && $file !~ m/win32$/i && $file !~ m/x64$/i)
        {
            my ($dir_srcs_compiler_ref, $dir_srcs_header_ref) = GetSrcFiles($filepath);
            my @dir_srcs_compiler = @$dir_srcs_compiler_ref;
            my @dir_srcs_header = @$dir_srcs_header_ref;

            for (@dir_srcs_compiler)
            {
                push(@compiler_files, "$_");
            }

            for (@dir_srcs_header)
            {
                push(@header_files, "$_");
            }

            next;
        }

        next if (! -f "$filepath");

        next if ($file !~ m/\.lw$/ && $file !~ m/\.cpp$/ && $file !~ m/\.c$/ && $file !~ m/\.h/ && $file !~ m/\.lwh/);

        my $ignored = 0;
        for (@source_ignore)
        {
            $ignored = 1 if ($_ eq $file);
        }
        next if ($ignored);

        print "Found File!: $file\n" if $debug;
        
        if ($file =~ m/\.h/ || $file =~ m/\.lwh/)
        {
            push(@header_files, $file);
        }
        else
        {
            push(@compiler_files, $file);
        }
    }

    return (\@compiler_files, \@header_files);
}

sub AddCompilations
{
    my $text = shift;

    my $compilations_text;
    my $units;
    my $targets;

    my ($compiler_files_ref, $header_files_ref) = GetSrcFiles($sample_dir);
    my @compiler_files = @$compiler_files_ref;
    my @header_files = @$header_files_ref;

    if (scalar @source_extra_compilations)
    {
        foreach (@source_extra_compilations){
            push(@compiler_files, $_);
        }
    }

    if (scalar @source_extra_headers)
    {
        foreach (@source_extra_headers){
            push(@header_files, $_);
        }
    }

    @compiler_files = sort @compiler_files;
    my @objects;

    for my $src (@compiler_files)
    {
        my $name = $src;
        $name =~ s/\.(lw|cpp)$//;
        $name =~ s/.*\///;

        my $is_so;
        for my $so (@so_libs)
        {
            if ($so eq $name)
            {
                $is_so = 1;
                last;
            }
        }
        next if $is_so;

        my $compiler = "LWCC";
        my $target = "$name.o";

        if ($MPI && $src =~ m/\.cpp$/)
        {
            $compiler = "MPICXX";
            $target = "${name}_mpi.o";
        }

        if (!$libLWRTC || ($libLWRTC && !($src =~ m/\.lw$/)))
        {
            push(@objects, {
                    SRC => $src,
                    TARGET => $target,
                    COMPILER => $compiler,
                });
        }
    }

    for my $object (@objects)
    {
        my $is_so;
        for my $so (@so_libs)
        {
            if ($so eq $name)
            {
                $is_so = 1;
                last;
            }
        }
        next if $is_so;

        my $compiler = "LWCC";
        $compiler = $object->{COMPILER} if ($object->{COMPILER});

        $compilations_text .= "
$object->{TARGET}:$object->{SRC}";

        my $cmd;
        for (@lwstom_objects)
        {
            if ($_->{SRC} eq $object->{SRC})
            {
                $cmd = "\$($compiler) \$(INCLUDES) \$(ALL_CCFLAGS)";
                
                for my $lwstom_sm(@{$_->{GENCODE}->{sm}})
                {
                    $cmd .= " -gencode arch=compute_$lwstom_sm,code=sm_$lwstom_sm";
                }
                
                $cmd .= " \$(GENCODE_FLAGS) -o \$@ -c \$<";
            }
        }

        if (!$cmd)
        {
            if ($compiler eq "LWCC")
            {
                $cmd = "\$($compiler) \$(INCLUDES) \$(ALL_CCFLAGS) \$(GENCODE_FLAGS) -o \$@ -c \$<";
            }
            elsif ($compiler eq "MPICXX")
            {
                $cmd = "\$($compiler) \$(INCLUDES) \$(MPI_CCFLAGS) -o \$@ -c \$<";
            }
            else
            {
                $cmd = "\$($compiler) \$(INCLUDES) \$(ALL_CCFLAGS) -o \$@ -c \$<";
            }
        }

        $compilations_text .= "
	<<CONDITIONAL>>$cmd
";

        $units .= "$object->{TARGET} ";
        $targets .= "$object->{TARGET} ";
    }

    for my $static (@statics)
    {
        $compilations_text .= "
$static->{TARGET}:$static->{SRC}";

        $compilations_text .= "
	<<CONDITIONAL>>\$(LWCC) \$(ALL_LDFLAGS) -lib -o \$@ \$< \$(LIBRARIES) 
";

        my $objectStatic = $static->{TARGET};
        $objectStatic =~ s/\.a/\.o/;
        
        if ($targets =~ m/$objectStatic /)
        {
            $targets =~ s/$objectStatic /$static->{TARGET} /;
        }
        else
        {
            $targets .= "$static->{TARGET} ";
        }

        $units .= "$static->{TARGET} ";
    }

    if (@statics)
    {
        my $qnx_arbin = "LWCCFLAGS += -arbin \$(QNX_HOST)/usr/bin/aarch64-unknown-nto-qnx7.1.0-ar";
        $text = Tag($text, "QNX_ARBIN_STATIC_LIB", $qnx_arbin);
    }

    for my $so (@so_libs)
    {
        $compilations_text .= "
$so.o:$so.cpp
	<<CONDITIONAL>>\$(LWCC) \$(INCLUDES) \$(ALL_CCFLAGS) --compiler-options '-fPIC' \$(GENCODE_FLAGS) -o \$@ -c \$<

$so.so.1:$so.o
	<<CONDITIONAL>>\$(LWCC) -shared \$(ALL_LDFLAGS) \$(GENCODE_FLAGS) -o \$@ \$+ \$(LIBRARIES)
	<<CONDITIONAL>>mkdir -p ../../../bin/\$(TARGET_ARCH)/\$(TARGET_OS)/\$(BUILD_TYPE)
	<<CONDITIONAL>>cp \$@ ../../../bin/\$(TARGET_ARCH)/\$(TARGET_OS)/\$(BUILD_TYPE)
";

        $units .= "$so.o $so.so.1 ";
    }

    $text = Tag($text, "COMPILATIONS", $compilations_text);
    $text = Tag($text, "TARGETS", $targets);
    $text = Tag($text, "CLEANUP", $units);

    return $text;
}

sub AddLinks
{
    my $text = shift;

    my $link_text = "<<CONDITIONAL>>\$($linker)";

    if ($linker eq "LWCC")
    {
        $link_text .= ' $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)';
    }
    elsif ($linker eq "MPICXX")
    {
        $link_text .= ' $(MPI_LDFLAGS) -o $@ $+ $(LIBRARIES)';
    }
    else
    {
        $link_text .= ' $(LDFLAGS) -o $@ $+ $(LIBRARIES)';
    }

    $text = Tag($text, "LINK", $link_text);

    return $text;
}

sub AddCFlags
{
    my $text = shift;
    my $builder = shift;
    my $cflags_text;

    if ($builder eq "makefile")
    {
        $cflags_text = "ALL_CCFLAGS +=";
        my $parallel_lwcc = 1;

        for my $flag (@cflags)
        {
            if ($flag =~ m/--threads 1/)
            {
                $parallel_lwcc = 0;
            }
            $cflags_text .= " $flag";
        }
        if ($parallel_lwcc)
        {
            $cflags_text .= " --threads 0";
        }

        # add c++11 std for all samples
        if (!($cflags_text =~ m/std=c++.*/)) 
        {
            $cflags_text .= " --std=c++11";
        }

        for my $lib (@libs)
        {
            if (ref($lib) eq 'HASH' && $lib->{content} eq "GLEW" || $lib eq "GLEW")
            {
                $cflags_text .= '
# Use -no-pie if it\'s supported by the host compiler.
$(shell echo "int main() { return 0; }" > test.c; $(HOST_COMPILER) -no-pie test.c 2>/dev/null)
NOPIE ?= $(shell find a.out 2>/dev/null)

ifneq ($(NOPIE),)
    ALL_CCFLAGS += -Xcompiler -no-pie
    ALL_LDFLAGS += -Xcompiler -no-pie
endif

$(shell rm -f a.out test.c 2>/dev/null)
';
            }
        }

        $text = Tag($text, "CCFLAGS", $cflags_text);
    }
    elsif ($builder eq "vcxproj")
    {
        my $parallel_lwcc = 1;
        for my $flag (@cflags)
        {
            if ($flag =~ m/-dc/)
            {
                $text = Tag($text, "LWCC_FLAGS", "      <GenerateRelocatableDeviceCode>true</GenerateRelocatableDeviceCode>\n<<LWCC_FLAGS>>");
            }
            elsif ($flag =~ m/-ewp/)
            {
                $text = Tag($text, "LWCC_FLAGS", "      <ExtensibleWholeProgramCompilation>true</ExtensibleWholeProgramCompilation>\n<<LWCC_FLAGS>>");
            }
            elsif ($flag =~ m/--threads 1/) # for samples like segmentationTreeThrust which face memory issue with multi-threaded compilation
            {
                $text = Tag($text, "ADDITIONAL_OPTIONS", " --threads 1 <<ADDITIONAL_OPTIONS>>");
                $parallel_lwcc = 0;
            }
        }
        if ($parallel_lwcc)
        {
            $text = Tag($text, "ADDITIONAL_OPTIONS", " --threads 0 <<ADDITIONAL_OPTIONS>>");
        }
    }
    
    return $text;
}

sub AddLibraryHelper
{
    my $text = shift;

    my $library_helper = 'LIBSIZE := 
ifneq ($(TARGET_OS),darwin)
ifeq ($(TARGET_SIZE),64)
LIBSIZE := 64
endif
endif';

    $text = Tag($text, "LIBRARY_HELPERS", $library_helper);

    return $text;
}

sub AddElwVars
{
    my $text = shift;

    my $elw_var_text;
    for my $var (@elw_vars)
    {
        $elw_var_text .= "$var ";
    }

    $text = Tag($text, "ELW_VARS", $elw_var_text);

    return $text;
}

sub AddSOLibs
{
    my $text = shift;

    my $so_text;
    for my $so (@so_libs)
    {
        $so_text .= "$so.so.1 ";
    }
    $text = Tag($text, "SO_NAME", $so_text);

    return $text;
}

sub AddLibs
{
    my $text = shift;
    my $builder = shift;
    my $library_text;
    my $libsize;
    my $lwda_drv = 0;
    my $regular_lib_needed = 0;

    for (@library_paths)
    {
        my $path;
        if (not ref($_))
        {
            $path = $_;
        }
        elsif (ref($_) eq 'HASH')
        {
            $path = $_->{content};
            my $pathOS = $_->{os};
            next if ($pathOS !~ m/linux/i && $builder eq "makefile");
            next if ($pathOS !~ m/windows/i && $builder eq "vcxproj");
        }

        if ($builder eq "makefile")
        {
            $library_text = "LIBRARIES +=" if (!$library_text);
            $library_text .= " -L$path";
        }
        elsif ($builder eq "vcxproj")
        {
            $text = Tag($text, "ADDITIONAL_LIB_DIRS", "$path;<<ADDITIONAL_LIB_DIRS>>");
        }
    }

    if ($library_text =~ m/LIBSIZE/)
    {
        $text = AddLibraryHelper($text);
    }

    for (@libs)
    {
        my $libname;
        if (not ref($_))
        {
            $libname = $_;
        }
        elsif (ref($_) eq 'HASH')
        {
            my $libOS = $_->{os};
            next if ($libOS !~ m/linux/i && $builder eq "makefile");
            next if ($libOS !~ m/windows/i && $builder eq "vcxproj");
            
            my $libArch = $_->{arch};
            $libname = $_->{content};
        
            if ($libOS =~ m/windows/i)
            {
                if ($libArch eq "32")
                {
                    $text = AddPlatformSpecific($text);
                    $text = Tag($text, "PLATFORM_SPECIFIC_LINKER_32", "<AdditionalDependencies>$libname.lib;%(AdditionalDependencies)</AdditionalDependencies>\n<<PLATFORM_SPECIFIC_LINKER_32>>");
                    next;
                }
                elsif ($libArch eq "64")
                {
                    $text = AddPlatformSpecific($text);
                    $text = Tag($text, "PLATFORM_SPECIFIC_LINKER_64", "<AdditionalDependencies>$libname.lib;%(AdditionalDependencies)</AdditionalDependencies>\n<<PLATFORM_SPECIFIC_LINKER_64>>");
                    next;
                }
            }
        }
        
        if ($libname !~ m/^lwca$/i)
        {
            if ($builder eq "makefile")
            {
                $library_text = "LIBRARIES +=" if (!$library_text);
                $library_text .= " -l$libname";
            }
            elsif ($builder eq "vcxproj")
            {
                if (!$regular_lib_needed)
                {
                    $text = Tag($text, "ADDITIONAL_INCLUDE_DIRS", "\$(LwdaToolkitIncludeDir);<<ADDITIONAL_INCLUDE_DIRS>>");
                    $regular_lib_needed = 1;
                }
                
                $library_text .= "$libname.lib;";
            }
        }
        else
        {
            if ($builder eq "makefile")
            {
                $text = AddLwdaDriver($text);
            }
            elsif ($builder eq "vcxproj")
            {
                $library_text .= "$libname.lib;";
                $lwda_drv = 1;
            }
        }
    }
    
    $library_text .= "lwdart_static.lib;" if (!$lwda_drv && $builder eq "vcxproj");

    $text = Tag($text, "ADDITIONAL_LIBS", "$library_text<<ADDITIONAL_LIBS>>") if ($builder eq "vcxproj");
    $text = Tag($text, "LIBRARIES", $library_text) if ($builder eq "makefile");

    return $text;
}

sub AddLwdaDriver
{
    my $text = shift;
    my $liblwda_template = '
ifeq ($(TARGET_OS),darwin)
  ALL_LDFLAGS += -Xcompiler -F/Library/Frameworks -Xlinker -framework -Xlinker LWCA
else
  ifeq ($(TARGET_ARCH),x86_64)
    LWDA_SEARCH_PATH ?= $(LWDA_PATH)/lib64/stubs
    LWDA_SEARCH_PATH += $(LWDA_PATH)/lib/stubs
    LWDA_SEARCH_PATH += $(LWDA_PATH)/targets/x86_64-linux/lib/stubs
  endif

  ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
    LWDA_SEARCH_PATH ?= $(LWDA_PATH)/targets/armv7-linux-gnueabihf/lib/stubs
  endif

  ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
    LWDA_SEARCH_PATH ?= $(LWDA_PATH)/targets/aarch64-linux/lib/stubs
  endif

  ifeq ($(TARGET_ARCH)-$(TARGET_OS),sbsa-linux)
    LWDA_SEARCH_PATH ?= $(LWDA_PATH)/targets/sbsa-linux/lib/stubs
  endif

  ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-android)
    LWDA_SEARCH_PATH ?= $(LWDA_PATH)/targets/armv7-linux-androideabi/lib/stubs
  endif

  ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-android)
    LWDA_SEARCH_PATH ?= $(LWDA_PATH)/targets/aarch64-linux-androideabi/lib/stubs
  endif

  ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-qnx)
    LWDA_SEARCH_PATH ?= $(LWDA_PATH)/targets/ARMv7-linux-QNX/lib/stubs
  endif

  ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
    LWDA_SEARCH_PATH ?= $(LWDA_PATH)/targets/aarch64-qnx/lib/stubs
    ifdef TARGET_OVERRIDE
        LWDA_SEARCH_PATH := $(LWDA_PATH)/targets/$(TARGET_OVERRIDE)/lib/stubs
    endif
  endif
  
  ifeq ($(TARGET_ARCH),ppc64le)
    LWDA_SEARCH_PATH ?= $(LWDA_PATH)/targets/ppc64le-linux/lib/stubs
  endif

  ifeq ($(HOST_ARCH),ppc64le)
    LWDA_SEARCH_PATH += $(LWDA_PATH)/lib64/stubs
  endif

  LWDALIB ?= $(shell find -L $(LWDA_SEARCH_PATH) -maxdepth 1 -name liblwda.so 2> /dev/null)
  ifeq ("$(LWDALIB)","")
    $(info >>> WARNING - liblwda.so not found, LWCA Driver is not installed.  Please re-install the driver. <<<)
    SAMPLE_ENABLED := 0
  else
    LWDALIB := $(shell echo $(LWDALIB) | sed "s/ .*//" | sed "s/\/liblwda.so//" ) 
    LIBRARIES += -L$(LWDALIB) -llwda
  endif
endif
';

    $text = AddUbuntu($text);
    $text = Tag($text, "LIBLWDA_TEMPLATE", $liblwda_template);
    $conditional_exec = 1;

    return $text;
}

sub AddIncludePaths
{
    my $text = shift;
    my $builder = shift;
    my $include_text;
    if ($builder eq "makefile")
    {
        $include_text = "INCLUDES +=";
    }
    for (@include_paths)
    {
        my $path;
        if (not ref($_))
        {
            $path = $_;
        }
        elsif (ref($_) eq 'HASH')
        {
            $path = $_->{content};
            my $pathOS = $_->{os};
            next if ($pathOS !~ m/linux/i && $builder eq "makefile");
            next if ($pathOS !~ m/windows/i && $builder eq "vcxproj");
        }
        if ($builder eq "makefile")
        {
            $include_text .= " -I$path";
        }
        elsif ($builder eq "vcxproj")
        {
            $include_text .= "$path;";
        }
    }
    if ($builder eq "makefile")
    {
        $text = Tag($text, "INCLUDES_PATHS", $include_text);
    }
    elsif ($builder eq "vcxproj")
    {
        $text = Tag($text, "ADDITIONAL_INCLUDE_DIRS", $include_text);
    }
    return $text;
}

sub AddPtx
{
    my $text = shift;
    
    my $ptx_target = "\$(PTX_FILE): ${ptx}.lw";
    $ptx_target .= '
	<<CONDITIONAL>>$(LWCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -ptx $<
	<<CONDITIONAL>>mkdir -p data
	<<CONDITIONAL>>cp -f $@ ./data
	<<CONDITIONAL>>mkdir -p ../../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	<<CONDITIONAL>>cp -f $@ ../../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)';

    $text = Tag($text, "PTX", "PTX_FILE := ${ptx}\${TARGET_SIZE}.ptx");
    $text = Tag($text, "PTX_FILE", '$(PTX_FILE)');
    $text = Tag($text, "PTX_TARGET", $ptx_target);
    $text = Tag($text, "PTX_CLEAN_LOCAL", 'data/$(PTX_FILE) $(PTX_FILE)');
    $text = Tag($text, "PTX_CLEAN_BIN", 'rm -rf ../../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)/$(PTX_FILE)');

    return $text;
}

sub AddFatbin
{
    my $text = shift;
    
    my $fatbin_target = "\$(FATBIN_FILE): ${fatbin}.lw";
    $fatbin_target .= '
	<<CONDITIONAL>>$(LWCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -fatbin $<
	<<CONDITIONAL>>mkdir -p data
	<<CONDITIONAL>>cp -f $@ ./data
	<<CONDITIONAL>>mkdir -p ../../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	<<CONDITIONAL>>cp -f $@ ../../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)';

    $text = Tag($text, "FATBIN", "FATBIN_FILE := ${fatbin}\${TARGET_SIZE}.fatbin");
    $text = Tag($text, "FATBIN_FILE", '$(FATBIN_FILE)');
    $text = Tag($text, "FATBIN_TARGET", $fatbin_target);
    $text = Tag($text, "FATBIN_CLEAN_LOCAL", 'data/$(FATBIN_FILE) $(FATBIN_FILE)');
    $text = Tag($text, "FATBIN_CLEAN_BIN", 'rm -rf ../../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)/$(FATBIN_FILE)');

    return $text;
}

sub AddSubsystemType
{
    my $text = shift;
    my $builder = shift;
    my $subsystem_text;

    if ($builder eq "vcxproj")
    {
        if ($directX eq "12")
        {
            $subsystem_text = "Windows";
        }
        else
        {
            $subsystem_text = "Console";
        }
        $text = Tag($text, "SUBSYSTEM_TYPE", $subsystem_text) if ($subsystem_text);
    }
}

sub AddWin10SDK
{
    my $text = shift;
    my $builder = shift;
    my $win10_sdk_text;

    if ($builder eq "vcxproj")
    {
        $win10_sdk_text = <<END;
  <PropertyGroup Condition="'\$(WindowsTargetPlatformVersion)'==''">
    <LatestTargetPlatformVersion>\$([Microsoft.Build.Utilities.ToolLocationHelper]::GetLatestSDKTargetPlatformVersion('Windows', '10.0'))</LatestTargetPlatformVersion>
    <WindowsTargetPlatformVersion Condition="'\$(WindowsTargetPlatformVersion)' == ''">\$(LatestTargetPlatformVersion)</WindowsTargetPlatformVersion>
    <TargetPlatformVersion>\$(WindowsTargetPlatformVersion)</TargetPlatformVersion>
  </PropertyGroup>
END

        $text = Tag($text, "USE_WIN10_SDK", $win10_sdk_text) if ($win10_sdk_text);
    }
    else
    {
        $text = Tag($text, "USE_WIN10_SDK", "");
    }
}

sub AddGencode
{
    my $text = shift;
    my $builder = shift;
    my $gencode_text;

    return $text if (scalar(@desktop_sms) == 0 && scalar(@tegra_sms) == 0 && !$fallback_min_ptx);

    @tegra_sms = sort @tegra_sms;
    @desktop_sms = sort @desktop_sms;

    if ($builder eq "makefile")
    {
        my $arm_sm_string = join(' ', @tegra_sms);
        my $desktop_sm_string = join(' ', @desktop_sms);

        if ($arm_sm_string eq $desktop_sm_string)
        {
            $gencode_text = <<END;
# Gencode arguments
SMS ?= $desktop_sm_string

END
        }
        else
        {
            $gencode_text = <<END;
# Gencode arguments
ifeq (\$(TARGET_ARCH),\$(filter \$(TARGET_ARCH),armv7l aarch64 sbsa))
SMS ?= $arm_sm_string
else
SMS ?= $desktop_sm_string
endif

END
        }

        if (!$fallback_min_ptx)
        {
            $conditional_exec = 1;

            $gencode_text .= <<END;
ifeq (\$(SMS),)
\$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

END
        }

        $gencode_text .= <<END;
ifeq (\$(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in \$(SMS)
\$(foreach sm,\$(SMS),\$(eval GENCODE_FLAGS += -gencode arch=compute_\$(sm),code=sm_\$(sm)))

END

        if ($fallback_min_ptx)
        {
            my ($desktop_sms_ref, $tegra_sms_ref) = ParseSMs(0, $supported_sms, $fallback_min_ptx);
            my @desktop_sms = @$desktop_sms_ref;
            my @tegra_sms = @$tegra_sms_ref;
            @desktop_sms = sort(@desktop_sms);
            @tegra_sms = sort(@tegra_sms);
            my $lowest_desktop_sm = @desktop_sms[0];
            my $lowest_tegra_sm = @tegra_sms[0];

            if ($lowest_desktop_sm eq $lowest_tegra_sm)
            {
                $gencode_text .= <<END;
ifeq (\$(SMS),)
# Generate PTX code from SM $lowest_desktop_sm
GENCODE_FLAGS += -gencode arch=compute_$lowest_desktop_sm,code=compute_$lowest_desktop_sm
endif

END
            }
            else
            {
                $gencode_text .= <<END;
ifeq (\$(SMS),)
ifeq (\$(TARGET_ARCH),\$(filter \$(TARGET_ARCH),armv7l aarch64 sbsa))
# Generate PTX code from SM $lowest_tegra_sm
GENCODE_FLAGS += -gencode arch=compute_$lowest_tegra_sm,code=compute_$lowest_tegra_sm
else
# Generate PTX code from SM $lowest_desktop_sm
GENCODE_FLAGS += -gencode arch=compute_$lowest_desktop_sm,code=compute_$lowest_desktop_sm
endif
endif

END
            }
        }

        $gencode_text .= <<END;
# Generate PTX code from the highest SM architecture in \$(SMS) to guarantee forward-compatibility
HIGHEST_SM := \$(lastword \$(sort \$(SMS)))
ifneq (\$(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_\$(HIGHEST_SM),code=compute_\$(HIGHEST_SM)
endif
endif
END

        $text = Tag($text, "GENCODE", $gencode_text) if ($gencode_text);
    }
    elsif ($builder eq "vcxproj")
    {
        if ($fallback_min_ptx)
        {
            my ($desktop_sms_ref, $tegra_sms_ref) = ParseSMs(0, $supported_sms, $fallback_min_ptx);
            my @all_sms = sort(@$desktop_sms_ref, @$tegra_sms_ref);
            my $lowest_sm = @all_sms[0];

            $gencode_text .= "compute_$lowest_sm,compute_$lowest_sm;";
        }

        for my $sm (@desktop_sms)
        {
            $gencode_text .= "compute_$sm,sm_$sm;";
        }

        $text = Tag($text, "COMPUTE_SMS", $gencode_text) if ($gencode_text);
    }

    return $text;
}

#
# Wrapper around Replace to abstract the tag (<<FOO>>) syntax
#
sub TagOne
{
    my ($file_text, $tag, $replace) = @_;

    return ReplaceOne($file_text, "<<$tag>>", $replace);
}

#
# Wrapper around Replace to abstract the tag (<<FOO>>) syntax
#
sub Tag
{
    my ($file_text, $tag, $replace) = @_;

    return Replace($file_text, "<<$tag>>", $replace);
}

#
# General purpose subroutine for replacing text in a variable
#
sub Replace
{
    my ($file_text, $tag, $replace) = @_;

    $file_text =~ s/$tag/$replace/g;

    return $file_text;
}

#
# General purpose subroutine for replacing one instance of text in a variable
#
sub ReplaceOne
{
    my ($file_text, $tag, $replace) = @_;

    $file_text =~ s/$tag/$replace/;

    return $file_text;
}

sub ParseXMLInfo
{
    my $parse = shift;
    my $data = XMLin($parse, ForceArray => [
        "keyword",
        "library",
        "supported_sm_architectures",
        "from",
        "sm",
        "flag",
        "object",
        "static",
        "path",
        "clean",
        "ignore",
        "build",
        "extracompilation",
        "extraheader",
        "extranone",
        "include",
        "exclude",
        "elwvar",
        "elw",
        "driver",
        "safe",
        "preprocessor",
        "dependency",
        "qatest",
        "concept",
    ]);
    my $recipe = $data->{recipe};

    if (ref($recipe) eq "ARRAY")
    {
        print "Invalid info.xml has multiple recipes!\n";
        return 1;
    }

    #
    # Reset the globals
    #
    $linker = "LWCC";
    $platform_specific_added = 0;
    undef $name;
    undef $title;
    undef $description;
    undef $whitepaper;
    undef $concept_text;
    undef @sample_supported_sms;
    undef @driver_api;
    undef @runtime_api;
    undef $makefile;
    undef $solution;
    undef $vscode;
    undef $src_ext;
    undef $ptx;
    undef $fatbin;
    undef $cpp11;
    undef $cpp14;
    undef $glibc;
    undef $gccMilwersion;
    undef $lwsolver;
    undef $driver;
    undef $openGL;
    undef $openGLES;
    undef $freeimage;
    undef $directX;
    undef $openMP;
    undef $MPI;
    undef $uvm;
    undef $pthreads;
    undef $x11;
    undef $vulkan;
    undef $EGL;
    undef $egloutput;
    undef $lwsci;
    undef $lwmedia;
    undef $screen;
    undef $libLWRTC;
    undef $fallback_min_ptx;
    undef $conditional_exec;
    undef @desktop_sms;
    undef @tegra_sms;
    undef @libs;
    undef @so_libs;
    undef @elw_vars;
    undef @sms;
    undef @cflags;
    undef @lwstom_objects;
    undef @source_ignore;
    undef @source_extra_compilations;
    undef @source_extra_headers;
    undef @source_extra_nones;
    undef @statics;
    undef @postbuild_events;
    undef @postbuild_eventsclean;
    undef @library_paths;
    undef @include_paths;
    undef @extra_cleans;
    undef @required_dependencies;
    undef @supported_platforms;
    undef @additional_preprocessor;
    undef @qatest_parameters;

    if ($data->{name})
    {
        $name = $data->{name};
    }
    else
    {
        print "Warning: Missing name in $parse\n";
        return 1;
    }

    $title = $data->{title};
    $description = $data->{description};
    # Process description
    $description =~ s/\<!\[CDATA\[//ig;
    $description =~ s/\]\]\>//ig;

    $whitepaper = $data->{whitepaper};

    $concept_text = "";
    for my $concept (@{$data->{keyconcepts}->{concept}})
    {
        if (ref($concept) eq "HASH" && $concept->{level})
        {
            $concept_text .= "$concept->{content}, ";
        }
        else
        {
            $concept_text .= "$concept, ";
        }
    }
    $concept_text =~ s/, $//;


    for my $api (@{$data->{lwda_api_list}->{toolkit}})
    {
        if (ref($api) eq "HASH" && $api->{level})
        {
            push @runtime_api, "$api->{content}\n";
        }
        else
        {
            push @runtime_api, "$api";
        }
    }

    for my $api (@{$data->{lwda_api_list}->{driver}})
    {
        if (ref($api) eq "HASH" && $api->{level})
        {
            push @driver_api, "$api->{content}";
        }
        else
        {
            push @driver_api, "$api";
        }
    }

    my ($desktop_sms_tmp, $tegra_sms_tmp)  = ParseSMsv2(@{$data->{supported_sm_architectures}}, $supported_sms);
    my @sms_tmp;
    @sms_tmp = uniq sort(@{$desktop_sms_tmp}, @{$tegra_sms_tmp});

    for (@sms_tmp)
    {
        push @sample_supported_sms, "$_";
    }

    for (@{$data->{supportedbuilds}->{build}})
    {
        $makefile = 1 if ($_ =~ m/makefile/i);
        $solution = 1 if ($_ =~ m/solution/i);
        $vscode   = 1 if ($_ =~ m/vscode/i);
    }

    for (@{$data->{keywords}->{keyword}})
    {
        $driver = 1 if ($_ =~ m/driver/i);

        $openMP = 1 if ($_ =~ m/openmp/i);

        $openGL = 1 if ($_ =~ m/opengl$/i);

        $openGLES = 1 if ($_ =~ m/opengl es$/i);

        $MPI = 1 if ($_ =~ m/MPI/i);

        $libLWRTC = 1 if ($_ =~ m/libLWRTC/i);
        $lwsolver = 1 if ($_ =~ m/lwsolver/i);
        $directX = "9" if ($_ =~ m/d3d9/i);
        $directX = "10" if ($_ =~ m/d3d10/i);
        $directX = "11" if ($_ =~ m/d3d11/i);
        $directX = "12" if ($_ =~ m/d3d12/i);

        $cpp11 = 1 if ($_ =~ m/CPP11/i);
        $cpp14 = 1 if ($_ =~ m/CPP14/i);
        $glibc = 1 if ($_ =~ m/GLIBC/i);
        $gccMilwersion = $_ if ($_ =~ m/GCC/i);

        $uvm = 1 if ($_ =~ m/uvm/i);
        $pthreads = 1 if ($_ =~ m/pthreads/i);
    }
    
    for (@{$data->{additional_preprocessor}->{preprocessor}})
    {
        push(@additional_preprocessor, $_);
    }
        
    for (@{$data->{libraries}->{library}})
    {
        push(@libs, $_);

        if (ref($_) eq "HASH" && $_->{content} =~ m/freeimage/i ||
            ref($_) ne "HASH" && $_ =~ m/freeimage/i)
        {
            $freeimage = 1;
        }
    }

    for (@{$data->{shared_libs}->{library}})
    {
        push(@so_libs, $_);
    }

    for (@{$data->{elwvars}->{elwvar}})
    {
        if (ref($_) ne "HASH")
        {
            push(@elw_vars, $_);
        }
        else
        {
            if ($_->{qaonly} ne "yes")
            {
                push(@elw_vars, $_->{content});
            }
        }
    }

    for (@{$data->{librarypaths}->{path}})
    {
        push(@library_paths, $_);
    }
        
    for (@{$data->{includepaths}->{path}})
    {
        push(@include_paths, $_);
    }

    for (@{$data->{sources}->{ignore}})
    {
        push(@source_ignore, $_);
    }

    for (@{$data->{sources}->{extracompilation}})
    {
        push(@source_extra_compilations, $_);
    }

    for (@{$data->{sources}->{extraheader}})
    {
        push(@source_extra_headers, $_);
    }

    for (@{$data->{sources}->{extranone}})
    {
        push(@source_extra_nones, $_);
    }

    my ($desktop_sms, $tegra_sms) = ParseSMs($data->{gencode}, $supported_sms);
    @desktop_sms = @{$desktop_sms};
    @tegra_sms = @{$tegra_sms};

    for (@{$data->{cflags}->{flag}})
    {
        push(@cflags, $_);
    }

    for (@{$data->{compilations}->{object}})
    {
        push(@lwstom_objects, {
                SRC => $_->{src},
                GENCODE => $_->{gencode}
            });
    }

    for (@{$data->{compilations}->{static}})
    {
        push(@statics, {
                SRC => $_->{src},
                TARGET => $_->{target},
                COMPILER => $_->{compiler}
            });
    }

    for (@{$data->{cleanextras}->{clean}})
    {
        push(@extra_cleans, $_);
    }

    for (@{$data->{postbuildevent}->{event}})
    {
        push(@postbuild_events, {OS => $_->{os}, EVENT => $_->{content}});
    }

    for (@{$data->{postbuildevent}->{eventclean}})
    {
        push(@postbuild_eventsclean, {OS => $_->{os}, EVENT => $_->{content}});
    }

    $ptx = $data->{ptx} if ($data->{ptx});
    $fatbin = $data->{fatbin} if ($data->{fatbin});

    $fallback_min_ptx = 1 if ($data->{fallback_min_ptx} eq "true");

    for (@{$data->{required_dependencies}->{dependency}})
    {
        if ($_ =~ m/x11/i)
        {
            $x11 = 1;
        }

        if ($_ =~ m/egloutput/i)
        {
            $egloutput = 1;
        }

        if ($_ =~ m/screen/i)
        {
            $screen = 1;
        }

        if ($_ =~ /^EGL$/i)
        {
            $EGL = 1;
        }

        if ($_ =~ m/vulkan/i)
        {
            $vulkan = 1;
        }

        if ($_ =~ /^lwsci$/i)
        {
            $lwsci = 1;
        }

        if ($_ =~ /^lwmedia$/i)
        {
            $lwmedia = 1;
        }

        push @required_dependencies, $_;
    }

    for (@{$data->{supported_elws}->{elw}})
    {
        my $platform = $_->{platform};
        my $arch = $_->{arch};

        if ($platform && $arch)
        {
            push @supported_platforms, [$platform, $arch];
        }
        elsif ($arch)
        {
            push @supported_platforms, [$arch, 1];
        }
        elsif ($platform)
        {
            push @supported_platforms, [$platform, 1];
        }
    }

    for (@{$data->{qatests}->{qatest}})
    {
        push @qatest_parameters, $_; 
    }

    return 0;
}

sub UniqArray
{
    my %seen;
    my @return_array;

    foreach (@_)
    {
        unless ($seen{$_})
        {
            push @return_array, $_;
            $seen{@_} = 1;
        }
    }

    return @return_array;
}

sub HandleCommandLineOptions
{
    if (!$info)
    {
        print "Info path must be given!\n\n";
        Usage();
        exit 1;
    }

    if ($builder_types)
    {
        my @types = sort split(',', $builder_types);

        for (@types)
        {
            if ($_ =~ m/makefile/i)
            {
                $generate_makefile = 1;
            }
            elsif ($_ =~ m/^vs$/i)
            {
                @build_vs_years = @supported_vs_years;
            }
            elsif ($_ =~ m/vs(\d+)/i)
            {
                my $supported = 0;
                for my $lwr_supported_year (@supported_vs_years)
                {
                    $supported = 1 if ($lwr_supported_year == $1);
                }

                if ($supported)
                {
                    push (@build_vs_years, "$1");
                }
                else
                {
                    print "Unsupported Visual Studio year requested: $1!\n\n";
                    Usage();
                    exit 1;
                }
            }
            elsif ($_ =~ m/eris/i)
            {
                $generate_eris_vlct = 1;
                $generate_eris_vlct = 2 if ($_ =~ m/eris_ppc64le/i); # Generate samples_tests_ppc64le.vlct for ppc64le
            }
        }
    }
    else
    {
        @build_vs_years = @supported_vs_years;
        $generate_makefile = 1;
    }

    if ($target_os && !$target_arch)
    {
        $target_arch = "1";
    }
}

sub Clean
{
    unlink "$sample_dir/findgllib.mk";
    unlink "$sample_dir/Makefile";
}

sub verify_comparator
{
    my $line1 = shift;
    my $line2 = shift;

    $line1 =~ s/\s*$//;
    $line2 =~ s/\s*$//;

    return $line1 ne $line2;
}

sub Verify
{
    my $failed = 0;
    my $verifyDir = shift;

    opendir(GENERATED, "$output_dir");
    while(my $file = readdir(GENERATED))
    {
        next if ($file eq "." || $file eq "..");

        print "Verifying $file\n";
        my $different = compare_text("$output_dir/$file", "$verifyDir/$file", \&verify_comparator);

        if ($different)
        {
            print "ERROR: $file differs between generated and present!\n";
            print "ERROR: $output_dir/$file and $verifyDir/$file are different\n";
            
            $failed = 1;
        }
    }
    closedir(GENERATED);

    return $failed;
}

sub Run
{
    my $ret = RunReturnCode(shift);
    die "Command failed!" if $ret;
}

sub RunReturnCode
{
    my $cmd = shift;
    print ">>> $cmd\n" if $debug;
    my $ret = system "$cmd";
    return $ret >> 8;
}

sub Usage
{
    my $type_set = "makefile,vs";
    $type_set .= ",vs$_" for (@supported_vs_years);

    my $usage_string = <<END;
Usage: perl generate_builders.pl [options]

Options:
    --info=PATH               : Path to info.xml location. If directory, will
                                relwrsively search for info.xml files.
    --verify=PATH             : Path to verify against info.xml. Enables verification.
    --type=TYPESTRING         : A comma-delimited list of builders to process.
                                Elements must be subset of {$type_set}.
    --generate-eris-vlct      : Generates Samples vulcan/eris testsuite file "samples.vlct".
    --p4-edit                 : Enable automatic file checkout.
    --p4-cl=CL                : If file checkout is enabled, specify CL to check out
                                files in. Defaults to a new CL.
    --debug                   : Print additional debug information.
    --quiet                   : Squelch all output.
    --supported-sms=SM_STRING : Pass supported SMs string to generate all the makefiles
                                for samples. example: "20 30 32 35"
    -h|--help                 : Print this dialogue.

END

    print $usage_string;
}
