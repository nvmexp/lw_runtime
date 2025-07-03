package dvsLwdaSamples;

use strict;
use warnings;
use Exporter;
use vars qw($VERSION @ISA @EXPORT @EXPORT_OK);
use FindBin;
use File::Copy;
use File::Basename;
use File::Path qw(mkpath);
use Getopt::Long;
use Cwd;
use if $^O eq "MSWin32", "dvsLwdaSamplesWin32";
use if $^O eq "cygwin", "dvsLwdaSamplesWin32";
use if $^O ne "MSWin32", "dvsLwdaSamplesLinux";

$VERSION     = 1.00;
@ISA         = qw(Exporter);
@EXPORT      = qw(exec print_test_preambule print_test_summary
                  print_results_summary start_timer stop_timer
                  print_timing_summary build_samples_blacklist
                  build_samples_list build_filtered_samples_list
                  copy_header_files
                  set_commands set_commands_docker set_elwironment set_paths set_elwironment_docker
                  copy_msvc_build_extensions copy_installation
                  build_samples build_samples_docker package_samples build_libraries
                  set_compression_commands merge_required_packages
                  build_and_run_samples run_samples copy_library_files
                  find_rootdir get_num_failures build_keyword_samples
                  clobber_samples set_run_elwironment
                  read_command_line_options);
@EXPORT_OK   = @EXPORT;

use constant BLACK_LIST_FILE => 'exclude_list.txt';
use constant SAMPLES_LIST_FILE => 'samples_list.txt';

# Command-Line Options
my $mode = "";
my $toolkit_rootdir = "";
my $driver_rootdir = "";
my $samples_rootdir = "";
my $samples_dir = "";
my $samples_scripts_dir = "";
my $rootdir = "";
my $os = "";
my $host_arch = "";
my $target_arch = "";
my $abi = "";
my $build = "";
my $p4_build_rootdir = "";
my $package = "";
my $help = 0;
my $debug = 0;
my $use_docker = 0;
my $package_sources_only = 0;
my $test_name = "";
my $test_list_file = "";
my $filter_list_file = "";
my $action = "all";
my $parallel = 16;
my $environment = "dev";
my $p4_tools_dir = "";
my $p4_toolkit_dir = "";
my $compiler_export = "";
my $skip_blacklist = 0;
my $skip_verification = 0;
my $vulcan_toolkit_dir = "";
my $msvc_version = "";
my $skip_build_files_generation = 0;
my $perl_override = "";
my $linux_perl_path;

# Limits
my $timeout = 120;
my $test_output_limit = 10000;

# Build Stats
my $passed = 0;
my $failed = 0;
my $waived = 0;

# Timing Stats
my $start_time = "";
my $stop_time = "";
my @time_measurements = ();

# Compression Commands
my $unzip = "";
my $delete_pkg_cmd = "";
my $pkg_cmd = "";

# Permitted MSVC Versions
my @supported_msvc = ("VC11", "VC12", "VS2019");

# Initialize package
sub check_paths
{
  if (not -e $p4_build_rootdir) {
    print STDERR "Error: p4 build rootdir '$p4_build_rootdir' does not exist";
    exit 1;
  }

  if ($environment eq "rel" && not -e $p4_toolkit_dir) {
    print STDERR "Error: p4 toolkit dir '$p4_toolkit_dir' does not exist";
    exit 1;
  }

  if ($environment eq "rel" && not -e $p4_tools_dir) {
    print STDERR "Error: p4 tools dir '$p4_tools_dir' does not exist";
    exit 1;
  }

  if (not -e $toolkit_rootdir) {
    print STDERR "Error: toolkit rootdir '$toolkit_rootdir' does not exist";
    exit 1;
  }

  #if (not -e $driver_rootdir) {
  #  print STDERR "Error: driver rootdir '$driver_rootdir' does not exist";
  #  exit 1;
  #}

  if ($environment eq "rel" && not -e $samples_rootdir) {
    print STDERR "Error: samples rootdir '$samples_rootdir' does not exist";
    exit 1;
  }
}

sub get_num_failures
{
  return $failed;
}

# Wrapper for system that logs the commands so you can see what it did
sub exec
{
    my ($cmd) = @_;
    my  $ret = 0;

    if ($cmd eq "") {
      return;
    }

    print "\n$cmd\n" if $debug;

    # Clear subprocess return value.
    # This is needed to avoid false failures propagating from a legitimate
    # failure.  For some reason a failing test can trigger bad return values
    # in subsequent tests without this.
    $? = 0;

    eval {
        local $SIG{ALRM} = sub {die "alarm\n"};
        alarm (60 * $timeout);

        # The stderr redirect had to be removed as it was causing unexpected
        # failures on Windows
        if ($os eq "win32") {
            open(SYSCMDHANDLE, "$cmd  |") || die "Unable to open system command pipe:$!\n";
        }
        else {
            open(SYSCMDHANDLE, "$cmd 2>&1  |") || die "Unable to open system command pipe:$!\n";
        }

        # truncate output
        my $linenum = 0;
        while (<SYSCMDHANDLE>){
            $linenum += 1;
            if ($linenum > $test_output_limit) {
                print "^^^^ OUTPUT TRUNCATED\n" if($linenum == $test_output_limit);
                last;
            }
            print $_;
        }
        close (SYSCMDHANDLE);

        # grab return of subprocess
        $ret = $?;
        printf "Application return code: $ret\n" if $debug;

        alarm 0;
    };

    if ($@) {
        if ($@ eq "alarm") {
            my @exelwtable = split(' ', $cmd, 2);
            printf "\n Error: Application timed out. Killing $exelwtable[0].\n";
            system ("killall ".$exelwtable[0]);
        }
        else {
            my $err = $@;
            printf "\n Error: Application died unexpectedly : $err\n";
        }
        $ret = 1;
    }

    my $signals  = $ret & 127;
    my $app_exit = $ret >> 8;
    my $dumped_core = $ret & 0x80;

    return $ret if ($ret == 0 || $ret == 1 || $ret == 2);
    return $app_exit if ($app_exit == 1 or $app_exit == 2);

    if (($app_exit != 0)) {
        printf "\n Error: Application exited with status $app_exit\n";
    }
    if ($signals != 0) {
        printf "\n Error: Application received signal $signals\n";
    }
    if ($dumped_core != 0) {
        printf "\n Error: Application generated a core dump\n";
    }

    return 1;
}

sub find_rootdir
{
  my ($path) = @_;
  my @components = split(/[\/\\]+/, $path);
  my $index = 0;

  for my $component (@components) {
    last if $components[$index] eq "sw";
    $index += 1;
  }

  $path = "/" . join ('/', @components[0..$index-1]);
  $path =~ s#//#/#g;
  $path =~ s#^/([a-zA-Z]):#$1:#;

  if ($mode eq "dvs" and $os eq "win32") {
    $path =~ s#/#\\#g;
  }

  return $path
}

sub print_usage
{
  my $script_name = basename ($0);

  print STDERR "Usage:  $script_name <arguments> <options>\n";
  print STDERR "\n";
  print STDERR "Required Arguments:\n";
  print STDERR "  --mode <mode>                  : manual, qa, or dvs\n";
  print STDERR "  --toolkit-rootdir <dir>        : toolkit root directory\n";
  print STDERR "  --driver-rootdir <dir>         : driver root directory\n";
  print STDERR "  --arch i386|i686|x86_64        : architecture type for --host-arch and --target-arch\n";
  print STDERR "  --os win32|Darwin|Linux        : operating system\n";
  print STDERR "  --build debug|release          : build type\n";
  print STDERR "  --p4build <dir>                : p4 build directory\n";
  print STDERR "  --package <pathname>           : full pathname of the result package\n";
  print STDERR "\n";
  print STDERR "Options:\n";
  print STDERR "  --help                         : Print this help message\n";
  print STDERR "  --debug                        : Print extra debug messages\n";
  print STDERR "  --package-sources-only         : Package only sources & build files, skip binaries building\n";
  print STDERR "  --host-arch i386|i686|x86_64|ARMv7|aarch64|ppc64le   : architecture of the build machine\n";
  print STDERR "  --target-arch i386|i686|x86_64|ARMv7|aarch64|ppc64le : architecture of the target machine\n";
  print STDERR "  --abi gnueabi|gnueabihf        : ABI for ARMv7. Default is gnueabi\n";
  print STDERR "  --testname <name>              : run only the tests whose name includes <name>\n";
  print STDERR "  --test-list-file <path>        : white list of tests to run, one per line\n";
  print STDERR "  --filter-list-file <path>      : black list of tests not to run, one per line\n";
  print STDERR "  --action clobber|build|run|all : build and/or run the samples. Default is '$action'.\n";
  print STDERR "  --parallel <N>                 : parallel build factor. Default is '$parallel'.\n";
  print STDERR "  --keywords-include <k>[,<k>]*  : only include the samples that match any of the keywords.\n";
  print STDERR "  --keywords-exclude <k>[,<k>]*  : exclude the samples that match any of the keywords.\n";
  print STDERR "  --environment dev|rel          : are the binaries/libraries/headers in a development tree or in an official release location? Default is '$environment'.\n";
  print STDERR "  --samples-rootdir <dir>        : samples root directory. Default is <toolkit_dir>/samples.\n";
  print STDERR "  --compiler-export <dir>        : compiler export directory. Default is <toolkit_dir>/compiler/export.\n";
  print STDERR "  --skip-blacklist               : skips the default blacklist (" . BLACK_LIST_FILE . ").\n";
  print STDERR "  --skip-verification            : skips the builder and info.xml verification.\n";
  print STDERR "  --vulcan-toolkit-dir           : Uses toolkit installed in vulcan directory. \n";
  print STDERR "  --msvc-version                 : Permitted values are VC10, VC11, VC12. Default is VC10. Only usable when --os=win32. \n";
  print STDERR "  --perl-override                : Overrides perl binary used for generating Makefiles. Only usable when --os=Linux. \n";
}

sub read_command_line_options
{
  my ($mode_ref, $action_ref, $elwironment_ref, $keywords_include, $keywords_exclude, $use_docker_ref) = @_;

  $mode = $$mode_ref;
  $action = $$action_ref;
  $environment = $$elwironment_ref;

  my $arch = "";
  my $keywords_include_string = "";
  my $keywords_exclude_string = "";

  GetOptions(
    "mode=s"                   => \$mode,
    "toolkit-rootdir=s"        => \$toolkit_rootdir,
    "driver-rootdir=s"         => \$driver_rootdir,
    "arch=s"                   => \$arch,
    "host-arch=s"              => \$host_arch,
    "target-arch=s"            => \$target_arch,
    "os=s"                     => \$os,
    "abi=s"                    => \$abi,
    "build=s"                  => \$build,
    "p4build=s"                => \$p4_build_rootdir,
    "package=s"                => \$package,
    "help|h"                   => \$help,
    "debug"                    => \$debug,
    "use-docker"               => \$use_docker,
    "package-sources-only"     => \$package_sources_only,
    "testname=s"               => \$test_name,
    "test-list-file=s"         => \$test_list_file,
    "filter-list-file=s"       => \$filter_list_file,
    "action=s"                 => \$action,
    "parallel=n"               => \$parallel,
    "keywords-include=s"       => \$keywords_include_string,
    "keywords-exclude=s"       => \$keywords_exclude_string,
    "environment=s"            => \$environment,
    "p4-toolkit-dir=s"         => \$p4_toolkit_dir,
    "p4-tools-dir=s"           => \$p4_tools_dir,
    "samples-rootdir=s"        => \$samples_rootdir,
    "compiler-export=s"        => \$compiler_export,
    "skip-blacklist"           => \$skip_blacklist,
    "skip-verification"        => \$skip_verification,
    "vulcan-toolkit-dir=s"     => \$vulcan_toolkit_dir,
    "msvc-version=s"           => \$msvc_version,
    "perl-override=s"          => \$perl_override,
  ) or die "Error while processing the command-line options. Exiting.";

  if ($help) {
    print_usage();
    exit 0;
  }

  if ($mode eq "qa" and ($driver_rootdir eq ""))
    {
      print STDERR "Error: In QA mode, --driver-rootdir should be passed.\n\n";
      print_usage ();
      exit 1;
    }

  if ($arch ne "" and ($host_arch ne "" or $target_arch ne ""))
    {
      print STDERR "Error: cannot use --arch with --host-arch or --target-arch.\n\n";
      print_usage ();
      exit 1;
    }

  if ($environment eq "dev" and ($vulcan_toolkit_dir ne ""))
    {
       print STDERR "Error: cannot pass --vulcan-toolkit-dir when --environment=dev.\n\n";
       print_usage ();
       exit 1;
    }

  $skip_build_files_generation = 1 if ($vulcan_toolkit_dir ne "");

  if ($arch ne "")
    {
      $target_arch = $arch;
      $host_arch = $arch;
    }
  $host_arch = "i386" if ($host_arch eq "i686" && $os eq "Darwin"); 
  $host_arch = "i686" if ($host_arch eq "i386" && $os ne "Darwin");
  $target_arch = "i386" if ($target_arch eq "i686" && $os eq "Darwin"); 
  $target_arch = "i686" if ($target_arch eq "i386" && $os ne "Darwin");

  $p4_build_rootdir = find_rootdir ($toolkit_rootdir) if $p4_build_rootdir eq "";
  # append the p4 build folder to the paths passed as parameters 
  if ($mode eq "dvs" && ($action eq "build" || $action eq "clobber"))
  {
    if ($os eq "win32") {
        $toolkit_rootdir =~ s#^\\##;
        $driver_rootdir  =~ s#^\\##;
        $package         =~ s#^\\##;
    } else {
        $toolkit_rootdir =~ s#^/##;
        $driver_rootdir  =~ s#^/##;
        $package         =~ s#^/##;
    }
 
    # XXX Nuke p4_build_rootdir and have DVS config files pass the full paths instead. 
    $toolkit_rootdir = "$p4_build_rootdir/$toolkit_rootdir";
    $driver_rootdir  = "$p4_build_rootdir/$driver_rootdir";
    $package         = "$p4_build_rootdir/$package";
  }

  $p4_tools_dir = "$p4_build_rootdir/sw/tools" if $p4_tools_dir eq "";
  $p4_toolkit_dir = $toolkit_rootdir if $p4_toolkit_dir eq "";

  $samples_dir = $samples_rootdir;
  $samples_rootdir = "$toolkit_rootdir/$samples_rootdir/Samples";
  $samples_scripts_dir = "$samples_rootdir/../scripts";

  if ($mode eq "dvs" && $os eq "win32") {
    $toolkit_rootdir  =~ s#/#\\#g;
    $driver_rootdir   =~ s#/#\\#g;
    $package          =~ s#/#\\#g;
    $p4_build_rootdir =~ s#/#\\#g;
    $p4_toolkit_dir   =~ s#/#\\#g;
    $p4_tools_dir     =~ s#/#\\#g;
    $samples_rootdir  =~ s#/#\\#g;
    $samples_scripts_dir =~ s#/#\\#g;
  }

  if ($debug) {
    print "\n";
    print "COMMAND-LINE OPTIONS:\n";
    print "  mode             = $mode\n";
    print "  toolkit_rootdir  = $toolkit_rootdir\n";
    print "  driver_rootdir   = $driver_rootdir\n";
    print "  host_arch        = $host_arch\n";
    print "  target_arch      = $target_arch\n";
    print "  os               = $os\n";
    print "  abi              = $abi\n";     
    print "  build            = $build\n";
    print "  p4_build_rootdir = $p4_build_rootdir\n";
    print "  package          = $package\n";
    print "  help             = $help\n";
    print "  debug            = $debug\n";
    print "  use-docker       = $use_docker\n";
    print "  package-sources-only = $package_sources_only\n";
    print "  testname         = $test_name\n";
    print "  test_list_file   = $test_list_file\n";
    print "  filter_list_file = $filter_list_file\n";
    print "  action           = $action\n";
    print "  parallel         = $parallel\n";
    print "  keywords_include = $keywords_include_string\n";
    print "  keywords_exclude = $keywords_exclude_string\n";
    print "  environment      = $environment\n";
    print "  p4_toolkit_dir   = $p4_toolkit_dir\n";
    print "  p4_tools_dir     = $p4_tools_dir\n";
    print "  samples_rootdir  = $samples_rootdir\n";
    print "  vulcan_toolkit_dir = $vulcan_toolkit_dir\n";
    print "  msvc_version     = $msvc_version\n" if ($os eq "win32");
    print "  perl_override    = $perl_override\n";
    print "\n";
  }

  foreach my $k (split ',', $keywords_include_string) {
    ${$keywords_include}{$k} = $k;
  }

  foreach my $k (split ',', $keywords_exclude_string) {
    ${$keywords_exclude}{$k} = $k;
  }

  if ($mode eq "" ||
      $toolkit_rootdir eq "" || $samples_rootdir eq "" ||
      ($host_arch ne "x86_64" && $host_arch ne "i686" && $host_arch ne "i386" &&
       $host_arch ne "ARMv7" && $host_arch ne "aarch64" && $host_arch ne "ppc64le") ||
      ($target_arch ne "x86_64" && $target_arch ne "i686" && $target_arch ne "i386" &&
       $target_arch ne "ARMv7" && $target_arch ne "aarch64" && $target_arch ne "ppc64le") ||
      ($os ne "win32" && $os ne "Linux" && $os ne "Darwin") ||
      ($host_arch eq "ARMv7" && $abi ne "" && $abi ne "gnueabi" && $abi ne "gnueabihf") ||
      ($target_arch eq "ARMv7" && $abi ne "" && $abi ne "gnueabi" && $abi ne "gnueabihf") ||
      ($build ne "debug" && $build ne "release") ||
      ($action ne "build" && $action ne "run" && $action ne "all" && $action ne "clobber") ||
      ($environment ne "dev" && $environment ne "rel"))
  {
    print STDERR "Error: missing argument\n\n";
    print_usage();
    exit 1;
  }

  if ($environment eq "rel" && ($p4_toolkit_dir eq "" || $p4_tools_dir eq ""))
    {
      print STDERR "Error: with --environment=rel, --p4-toolkit-dir and --p4-tools must be set.\n\n";
      print_usage ();
      exit 1;
    }

  if($abi ne "") {
    $abi = "_$abi";
  } else {
    if($target_arch eq "ARMv7") {
        $abi = "_gnueabi";
    }
  }
  
  if ($mode eq "dvs" and
      ($action eq "build" or $action eq "clobber") and
      ($p4_build_rootdir eq "" or $driver_rootdir eq "" or $package eq ""))
  {
    print STDERR "Error: missing argument\n\n";
    print_usage();
    exit 1;
  }

  if ($msvc_version ne "" &&  $os ne "win32")
  {
    print STDERR "Error: --msvc-version option only permitted with --os=win32\n\n";
    print_usage();
    exit 1;
  }

  if ($perl_override ne "")
  {
    if ($os ne "Linux")
    {
      print STDERR "Error: --perl-override option only permitted with --os=Linux\n\n";
      print_usage();
      exit 1;
    }

    $linux_perl_path = $perl_override;
  }
  else
  {
    $linux_perl_path = "$p4_tools_dir/linux/perl-5.18.1/bin/perl";
  }

  if ($msvc_version ne "")
  {
    my $supported = 0;
    for my $lwr_supported_msvc (@supported_msvc)
    {
      $supported = 1 if ($lwr_supported_msvc =~ m/$msvc_version/i);
    }
    if (!$supported)
    {
      print "Unsupported MSVC version requested: $msvc_version!\n\n";
      Usage();
      exit 1;
    }
  }

  check_paths ();

  $$mode_ref = $mode;
  $$action_ref = $action;
  $$elwironment_ref = $environment;
  $$use_docker_ref = $use_docker;
}

sub set_run_elwironment
{
  my $toolkit_bin_path = "$toolkit_rootdir/bin";
  $toolkit_bin_path = "$toolkit_rootdir/bin/${target_arch}_${os}${abi}_${build}" if $environment eq "dev";

  my $build_path = "";
  $build_path = "$toolkit_rootdir/build" if $environment eq "dev";

  if ($os eq "win32") {
      $toolkit_bin_path =~ s#^\\##;
      $build_path       =~ s#^\\##;

      $ELW{'LWDA_PATH'} = $toolkit_rootdir;
      $ELW{'LWDA_PATH_V10_0'} = "${toolkit_bin_path}";
      $ELW{'LWDA_PATH_V10_1'} = "${toolkit_bin_path}";
      $ELW{'LWDA_PATH_V10_2'} = "${toolkit_bin_path}";
      $ELW{'LWDA_PATH_V11_0'} = "${toolkit_bin_path}";
      $ELW{'LWDA_PATH_V11_1'} = "${toolkit_bin_path}";
      $ELW{'LWDA_PATH_V11_2'} = "${toolkit_bin_path}";

      $ELW{'PATH'} = "${toolkit_bin_path};${build_path};$ELW{'PATH'}";
  }

  if ($os eq "Linux") {
      $ELW{'LD_LIBRARY_PATH'} = "" if not defined $ELW{'LD_LIBRARY_PATH'};
      $ELW{'LD_LIBRARY_PATH'} = "$toolkit_bin_path:$ELW{'LD_LIBRARY_PATH'}";
      $ELW{'PATH'} = "${toolkit_bin_path}:${build_path}:$ELW{'PATH'}";
  } 
  
  if ($os eq "Darwin") {
      $ELW{'LD_LIBRARY_PATH'} = "" if not defined $ELW{'LD_LIBRARY_PATH'};
      $ELW{'DYLD_LIBRARY_PATH'} = "$toolkit_bin_path:$ELW{'LD_LIBRARY_PATH'}";
      $ELW{'PATH'} = "${toolkit_bin_path}:${build_path}:$ELW{'PATH'}";
  }

  if ($debug) {
      print "\n";
      print "ENVIRONMENT:\n";
      print "  PATH = $ELW{'PATH'}\n";
      print "  LWDA_PATH = $ELW{'LWDA_PATH'}\n" if ($os eq "win32");
      print "  LD_LIBRARY_PATH = $ELW{'LD_LIBRARY_PATH'}\n" if ($os eq "Linux");
      print "  DYLD_LIBRARY_PATH = $ELW{'DYLD_LIBRARY_PATH'}\n" if ($os eq "Darwin");
      print "\n";
  }
}

sub print_test_preambule
{
  my $step = $_[0];
  my $sample = $_[1];
  my $lwrrent_time= get_lwrrent_time();

  print "\n";
  print "&&&& INFO LWCA sample [${sample}] at $lwrrent_time\n";
  print "&&&& RUNNING ${sample}\n" if ($vulcan_toolkit_dir ne "");
  print "&&&& $step ${sample}\n";
  print "\n";
}

sub print_test_summary
{
  my $sample = $_[0];
  my $ret = $_[1];

  $passed += 1 if $ret == 0;
  $failed += 1 if $ret == 1;
  $waived += 1 if $ret == 2;
  $failed += 1 if $ret >= 3;

  print "\n";
  print "&&&& PASSED $sample\n" if $ret == 0;
  print "&&&& FAILED $sample\n"  if $ret == 1;
  print "&&&& WAIVED $sample\n"  if $ret == 2;
}

sub print_results_summary
{
  my $total = $passed + $waived + $failed;
  my $success = $passed + $waived;
  my $score = $total > 0 ? 100 * $success / $total : 0;

  print "\n";
  print "RESULTS\n";
  print "Passed : $passed\n";
  print "Waived : $waived\n";
  print "Failed : $failed\n";
  printf "DVS SCORE: %.1f\n", $score;

  exit (1) if ($failed >= 1);
}

sub get_lwrrent_time
{
   my ($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime(time);
   $year += 1900;
   $mon += 1;
   return sprintf ("%04d-%02d-%02d %02d:%02d:%02d", $year, $mon, $mday, $hour, $min, $sec);
}

sub start_timer
{
  my ($name) = @_;
  $start_time = time();
  $stop_time = 0;
}

sub stop_timer
{
  my ($name) = @_;
  $stop_time = time();
  my @measurement = ($name, $start_time, $stop_time);
  push @time_measurements, \@measurement;
}

sub print_timing_summary
{
  print "\n";
  print "TIMING RESULTS\n";
  for my $measurement (@time_measurements) {
    my ($name, $start, $stop) = @{$measurement};
    my $duration = $stop - $start;
    printf "%-12s : %d seconds\n", $name, $duration;
  }
}

sub build_samples_blacklist
{
  if ($skip_blacklist)
    {
      print "Skipping default blacklist as requested.\n";
      return;
    }

  my $blacklist;
  my $black_list_file;

  if (ref($_[0]) eq 'HASH')
    {
      # We're being passed just the blacklist ref directly
      $blacklist = $_[0];
    }
  else
    {
      # We're being passed a hash with the options
      my %params = @_;
      $blacklist = $params{BLACKLIST_REF};

      $black_list_file = $params{BLACK_LIST_FILE} if ($params{BLACK_LIST_FILE});
      $mode = $params{MODE} if ($params{MODE});
      $os = $params{OS} if ($params{OS});
      $target_arch = $params{ARCH} if ($params{ARCH});
      $build = $params{BUILD} if ($params{BUILD});
    }

  if (!$black_list_file)
    {
      $black_list_file = $FindBin::RealBin."/".BLACK_LIST_FILE;
    }

  open (BLACKLIST, $black_list_file) or die "Unable to open the samples black list file ($black_list_file)"; 

  while (my $line = <BLACKLIST>)
    {
      $line =~ s/#.*//;   # ignore comments by erasing them
      $line =~ s/\s+$//;  # remove trailing white space
      $line =~ s/^\s+//;  # remove initial white space
      chomp $line;       # remove trailing newline characters

      next if ($line =~ m/^(\s)*$/); # ignore blank lines

      if ($line =~ m/^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+"(.*)"$/)
        {
          my $blacklist_mode = $1;
          my $blacklist_os = $2;
          my $blacklist_arch = $3;
          my $blacklist_build = $4;
          my $blacklist_testname = $5;
          my $blacklist_reason = $6;

          $blacklist_arch = "i386" if ($blacklist_arch eq "i686" and $os eq "Darwin");
          $blacklist_arch = "i686" if ($blacklist_arch eq "i386" and $os ne "Darwin");

          if (($blacklist_mode eq $mode        || $blacklist_mode eq "all") &&
              ($blacklist_os eq $os            || $blacklist_os eq "all") &&
              ($blacklist_arch eq $target_arch || $blacklist_arch eq "all") &&
              ($blacklist_build eq $build      || $blacklist_build eq "all"))
            {
              ${$blacklist}{$blacklist_testname} = "$blacklist_reason";
            }
        }
    }

  close (BLACKLIST);

  if ($debug)
    {
      print "\n";
      printf "BLACKLISTED SAMPLES (%d)\n", scalar keys %$blacklist;
      foreach my $sample (sort (keys %$blacklist))
        {
          print "  $sample\n";
        }
      print "\n";
    }
}

sub build_samples_list
{
  my ($samples_list, $blacklist, $filtered_samples, $keyword_samples) = @_;
  my $dflt_list_file = "$samples_scripts_dir/dvs/".SAMPLES_LIST_FILE;

  if ($test_name ne "") {
    build_samples_list_from_name ($samples_list, $blacklist, $filtered_samples, $keyword_samples);
  } elsif ($test_list_file ne "") {
    build_samples_list_from_file ($samples_list, $blacklist, $filtered_samples, $keyword_samples, $test_list_file);
  } else {
    build_samples_list_from_file ($samples_list, $blacklist, $filtered_samples, $keyword_samples, $dflt_list_file);
  }

  @$samples_list = sort (@$samples_list);

  if ($debug)
    {
      print "\n";
      printf "SAMPLES (%d)\n", scalar @{$samples_list};
      foreach my $sample (@$samples_list)
       {
          print "  $sample\n";
       }
      print "\n";
    }
}

sub build_samples_list_from_file
{
  my ($samples_list, $blacklist, $filtered_samples, $keyword_samples, $list_file) = @_;

  return if $list_file eq "";

  open (SAMPLE_LIST, $list_file) or die "Unable to open the list file ($list_file)"; 
  while (<SAMPLE_LIST>)
    {
      s/#.*//;   # ignore comments by erasing them
      s/\s+$//;  # remove trailing white space
      s/^\s+//;  # remove initial white space
      chomp;     # remove trailing newline characters

      next if /^(\s)*$/;            # ignore blank lines
      next if exists ${$filtered_samples}{$_};
      next if exists ${$blacklist}{$_};
      next if not exists ${$keyword_samples}{$_};

      push @{$samples_list}, $_; 
    } 
  close (SAMPLE_LIST);
}

sub build_samples_list_from_name
{
  my ($samples_list, $blacklist, $filtered_samples, $keyword_samples) = @_;

  return if $test_name eq "";
  return if exists ${$blacklist}{$test_name};
  return if not exists ${$keyword_samples}{$test_name};
  
  push @{$samples_list}, $test_name;
}

sub build_filtered_samples_list_from_file
{
  my ($filtered_samples) = @_;

  return if $filter_list_file eq "";

  open (FILTER_LIST, $filter_list_file) or die "Unable to open the filter list file ($filter_list_file)"; 
  while (<FILTER_LIST>)
    {
      s/#.*//;   # ignore comments by erasing them
      s/\s+$//;  # remove trailing white space
      s/^\s+//;  # remove initial white space
      chomp;     # remove trailing newline characters

      next if /^(\s)*$/;            # ignore blank lines

      ${$filtered_samples}{$_} = $_; 
    } 
  close (FILTER_LIST);
}

sub build_filtered_samples_list
{
  my ($filtered_samples) = @_;

  if ($filter_list_file ne "") {
    build_filtered_samples_list_from_file ($filtered_samples, $filter_list_file);
  }

  if ($debug)
    {
      print "\n";
      printf "FILTERED SAMPLES (%d)\n", scalar keys %$filtered_samples ;
      foreach my $sample (sort (keys %{$filtered_samples}))
        {
          print "  $sample\n";
        }
      print "\n";
    }
}

sub build_keyword_samples
{
  my ($keyword_samples, $keywords_include, $keywords_exclude) = @_;
  my $samples_list_file = "$samples_scripts_dir/dvs/".SAMPLES_LIST_FILE;
  my $keyword_include_count = scalar keys %{$keywords_include};
  my $keyword_exclude_count = scalar keys %{$keywords_exclude};
  my $res = 0;

  open (SAMPLE_LIST, $samples_list_file) or die "Unable to open the test list file ($samples_list_file)"; 
  while (<SAMPLE_LIST>)
    {
      s/#.*//;   # ignore comments by erasing them
      s/\s+$//;  # remove trailing white space
      s/^\s+//;  # remove initial white space
      chomp;     # remove trailing newline characters

      next if /^(\s)*$/;            # ignore blank lines

      my $sample = $_; 

      # no keyword options means accept all the samples
      if ($keyword_include_count == 0 and $keyword_exclude_count == 0) {
          ${$keyword_samples}{$sample} = $sample;
          next;
      }

      # no info.xml means missing sample, therefore it's automatic exclude
      # happens on DVS run machine where the excluded samples are already excluded from the package       
      $res = open (INFO_XML, "$samples_rootdir/$sample/info.xml");
      next if (not defined ($res));

      # no keyword include option means accepted unless excluded
      my $include = ($keyword_include_count == 0);
      my $exclude = 0;

      while (<INFO_XML>)
        {
          my $line = $_;
          chomp ($line);
          next if $line !~ m/<keyword>.*<\/keyword>/;
          my $k = $line;
          $k =~ s/.*<keyword>(.*)<\/keyword>.*/$1/; 
          $include = 1 if exists ${$keywords_include}{$k};
          $exclude = 1 if exists ${$keywords_exclude}{$k};
        }      
      close (INFO_XML);

      ${$keyword_samples}{$sample} = $sample if $include and not $exclude;
    } 
  close (SAMPLE_LIST);

  if ($debug)
    {
      print "\n";
      printf "KEYWORD SAMPLES (%d)\n", scalar keys %{$keyword_samples};
      foreach my $sample (sort (keys %{$keyword_samples}))
        {
          print "  $sample\n";
        }
      print "\n";
    }
}

sub build_sample_ilwocations
{
  my ($sample) = @_;
  my @ilwocations = ();
  my @elwvars = ();
  my $res = 0;

  my $info_file = "";
  my @info_file_candidates = ("$samples_rootdir/$sample/info.xml", 
                              "$samples_rootdir/$sample/NsightEclipse.xml",
                             );

  foreach my $info_file_candidate (@info_file_candidates)
    {
      if (-e $info_file_candidate)
        {
          $info_file = $info_file_candidate;
          last;
        }
    }

  $res = open (INFO_XML, "$samples_rootdir/$sample/info.xml");
  if (not defined($res)) {
    print STDERR "Error: Cannot open $samples_rootdir/$sample/info.xml\n";
    return @ilwocations;
  }

  # read the ilwocations from info.xml
  while (<INFO_XML>) {
    my $line = $_;
    chomp ($line);

    if ($line =~ m/<qatest>.*<\/qatest>/) {
      my $invocation = $line;
      $invocation =~ s/.*<qatest>(.*)<\/qatest>.*/$1/; 
      push @ilwocations, $invocation;
    }

    if ($line =~ m/<elwvar.*>.*<\/elwvar>/) {
      my $elwvar = $line;
      $elwvar =~ s/.*<elwvar.*>(.*)<\/elwvar>.*/$1/; 
      push @elwvars, $elwvar;
    }
  }
  close (INFO_XML);

  # if no invocation specified, then ilwoke with zero argument
  if (scalar (@ilwocations) == 0) {
    push @ilwocations, "";
  }

  # same with environment variables
  if (scalar (@elwvars) == 0) {
    push @elwvars, "";
  }

  return (\@elwvars, \@ilwocations);
}

sub verify_sample
{
  if ($skip_verification)
    {
      print "Skipping verification as requested.\n";
      return;
    }

  my $sample = shift;
  my $ret = 0;
 
  $ret = verify_sample_info($sample) if (!$ret);

  return $ret;
}

sub verify_sample_info
{
  my $sample = shift;

  if ($os eq "Linux")
    {
      return dvsLwdaSamples::exec "xmllint --noout --path $samples_scripts_dir --valid $samples_rootdir/$sample/info.xml";
    }

  return 0;
}

my $cygwin_toolkit_path;
my $cygwin_samples_root_path;
sub verify_sample_builder
{
  my $sample = shift;
  my $generate_cmd;

  if ($os eq "win32")
    {
      $generate_cmd = "$p4_tools_dir/win32/ActivePerl/5.10.0.1004/bin/perl $samples_scripts_dir/generate_builders.pl -info $samples_rootdir/$sample/info.xml -verify $samples_rootdir/$sample";
      if ($^O eq "cygwin")
        {
          if (!$cygwin_toolkit_path)
            {
              $cygwin_toolkit_path = `C:/cygwin/bin/cygpath.exe -w $p4_toolkit_dir`;
              chomp $cygwin_toolkit_path;
              $cygwin_toolkit_path =~ s#\\#/#g;
            }

          if (!$cygwin_samples_root_path)
            {
              $cygwin_samples_root_path = `C:/cygwin/bin/cygpath.exe -w $samples_rootdir`;
              chomp $cygwin_samples_root_path;
              $cygwin_samples_root_path =~ s#\\#/#g;
            }

          $generate_cmd = "$p4_tools_dir/win32/ActivePerl/5.10.0.1004/bin/perl $samples_scripts_dir/generate_builders.pl -info $cygwin_samples_root_path/$sample/info.xml -verify $cygwin_samples_root_path/$sample";
        }
    }
    elsif ($os eq "Linux")
      {
        $generate_cmd = "$linux_perl_path $samples_scripts_dir/generate_builders.pl -info $samples_rootdir/$sample/info.xml -verify $samples_rootdir/$sample";
      }

    if ($generate_cmd)
      {
        return dvsLwdaSamples::exec "$generate_cmd";
      }
    else
      {
        return 0;
      }
}

sub build_sample
{
  my ($sample, $make, $msbuild) = @_;
  my $ret = 0;
  
  $ret = verify_sample($sample);

  if ($os eq "win32")
    {
      $samples_rootdir =~ s#/#\\#g; 
      my $sample_dir = "$samples_rootdir\\$sample";
      my $project = basename("${sample}_vs2010.sln") if ($msvc_version =~ m/vc10/i);
      $project = basename("${sample}_vs2012.sln") if ($msvc_version  =~ m/vc11/i);
      $project = basename("${sample}_vs2019.sln") if ($msvc_version  =~ m/vs2019/i);

      my $project_absolute_path = $sample_dir."\\".$project;;
      if (!(-e $project_absolute_path)) {
        # We need to check if the samples project file exists if it doesn't it means given sample
        # is not supported on provided msvc version so we waive the build of such sample
        print STDERR "Waiving build of $project_absolute_path as the dependencies are not met.\n";
        return 2;
      }

      if ($^O eq "cygwin")
        {
          $project =~ s#\\#/#g;
          $project =~ s#([a-zA-Z])://*#/cygdrive/$1/#g;
          $sample_dir =~ s#\\#/#g;
          $sample_dir =~ s#([a-zA-Z])://*#/cygdrive/$1/#g;
        }
      $ret = dvsLwdaSamples::exec "cd $sample_dir && $msbuild /t:Rebuild $project" if (!$ret);
    }

  if ($os eq "Linux" or $os eq "Darwin")
    {
      my $sample_dir = "$samples_rootdir/$sample";
      $ret = dvsLwdaSamples::exec "$make clean -C $sample_dir" if (!$ret);
      $ret = dvsLwdaSamples::exec "$make build -C $sample_dir" if (!$ret);
    }

  return $ret;
}

sub check_sample_dependencies
{
  my ($sample) = @_;
  my $sample_dir = "$samples_rootdir/$sample";

  if ($os eq "Linux" || $os eq "Darwin")
    {
        # We only need to check the dependencies if the sample has dependencies
        {
          local $/ = undef;
          open MAKEFILE, "$samples_rootdir/$sample/Makefile" or return 0;
          my $make_text = <MAKEFILE>;
          close MAKEFILE;
          return 0 if ($make_text !~ m/check\.deps:/);
        }

      # if the exit code is 0, that means "Sample is ready" was found in the make output
      return dvsLwdaSamples::exec "make check.deps -C $sample_dir | grep \"Sample is ready\"";
    }

  return 0;
}

sub run_sample
{
  my ($sample) = @_;
  my $ilwoke = "";
  my $location = "";
  my $ret = 0;
  my ($elwvars_ref, $ilwocations_ref) = build_sample_ilwocations ($sample);
  my @elwvars = @$elwvars_ref;
  my @ilwocations = @$ilwocations_ref;

  if ($os eq "win32")
    {
      $location = "$samples_rootdir\\..\\bin";
      $location .= "\\win32" if ($target_arch eq "i686");
      $location .= "\\win64" if ($target_arch eq "x86_64");
      $location .= "\\Debug" if ($build eq "debug");
      $location .= "\\Release" if ($build eq "release");

      $ilwoke .= ".\\".basename($sample).".exe";

      my $sample_exe = $location."\\".basename($sample).".exe";
      if (!(-e $sample_exe && -x _)) {
        # We need to check if the samples exelwtable exists if it doesn't it means build failure
        # so we waive the exelwtion of such sample
        print STDERR "Waiving exelwtion of $sample_exe as the dependencies are not met.\n";
        return 2;
      }
    }


  if ($os eq "Linux")
    {
      $location = "$samples_rootdir/../bin/$target_arch/linux";
      $location .= "/debug" if ($build eq "debug");
      $location .= "/release" if ($build eq "release");
      $ilwoke = join(' ', @elwvars) . " ./".basename($sample);
    }

  if ($os eq "Darwin")
    {
      $location = "$samples_rootdir/../bin/i686" if ($target_arch eq "i386");
      $location = "$samples_rootdir/../bin/x86_64" if ($target_arch eq "x86_64");
      $location .= "/darwin";
      $location .= "/debug" if ($build eq "debug");
      $location .= "/release" if ($build eq "release");
      $ilwoke = join(' ', @elwvars) . " ./".basename($sample);
    }

  if ($os eq "Linux" || $os eq "Darwin")
    {

        my $sample_exe = join "", $location, "/", basename($sample);
        if (!(-e $sample_exe && -x _)) {
          # We need to check if the samples exelwtable exists if it doesn't it means build failure
          # so we waive the exelwtion of such sample
          print STDERR "Waiving exelwtion of $sample_exe as the dependencies are not met.\n";
          return 2;
        }
    }

  if ($^O eq "cygwin")
    {
      $location =~ s#\\#/#g if ($^O eq "cygwin");
      $ilwoke =~ s#\\#/#g if ($^O eq "cygwin");
    } 

  foreach my $invocation (@ilwocations)
    {
      $ret |= dvsLwdaSamples::exec "cd $location && $ilwoke $invocation";
    }

  return $ret;
}

sub install_files
{
  my ($src, $dst) = @_;
  my @dir_paths = ();

  $src =~ s#\\#/#g;
  $dst =~ s#\\#/#g;

  print STDERR "Copy $src to $dst\n" if $debug;
  my @files = glob($src);

  foreach my $file_path (@files)
    {
      my $file_name = basename ($file_path);

      if (-d $file_path)
        {
          push @dir_paths, $file_path;
        }
      else
        {
          my $dst_path;
          $dst_path = "$dst/$file_name" if -d $dst;
          $dst_path = "$dst" if not -d $dst;

          unlink "$dst_path" if -e "$dst_path";
          install_files_win32 ("$file_path", "$dst_path") if ($os eq "win32");
          install_files_linux ("$file_path", "$dst_path") if ($os eq "Linux" or $os eq "Darwin");
          chmod 0755, "$dst_path";
        }
    }

  foreach my $dir_path (@dir_paths)
    {
      my $dir_name = basename ($dir_path);
      mkpath ("$dst/$dir_name");
      install_files ("$dir_path/*", "$dst/$dir_name");
    }
}

sub copy_header_files
{
  my ($keywords_include, $keywords_exclude) = @_;
  my $bin_path = "$toolkit_rootdir/bin/${target_arch}_${os}${abi}_${build}";

  
  my $all = (scalar keys %{$keywords_include} == 0);
  my $lwrand   = ($all || exists ${$keywords_include}{"LWRAND"})   && !exists ${$keywords_exclude}{"LWRAND"};
  my $lwblas   = ($all || exists ${$keywords_include}{"LWBLAS"})   && !exists ${$keywords_exclude}{"LWBLAS"};
  my $lwfft    = ($all || exists ${$keywords_include}{"LWFFT"})    && !exists ${$keywords_exclude}{"LWFFT"};
  my $lwsparse = ($all || exists ${$keywords_include}{"LWSPARSE"}) && !exists ${$keywords_exclude}{"LWSPARSE"};
  my $lwsolver = ($all || exists ${$keywords_include}{"LWSOLVER"}) && !exists ${$keywords_exclude}{"LWSOLVER"};
  my $npp      = ($all || exists ${$keywords_include}{"NPP"})      && !exists ${$keywords_exclude}{"NPP"};
  my $lwjpeg   = ($all || exists ${$keywords_include}{"LWJPEG"})   && !exists ${$keywords_exclude}{"LWJPEG"};
  my $misc     = ($all || (!$lwrand && !$lwblas && !$lwfft && !$lwsparse && !$npp && !$lwsolver && !$lwjpeg));

  start_timer("Header Files");
  print STDERR "\nCOPY HEADER FILES\n\n" if $debug;

  if ($os eq "Linux" or $os eq "Darwin")
    {
      unlink  "$toolkit_rootdir/bin/bin";
      symlink $bin_path, "$toolkit_rootdir/bin/bin";
      unlink  "$toolkit_rootdir/bin/lib";
      symlink $bin_path, "$toolkit_rootdir/bin/lib";
    }

  mkpath ("$bin_path");
  mkpath ("$bin_path/thrust");
  mkpath ("$bin_path/lwb");
  mkpath ("$bin_path/lwca");
  mkpath ("$bin_path/lwca/std");
  mkpath ("$bin_path/lwca/std/detail");
  mkpath ("$bin_path/lwca/std/detail/libcxx");
  mkpath ("$bin_path/lwca/std/detail/libcxx/include");

  install_files ("$bin_path/stub/*.h", $bin_path);
  install_files ("$bin_path/include/*", $bin_path);
  install_files ("$toolkit_rootdir/lwca/tools/cooperative_groups/*", $bin_path);
  install_files ("$toolkit_rootdir/lwca/tools/liblwdacxxext/lwca/*", "$bin_path/lwca");
  install_files ("$toolkit_rootdir/liblwdacxx/include/*", $bin_path);
  install_files ("$toolkit_rootdir/liblwdacxx/libcxx/include/*", "$bin_path/lwca/std/detail/libcxx/include");

  install_files ("$toolkit_rootdir/lwrand/src/*.h", $bin_path) if $lwrand;
  install_files ("$toolkit_rootdir/lwblas/src/*.h", $bin_path) if $lwblas || $lwsolver;
  install_files ("$toolkit_rootdir/lwblas/lwblasMg/*.h", $bin_path) if $lwblas || $lwsolver;
  install_files ("$toolkit_rootdir/lwfft/branches/TOT/liblwfft/*.h", $bin_path) if $lwfft;
  install_files ("$toolkit_rootdir/lwsparse/branches/master/include/*.h", $bin_path) if $lwsparse || $lwsolver;
  install_files ("$toolkit_rootdir/lwSolver/src/lwdense/*.h", $bin_path) if $lwsolver;  # for lwsolverDn.h
  install_files ("$toolkit_rootdir/lwSolver/src/glu/*.h", $bin_path) if $lwsolver;      # for lwsolverRf.h
  install_files ("$toolkit_rootdir/lwSolver/src/pegasus/*.h", $bin_path) if $lwsolver;  # for lwsolverSp.h
  install_files ("$toolkit_rootdir/NPP/npp/include/*.h", $bin_path) if $npp;
  install_files ("$toolkit_rootdir/lwJPEG/branches/master/include/lwjpeg.h", $bin_path) if $lwjpeg;
  install_files ("$toolkit_rootdir/thrust/thrust/*", "$bin_path/thrust") if $misc or $npp;
  install_files ("$toolkit_rootdir/thrust/lwb/*", "$bin_path/lwb") if $misc or $npp;


  # Add EGL headers
  mkpath ("$bin_path/EGL");
  install_files ("$toolkit_rootdir/lwca/import/EGL/*.h", "$bin_path/EGL");

  stop_timer("Header Files");
}

sub copy_library_files
{
  my ($keywords_include, $keywords_exclude) = @_;
  my $bin_path = "$toolkit_rootdir/bin/${target_arch}_${os}${abi}_${build}";

  my $all = (scalar keys %{$keywords_include} == 0);
  my $lwrand   = ($all || exists ${$keywords_include}{"LWRAND"})   && !exists ${$keywords_exclude}{"LWRAND"};
  my $lwblas   = ($all || exists ${$keywords_include}{"LWBLAS"})   && !exists ${$keywords_exclude}{"LWBLAS"};
  my $lwfft    = ($all || exists ${$keywords_include}{"LWFFT"})    && !exists ${$keywords_exclude}{"LWFFT"};
  my $lwsparse = ($all || exists ${$keywords_include}{"LWSPARSE"}) && !exists ${$keywords_exclude}{"LWSPARSE"};
  my $lwsolver = ($all || exists ${$keywords_include}{"LWSOLVER"}) && !exists ${$keywords_exclude}{"LWSOLVER"};
  my $npp      = ($all || exists ${$keywords_include}{"NPP"})      && !exists ${$keywords_exclude}{"NPP"};
  my $lwjpeg   = ($all || exists ${$keywords_include}{"LWJPEG"})   && !exists ${$keywords_exclude}{"LWJPEG"};
  my $misc     = ($all || (!$lwrand && !$lwblas && !$lwfft && !$lwsparse && !$npp && !$lwsolver && !$lwjpeg));

  start_timer("Library Files");
  print STDERR "\nCOPY LIBRARY FILES\n\n" if $debug;

  mkpath ("$bin_path");

  stop_timer("Library Files");
}

sub set_commands_docker
{
  my ($make, $lwmake, $msbuild) = @_;

  if ($os eq "Linux" || $os eq "Darwin")
    {
      $$make = "/usr/bin/make ";

      $$make .= "dbg=1 " if $build eq "debug";

      $$make .= "TARGET_ARCH=x86_64 " if $target_arch eq "x86_64";
      $$make .= "TARGET_ARCH=ARMv7 "  if $target_arch eq "ARMv7";
      $$make .= "TARGET_ARCH=aarch64 "  if $target_arch eq "aarch64";
      $$make .= "TARGET_ARCH=ppc64le "  if $target_arch eq "ppc64le";

      my $driver_bin_path = "${driver_rootdir}/bin/${target_arch}_${os}${abi}_${build}";
      my $driver_phat_path = "$driver_rootdir/bin/phat_${os}_$build";
      my $package_bin_path = "$toolkit_rootdir/built/${target_arch}_${os}${abi}_${build}";

      my $extra_ldflags = "";
      $extra_ldflags .= "\"";

      if ($os eq "Darwin")
      {
        $extra_ldflags .= "-F${driver_bin_path} " if ($mode eq "qa");
        $extra_ldflags .= "-F${driver_phat_path} " if ($mode ne "qa");
        $extra_ldflags .= "-L${driver_bin_path} " if ($mode eq "qa");
        $extra_ldflags .= "-L${driver_phat_path} " if ($mode ne "qa");
        $extra_ldflags .= "-L${package_bin_path} ";
      }

      if ($os eq "Linux")
      {
        $extra_ldflags .= "-L${driver_bin_path} ";
        $extra_ldflags .= "-L${package_bin_path} ";
      }

      $extra_ldflags .= "\"";

      $$make .= "LWDA_PATH=$toolkit_rootdir/bin ";
      $$make .= "LWDA_SEARCH_PATH=$toolkit_rootdir/bin/${target_arch}_${os}${abi}_${build} ";
      $$make .= "EXTRA_LDFLAGS=${extra_ldflags} ";
    }

  if ($os eq "win32")
    {
      my $win_arch;
      $win_arch = "x64"  if $target_arch eq "x86_64";

      my $win_build;
      $win_build = "Debug"   if $build eq "debug";
      $win_build = "Release" if $build eq "release";

      if ($^O eq "MSWin32") {
          $$lwmake = "$p4_toolkit_dir\\build\\lwmake.exe";
      }

      $$msbuild = "C:\\BuildTools\\MSBuild\\Current\\Bin\\amd64\\MSBuild.exe ";
      $$msbuild .= " /m:2 "; # use 2 cpu threads
      $$msbuild .= " /p:UseElw=false ";
      $$msbuild .= " /p:Configuration=$win_build ";
      $$msbuild .= " /p:Platform=$win_arch ";
      $$msbuild .= " /p:PROCESSOR_ARCHITECTURE=AMD64 " if $target_arch eq "x86_64";

      # Turn of incremental builds. There is a Microsoft bug with it on win64. It
      # has no impact since we always clobber before building. Makes the
      # following message disappear: TRACKER : error TRK0002: Failed to execute
      # command: ".../mt.exe ...". The handle is invalid.
      $$msbuild .= " /p:TrackFileAccess=false ";
    }
}

sub set_commands
{
  my ($make, $lwmake, $msbuild) = @_;

  if ($os eq "Linux" || $os eq "Darwin")
    {
      $$make = "/usr/bin/make ";

      $$make .= "dbg=1 " if $build eq "debug";

      $$make .= "i386=1 "   if $target_arch eq "i386" || $target_arch eq "i686";
      $$make .= "x86_64=1 " if $target_arch eq "x86_64";
      $$make .= "ARMv7=1 "  if $target_arch eq "ARMv7";
      $$make .= "aarch64=1 "  if $target_arch eq "aarch64";
      $$make .= "ppc64le=1 "  if $target_arch eq "ppc64le";
    }

  if ($os eq "win32")
    {
      my $win_arch;
      $win_arch = "Win32" if $target_arch eq "i686";
      $win_arch = "x64"  if $target_arch eq "x86_64";

      my $win_build;
      $win_build = "Debug"   if $build eq "debug";
      $win_build = "Release" if $build eq "release";

      my $vc_install_dir;
      my $windows_sdk_dir;
      if ($^O eq "cygwin") {
          $vc_install_dir = "/cygwin/$p4_tools_dir/win32/msvc100sp1/VC/";
          $vc_install_dir =~ s#\\#/#g;
          $vc_install_dir =~ s#//*#/#g;
          $windows_sdk_dir = "C:/Program Files/Microsoft SDKs/Windows/v7.1";

          $$lwmake = "$p4_toolkit_dir/build/lwmake.exe";
      } elsif ($^O eq "MSWin32") {
          if ($environment eq "rel" && $vulcan_toolkit_dir ne "")
          {
            $windows_sdk_dir = "$toolkit_rootdir\\windsdk\\7.1" if ($msvc_version =~ m/vc10/i);
            $windows_sdk_dir = "$toolkit_rootdir\\windsdk\\8.1" if ($msvc_version =~ m/vc11/i);

            $vc_install_dir = "$p4_tools_dir\\msvc100sp1\\VC\\" if ($msvc_version =~ m/vc10/i);
            $vc_install_dir = "$p4_tools_dir\\msvc110\\VC\\" if ($msvc_version =~ m/vc11/i);
          }
          else
          {
            $windows_sdk_dir = "$p4_tools_dir\\sdk\\WinSDK\\7.1" if ($msvc_version =~ m/vc10/i);
            $windows_sdk_dir = "$p4_tools_dir\\sdk\\WinSDK\\8.1" if ($msvc_version =~ m/vc11/i);

            $vc_install_dir = "$p4_tools_dir\\win32\\msvc100sp1\\VC\\" if ($msvc_version =~ m/vc10/i);
            $vc_install_dir = "$p4_tools_dir\\win32\\msvc110\\VC\\" if ($msvc_version =~ m/vc11/i);
          }
          $vc_install_dir =~ s#/#\\#g;
          $vc_install_dir =~ s#\\\\*#\\#g;

          $$lwmake = "$p4_toolkit_dir\\build\\lwmake.exe";
      }

      $$msbuild = "$ELW{'MSBuildToolsPath'}\\MSBuild.exe ";
      $$msbuild = "$ELW{'MSBuildToolsPath'}\\MSBuild.exe " if ($vulcan_toolkit_dir ne "");
      $$msbuild .= " /p:UseElw=true ";
      $$msbuild .= " /p:Configuration=$win_build ";
      $$msbuild .= " /p:Platform=$win_arch ";
      $$msbuild .= " /p:PROCESSOR_ARCHITECTURE=AMD64 " if $target_arch eq "x86_64";
      $$msbuild .= " /p:VCInstallDir=$vc_install_dir ";
      $$msbuild .= " /p:WindowsSdkDir=\"$windows_sdk_dir\" ";

      if ($msvc_version  =~ m/vc11/i)
      {
        $$msbuild .= " /p:VisualStudioVersion=11.0 ";
        $$msbuild .= " /p:VCTargetsPath=$p4_tools_dir\\sdk\\dotNet\\4.0\\MSBuild\\Microsoft.Cpp\\v4.0\\v110\\ ";
        $$msbuild .= " /p:VCTargetsPath11=$p4_tools_dir\\sdk\\dotNet\\4.0\\MSBuild\\Microsoft.Cpp\\v4.0\\v110\\ ";
      }
      if ($msvc_version  =~ m/vc10/i)
      {
        $$msbuild .= " /p:VCTargetsPath=$p4_tools_dir\\sdk\\dotNet\\4.0\\MSBuild\\Microsoft.Cpp\\v4.0\\ ";
        $$msbuild .= " /p:VCTargetsPath10=$p4_tools_dir\\sdk\\dotNet\\4.0\\MSBuild\\Microsoft.Cpp\\v4.0\\ ";
      }

      $$msbuild .= " /p:FrameworkPathOverride=$p4_tools_dir\\sdk\\dotNet\\4.0\\Reference_assemblies\\v4.0 "; 
      $$msbuild .= " /p:MSBuildOverrideTasksPath=$p4_tools_dir\\sdk\\dotNet\\4.0\\Framework64\\v4.0.30319 ";

      # Turn of incremental builds. There is a Microsoft bug with it on win64. It
      # has no impact since we always clobber before building. Makes the
      # following message disappear: TRACKER : error TRK0002: Failed to execute
      # command: ".../mt.exe ...". The handle is invalid.
      $$msbuild .= " /p:TrackFileAccess=false ";
    }
    
  if ($environment eq "dev")
    {
      if ($os eq "Linux" || $os eq "Darwin")
      {
        my $driver_bin_path = "${driver_rootdir}/bin/${target_arch}_${os}${abi}_${build}";
        my $driver_phat_path = "$driver_rootdir/bin/phat_${os}_$build";
        my $package_bin_path = "$toolkit_rootdir/built/${target_arch}_${os}${abi}_${build}";
    
        my $extra_ldflags = "";
        $extra_ldflags .= "\"";

        if ($os eq "Darwin")
        {
          $extra_ldflags .= "-F${driver_bin_path} " if ($mode eq "qa");
          $extra_ldflags .= "-F${driver_phat_path} " if ($mode ne "qa");
          $extra_ldflags .= "-L${driver_bin_path} " if ($mode eq "qa");
          $extra_ldflags .= "-L${driver_phat_path} " if ($mode ne "qa");
          $extra_ldflags .= "-L${package_bin_path} ";
        }

        if ($os eq "Linux")
        {
          $extra_ldflags .= "-L${driver_bin_path} ";
          $extra_ldflags .= "-L${package_bin_path} ";
        }

        $extra_ldflags .= "\""; 

        $$make .= "LWDA_PATH=$toolkit_rootdir/bin ";
        $$make .= "LWDA_SEARCH_PATH=$toolkit_rootdir/bin/${target_arch}_${os}${abi}_${build} ";
        $$make .= "EXTRA_LDFLAGS=${extra_ldflags} ";
      }
    }

  if ($debug) {
      print "\n";
      print "COMMANDS:\n";
      print "  make = $$make\n";
      print "  msbuild = $$msbuild\n";
      print "  lwmake = $$lwmake\n";
      print "\n";
  }
}

sub set_elwironment
{
  if ($os eq "win32") {
    set_elwironment_win32 ($host_arch, $target_arch, $os, $build, $environment, $toolkit_rootdir, $driver_rootdir, $p4_tools_dir, $p4_toolkit_dir, $vulcan_toolkit_dir, $msvc_version, $debug);
  } else {
    set_elwironment_linux ($host_arch, $target_arch, $os, $abi, $build, $environment, $toolkit_rootdir, $driver_rootdir, $debug);
  }
}

sub set_elwironment_docker
{
  if ($os eq "win32") {
    set_elwironment_win32_docker ($host_arch, $target_arch, $os, $build, $environment, $toolkit_rootdir, $driver_rootdir, $p4_toolkit_dir, $debug);
  }
}

sub set_paths
{
  my ($mode) = @_;

  if ($os eq "win32") {
    set_paths_win32 ($mode, $target_arch, $os, $build, $environment, $toolkit_rootdir,
                     $driver_rootdir, $debug);
  } else {
    set_paths_linux ($mode, $target_arch, $os, $abi, $build, $environment, $toolkit_rootdir,
                     $driver_rootdir, $p4_toolkit_dir, $debug);
  }
}

sub copy_msvc_build_extensions
{
  if ($os eq "win32")
    {
      my @versions = ("10.0", "10.1", "10.2", "11.0", "11.1", "11.2");
      my $dst = $ELW{'LWDAPropsPath'};
      $dst =~ s#\\#/#g;
      $dst =~ s#^/cygwin##;

      foreach my $version (@versions)
        {
          my $src = "$p4_build_rootdir/sw/devtools/Nexus/Exports/LwdaToolkit/v$version";
          $src =~ s#\\#/#g;
          $src =~ s#^//*#/#;

          mkpath ($dst);
          install_files ("$src/MsBuildExtensions/*", $dst);
          install_files ("$src/BuildLwstomizationTasks/*", $dst);
        }
    }
}

sub copy_installation
{
  print STDERR "\nCOPY INSTALLATION\n\n" if $debug;

  if ($environment eq "rel")
    {
      print STDERR "Error: copy_installation() should not be called for release environment testing.\n";
      exit 1;
    }

  if ($os eq "win32")
    {
      $toolkit_rootdir =~ s#\\#/#g;

      my $toolkit_bin_path = "$toolkit_rootdir/bin/${target_arch}_${os}${abi}_${build}";
      my $local_path = "$toolkit_bin_path/local";

      my $bin_path = "$samples_rootdir/bin/win32/Release";
      $bin_path =~ s/win32/win64/ if $target_arch eq "x86_64";
      $bin_path =~ s/Release/Debug/ if $build eq "debug";

      mkpath ("$local_path");
      mkpath ("$local_path/bin");
      mkpath ("$local_path/bin/crt");
      mkpath ("$local_path/include");
      mkpath ("$local_path/include/crt");
      mkpath ("$local_path/include/thrust");
      mkpath ("$local_path/include/lwb");
      mkpath ("$local_path/include/cooperative_groups");
      mkpath ("$local_path/include/lwca");
      mkpath ("$local_path/libcxx");
      mkpath ("$local_path/libcxx/include");
      mkpath ("$local_path/lib");
      mkpath ("$local_path/lib64");
      mkpath ("$local_path/open64");
      mkpath ("$local_path/open64/bin");
      mkpath ("$local_path/open64/lib");
      mkpath ("$local_path/lwvm");
      mkpath ("$bin_path");

      install_files ("$toolkit_bin_path/*.exe", "$local_path/bin");
      install_files ("$toolkit_bin_path/lwvm/*", "$local_path/lwvm");
      install_files ("$toolkit_bin_path/open64/bin/*", "$local_path/open64/bin");
      install_files ("$toolkit_bin_path/open64/lib/*", "$local_path/open64/lib");
      install_files ("$toolkit_rootdir/pkg_specs/lwcc.profile.pkg.win32", "$local_path/bin/lwcc.profile");
      install_files ("$toolkit_bin_path/crt/*.stub", "$local_path/bin/crt");
      install_files ("$toolkit_bin_path/*.h", "$local_path/include");
      install_files ("$toolkit_bin_path/*.hpp", "$local_path/include");
      install_files ("$toolkit_bin_path/crt/*.h", "$local_path/include/crt");
      install_files ("$toolkit_bin_path/crt/*.hpp", "$local_path/include/crt");
      install_files ("$toolkit_bin_path/crt/lwfunctional", "$local_path/include/crt");
      install_files ("$toolkit_bin_path/thrust/*", "$local_path/include/thrust");
      install_files ("$toolkit_bin_path/lwb/*", "$local_path/include/lwb");
      install_files ("$toolkit_bin_path/cooperative_groups/*", "$local_path/include/cooperative_groups");
      install_files ("$toolkit_bin_path/lwca/*", "$local_path/include/lwca");

      install_files ("$toolkit_bin_path/*.lib", "$local_path/lib") if ($target_arch eq "i686");
      install_files ("$toolkit_bin_path/*.lib", "$local_path/lib64") if ($target_arch eq "x86_64");
      # Installing LWRTC stuff in $local_path for samples BUILD purpose.
      install_files ("$toolkit_bin_path/lwrtc/lib/x64/*", "$local_path/lib64") if ($target_arch eq "x86_64");
      install_files ("$toolkit_bin_path/lwrtc/include/*", "$local_path/include") if ($target_arch eq "x86_64");
      install_files ("$toolkit_bin_path/lwrtc/bin/*", "$local_path/bin") if ($target_arch eq "x86_64");
      # Installing LWRTC stuff in $toolkit_bin_path for samples RUN purpose, as LWDA_PATH_V7_0 is set to $toolkit_bin_path.
      install_files ("$toolkit_bin_path/lwrtc/lib/x64/*", "$toolkit_bin_path") if ($target_arch eq "x86_64");
      install_files ("$toolkit_bin_path/lwrtc/include/*", "$toolkit_bin_path") if ($target_arch eq "x86_64");
      install_files ("$toolkit_bin_path/lwrtc/bin/*", "$toolkit_bin_path") if ($target_arch eq "x86_64");

      install_files ("$toolkit_bin_path/*.dll", "$bin_path");

      # FreeImage related files
      mkpath ("$samples_rootdir/../Common/FreeImage/Dist/x64");
      install_files ("$samples_rootdir/../scripts/dvs/test_requisites/FreeImage*", "$samples_rootdir/../Common/FreeImage/Dist/x64");
      install_files ("$samples_rootdir/../scripts/dvs/test_requisites/x86_64_Windows/FreeImage*", "$samples_rootdir/../Common/FreeImage/Dist/x64");
    }
    elsif ( $os eq "Linux" or $os eq "Darwin")
    {
      my $toolkit_bin_path = "$toolkit_rootdir/bin/${target_arch}_${os}${abi}_${build}";

      my $lib_dir = "lib64";
      $lib_dir = "lib" if ($os eq "Darwin");

      install_files ("$toolkit_bin_path/lwrtc/$lib_dir/*", "$toolkit_bin_path") if ($target_arch eq "x86_64");
      install_files ("$toolkit_bin_path/lwrtc/include/*", "$toolkit_bin_path") if ($target_arch eq "x86_64");
      install_files ("$toolkit_bin_path/lwrtc/bin/*", "$toolkit_bin_path") if ($target_arch eq "x86_64");

      dvsLwdaSamples::exec "ln -sf $toolkit_bin_path $toolkit_rootdir/bin/include";
      dvsLwdaSamples::exec "ln -sf $toolkit_bin_path $toolkit_rootdir/bin/$lib_dir";
    }
}

sub set_compression_commands
{
  $unzip = "${p4_build_rootdir}\\sw\\tools\\win32\\infozip3\\unzip.exe";

  if ($os eq "win32")
  {
      $pkg_cmd .= "cd $toolkit_rootdir && ";
      $pkg_cmd .= "${p4_build_rootdir}\\sw\\tools\\win32\\infozip3\\zip.exe ";
      #$pkg_cmd .= "-0 ";   # zero compression
      $pkg_cmd .= "-u ";   # update, not create
      $pkg_cmd .= "-r ";   # relwrsive
      $pkg_cmd .= "-o ";   # latest time stamps
      $pkg_cmd .= "-q ";   # quiet output
      $pkg_cmd .= "$package ";

      $delete_pkg_cmd .= "del $package >null 2>&1";
  }

  if ($os eq "Linux" or $os eq "Darwin")
  {
      my $pkg = $package;
      $pkg =~ s/.*(LWCA-samples-.*)\.tar\.bz2/$1/;
      my $rm = "rm";
      $pkg_cmd .= "tar ";
      $pkg_cmd .= "--append ";
      $pkg_cmd .= "--file $package ";
      $pkg_cmd .= "--directory $toolkit_rootdir ";

      $delete_pkg_cmd .= "rm -f $package"
  }

  if ($debug) {
      print "\n";
      print "COMMANDS:\n";
      print "  pkg_cmd = $pkg_cmd\n";
      print "  delete_pkg_cmd = $delete_pkg_cmd\n";
      print "\n";
  }
}


sub build_libraries
{
  my ($make, $lwmake) = @_;

  start_timer("LWCA RUNTIME");

  if ($os eq "win32")
    {
      dvsLwdaSamples::exec "$lwmake -C $toolkit_rootdir\\lwca release PAR=4 HOST_ARCH=$host_arch TARGET_ARCH=$target_arch GPGPU_COMPILER_EXPORT=$compiler_export";
    }
  else
    {
      dvsLwdaSamples::exec "$make -s -C $toolkit_rootdir/lwca RELEASE=1 PAR=4 HOST_ARCH=$host_arch TARGET_ARCH=$target_arch GPGPU_COMPILER_EXPORT=$compiler_export";
    }

  stop_timer("LWCA RUNTIME");
}

sub merge_required_packages
{
  mkpath ("$toolkit_rootdir/built", "$toolkit_rootdir/bin");

  if ($os eq "Linux" or $os eq "Darwin")
    {
      my $tgt_dir = "$toolkit_rootdir";
      my $dst = "$tgt_dir/bin/${target_arch}_${os}${abi}_${build}";
      my $package_dir = "$toolkit_rootdir/build";

      mkpath ($dst);

      if (-e "${package_dir}/LWCA-lwblas-package.tar.bz2") {
        start_timer("LWBLAS");
        dvsLwdaSamples::exec "tar --extract --bzip2 --directory $tgt_dir --file ${package_dir}/LWCA-lwblas-package.tar.bz2";
        stop_timer("LWBLAS");
      }
    
      if (-e "${package_dir}/LWCA-lwfft-package.tar.bz2") {
        start_timer("LWFFT");
        dvsLwdaSamples::exec "tar --extract --bzip2 --directory $tgt_dir --file ${package_dir}/LWCA-lwfft-package.tar.bz2";
        stop_timer("LWFFT");
      }
      
      if (-e "${package_dir}/LWCA-lwrand-package.tar.bz2") {
        start_timer("LWRAND");
        dvsLwdaSamples::exec "tar --extract --bzip2 --directory $tgt_dir --file ${package_dir}/LWCA-lwrand-package.tar.bz2";
        stop_timer("LWRAND");
      }
      
      if (-e "${package_dir}/LWCA-lwsparse-package.tar.bz2") {
        start_timer("LWSPARSE");
        dvsLwdaSamples::exec "tar --extract --bzip2 --directory $tgt_dir --file ${package_dir}/LWCA-lwsparse-package.tar.bz2";
        stop_timer("LWSPARSE");
      }
      
      if (-e "${package_dir}/LWCA-lwsolver-package.tar.bz2") {
        start_timer("LWSOLVER");
        dvsLwdaSamples::exec "tar --extract --bzip2 --directory $tgt_dir --file ${package_dir}/LWCA-lwsolver-package.tar.bz2";
        stop_timer("LWSOLVER");
      }
    
      if (-e "${package_dir}/LWCA-NPP-package.tar.bz2") {
        start_timer("NPP");
        dvsLwdaSamples::exec "tar --extract --bzip2 --directory $tgt_dir --file ${package_dir}/LWCA-NPP-package.tar.bz2";
        stop_timer("NPP");
      }

      if (-e "${package_dir}/LWCA-lwJPEG-package.tar.bz2") {
        start_timer("LWJPEG");
        dvsLwdaSamples::exec "tar --extract --bzip2 --directory $tgt_dir --file ${package_dir}/LWCA-lwJPEG-package.tar.bz2";
        stop_timer("LWJPEG");
      }

      start_timer("LIBS");
      install_files ("$toolkit_rootdir/bin/${target_arch}_${os}_${build}/stub/liblwda.so", "$dst");
      dvsLwdaSamples::exec "ln -sf $dst/liblwda.so $dst/liblwda.so.1";
      stop_timer("LIBS");
    }

  if ($os eq "win32")
    {
      my $tgt_dir = "$toolkit_rootdir";
      my $dst = "$tgt_dir/bin/${target_arch}_${os}${abi}_${build}";
      my $package_dir = "$toolkit_rootdir\\build";

      mkpath ($dst);

      if (-e "${package_dir}\\LWCA-lwblas-package.zip") {
        start_timer("LWBLAS");
        dvsLwdaSamples::exec "$unzip -o -d $tgt_dir ${package_dir}\\LWCA-lwblas-package.zip";
        stop_timer("LWBLAS");
      }

      if (-e "${package_dir}\\LWCA-lwfft-package.zip") {
        start_timer("LWFFT");
        dvsLwdaSamples::exec "$unzip -o -d $tgt_dir ${package_dir}\\LWCA-lwfft-package.zip";
        stop_timer("LWFFT");
      }

      if (-e "${package_dir}\\LWCA-lwrand-package.zip") {
        start_timer("LWRAND");
        dvsLwdaSamples::exec "$unzip -o -d $tgt_dir ${package_dir}\\LWCA-lwrand-package.zip";
        stop_timer("LWRAND");
      }

      if (-e "${package_dir}\\LWCA-lwsparse-package.zip") {
        start_timer("LWSPARSE");
        dvsLwdaSamples::exec "$unzip -o -d $tgt_dir ${package_dir}\\LWCA-lwsparse-package.zip";
        stop_timer("LWSPARSE");
      }

      if (-e "${package_dir}\\LWCA-lwsolver-package.zip") {
        start_timer("LWSOLVER");
        dvsLwdaSamples::exec "$unzip -o -d $tgt_dir ${package_dir}\\LWCA-lwsolver-package.zip";
        stop_timer("LWSOLVER");
      }

      if (-e "${package_dir}\\LWCA-NPP-package.zip") {
        start_timer("NPP");
        dvsLwdaSamples::exec "$unzip -o -d $tgt_dir ${package_dir}\\LWCA-NPP-package.zip";
        stop_timer("NPP");
      }

      if (-e "${package_dir}\\LWCA-lwJPEG-package.zip") {
        start_timer("LWJPEG");
        dvsLwdaSamples::exec "$unzip -o -d $tgt_dir ${package_dir}\\LWCA-lwJPEG-package.zip";
        stop_timer("LWJPEG");
      }


      start_timer("LIBS");
      my $arch_dir = "";
      $arch_dir = "/x64" if $target_arch eq "x86_64";
      install_files ("$toolkit_rootdir/bin/${target_arch}_${os}_${build}/stub/lwca.lib", "$dst");
      stop_timer("LIBS");
    }
}

sub clobber_samples
{
  my ($make, $msbuild, $lwmake) = @_;
  my $ret = 1;

  my $driver_root = $driver_rootdir;
  start_timer("Clobber LWCA Samples");
  
  if (!$skip_build_files_generation)
  {
    if ($os eq "Linux")
    {
      $driver_root =~ s|/drivers/gpgpu$||;
      $driver_root =~ s|/drivers/gpgpu/$||;
      dvsLwdaSamples::exec "$make -C $samples_scripts_dir -f $samples_scripts_dir/createmk.mk DRIVER_ROOT=$driver_root USE_P4=1 GEN_BUILD_FILE=\"makefile\" PERL=$linux_perl_path";
    }
    if ($os eq "Darwin")
    {
      $driver_root =~ s|/drivers/gpgpu$||;
      $driver_root =~ s|/drivers/gpgpu/$||;
      dvsLwdaSamples::exec "/usr/bin/make -C $samples_scripts_dir -f $samples_scripts_dir/mac.mk DRIVER_ROOT=$driver_root USE_P4=1 GEN_BUILD_FILE=\"makefile\"";
    }
    if ($os eq "win32")
    {
       if ($^O eq "cygwin")
       {
          my $user_path = $ELW{'PATH'};

          $driver_root =~ s|/drivers/gpgpu$||;
          $driver_root =~ s|/drivers/gpgpu/$||;
          $driver_root = `C:/cygwin/bin/cygpath.exe -w $driver_root`;
          chomp $driver_root;
          $driver_root =~ s#\\#/#g;

          my $samples_rootdir_cygpath = `C:/cygwin/bin/cygpath.exe -w $samples_rootdir`;
          chomp $samples_rootdir_cygpath;
          $samples_rootdir_cygpath =~ s#\\#/#g;

          my $perl_path = "$p4_tools_dir/win32/ActivePerl/5.10.0.1004/bin/perl";
          $perl_path = `C:/cygwin/bin/cygpath.exe -w $perl_path`; 
          chomp $perl_path;
          $perl_path =~ s#\\#/#g;

          # Several paths breaks lwmake.exe while running under cygwin, See bug 200036239 for more details
          $ELW{'PATH'} = "";
          dvsLwdaSamples::exec "$lwmake -C $samples_rootdir_cygpath/scripts -f $samples_rootdir_cygpath//scripts/windows.mk DRIVER_ROOT=$driver_root PERL=$perl_path";
          $ELW{'PATH'} = $user_path;
       }
       else
       {
          $driver_root =~ s|\\drivers\\gpgpu$||;
          $driver_root =~ s|\\drivers\\gpgpu\\$||;
          dvsLwdaSamples::exec "$lwmake -C $samples_scripts_dir -f $samples_scripts_dir\\windows.mk DRIVER_ROOT=$driver_root PERL=$p4_tools_dir\\win32\\ActivePerl\\5.10.0.1004\\bin\\perl";
       }
    }
  }
  if ($os eq "Linux" or $os eq "Darwin")
    {
      $ret = dvsLwdaSamples::exec "$make clean -k -C $samples_rootdir -j $parallel" ;
    }

  if ($os eq "win32")
    {
      $samples_rootdir =~ s#/#\\#g; 
      my $project = basename("Samples_vs2010.sln");
      $project = basename("Samples_vs2012.sln") if ($msvc_version eq "vc11");
      my $max_cpu_count = $parallel - 1;
      my $processor_number = $parallel;
      my $targets = "Clean";
      $targets =~ s#/#\\#g; 
 
      if ($^O eq "cygwin")
        {
          $project =~ s#\\#/#g;
          $project =~ s#([a-zA-Z])://*#/cygdrive/$1/#g;
          $samples_rootdir =~ s#\\#/#g;
          $samples_rootdir =~ s#([a-zA-Z])://*#/cygdrive/$1/#g;
        }

      my $cmd = "cd $samples_rootdir && ";
      $cmd .= "$msbuild";
      $cmd .= "/m:$max_cpu_count /p:ProcessorNumber=$processor_number " if $parallel > 1;
      $cmd .= "/t:$targets $project";

      $ret = dvsLwdaSamples::exec $cmd;
    }
  
  # A build error will show up as a make exit code 2. Needs to be colwerted to 1.
  $ret = 1 if $ret > 0; 

  stop_timer("Clobber LWCA Samples");
}

sub build_sample_docker
{
  my ($sample, $make, $msbuild) = @_;
  my $ret = 0;

  $ret = verify_sample($sample);

  if ($os eq "win32")
    {
      $samples_rootdir =~ s#/#\\#g;
      my $sample_dir = "$samples_rootdir\\$sample";
      my $project = basename("${sample}_vs2019.sln") if ($msvc_version =~ m/VS2019/i);

      my $project_absolute_path = $sample_dir."\\".$project;;
      if (!(-e $project_absolute_path)) {
        # We need to check if the samples project file exists if it doesn't it means given sample
        # is not supported on provided msvc version so we waive the build of such sample
        print STDERR "Waiving build of $project_absolute_path as the dependencies are not met.\n";
        return 2;
      }

      if ($^O eq "cygwin")
        {
          $project =~ s#\\#/#g;
          $project =~ s#([a-zA-Z])://*#/cygdrive/$1/#g;
          $sample_dir =~ s#\\#/#g;
          $sample_dir =~ s#([a-zA-Z])://*#/cygdrive/$1/#g;
        }
      $ret = dvsLwdaSamples::exec "cd $sample_dir && $msbuild /t:Rebuild $project" if (!$ret);
    }
  return $ret;
}

sub build_samples_docker
{

  my ($samples, $make, $msbuild, $lwmake) = @_;

  my $driver_root = $driver_rootdir;

  start_timer("Build LWCA Samples");
  if (!$skip_build_files_generation)
  {
    if ($os eq "Linux")
    {
      $driver_root =~ s|/drivers/gpgpu/$||;
      $driver_root =~ s|/drivers/gpgpu$||;
      dvsLwdaSamples::exec "$make -C $samples_scripts_dir -f $samples_scripts_dir/createmk.mk DRIVER_ROOT=$driver_root GEN_BUILD_FILE=\"makefile\" PERL=\"$linux_perl_path\"";
    }

    if ($os eq "win32")
    {
      $driver_root =~ s|\\drivers\\gpgpu$||;
      $driver_root =~ s|\\drivers\\gpgpu\\$||;
      dvsLwdaSamples::exec "$lwmake -C $samples_scripts_dir -f $samples_scripts_dir\\windows.mk MINGW_INSTALL=$p4_tools_dir\\win64\\msys2\\20161116\\usr DRIVER_ROOT=$driver_root PERL=$p4_tools_dir\\win32\\ActivePerl\\5.10.0.1004\\bin\\perl";
    }
  }
  if (!$package_sources_only)
  {
    if ($os eq "Linux") 
    {
        my $ret = 0;
        print_test_preambule ("BUILDING", "all");

        foreach my $sample (sort (@$samples))
        {
          $ret = verify_sample($sample);
          last if $ret;
        }
        #my @projects = map $_ . "/Makefile", @$samples;

        $ret = dvsLwdaSamples::exec "$make clean -k -C $samples_rootdir/.. -j $parallel " if (!$ret);
        $ret = dvsLwdaSamples::exec "$make all -k -C $samples_rootdir/.. -j $parallel " if (!$ret);

       #       $ret = dvsLwdaSamples::exec "$make clean -k -C $samples_rootdir/.. -j $parallel  $headers_libs_path" if (!$ret);
      #$ret = dvsLwdaSamples::exec "$make all -k -C $samples_rootdir/.. -j $parallel  $headers_libs_path" if (!$ret);

        # A build error will show up as a make exit code 2. Needs to be colwerted to 1.
        $ret = 1 if $ret > 0;

        print_test_summary("all", $ret);
    }
    if ($os eq "win32")
    {
      foreach my $sample (@$samples)
      {
        print_test_preambule ("BUILDING", $sample);
        my $ret = build_sample_docker ($sample, $make, $msbuild);
        print_test_summary($sample, $ret);
      }
    }
  }

  stop_timer("Build LWCA Samples");

}

sub build_samples
{
  my ($samples, $make, $msbuild, $lwmake) = @_;

  my $driver_root = $driver_rootdir;

  if (!$skip_build_files_generation)
  {
    if ($os eq "Linux")
    {
      $driver_root =~ s|/drivers/gpgpu/$||;
      $driver_root =~ s|/drivers/gpgpu$||;
      dvsLwdaSamples::exec "$make -C $samples_scripts_dir -f $samples_scripts_dir/createmk.mk DRIVER_ROOT=$driver_root USE_P4=1 GEN_BUILD_FILE=\"makefile\" PERL=$linux_perl_path";
    }
    if ($os eq "Darwin")
    {
      $driver_root =~ s|/drivers/gpgpu/$||;
      $driver_root =~ s|/drivers/gpgpu$||;
      dvsLwdaSamples::exec "/usr/bin/make -C $samples_scripts_dir -f $samples_scripts_dir/mac.mk DRIVER_ROOT=$driver_root USE_P4=1 GEN_BUILD_FILE=\"makefile\"";
    }
    if ($os eq "win32")
    {
       if ($^O eq "cygwin")
       {
          my $user_path = $ELW{'PATH'};

          $driver_root =~ s|/drivers/gpgpu/$||;
          $driver_root =~ s|/drivers/gpgpu$||;
          $driver_root = `C:/cygwin/bin/cygpath.exe -w $driver_root`;
          chomp $driver_root;
          $driver_root =~ s#\\#/#g;

          my $samples_rootdir_cygpath = `C:/cygwin/bin/cygpath.exe -w $samples_rootdir`;
          chomp $samples_rootdir_cygpath;
          $samples_rootdir_cygpath =~ s#\\#/#g;

          my $perl_path = "$p4_tools_dir/win32/ActivePerl/5.10.0.1004/bin/perl";
          $perl_path = `C:/cygwin/bin/cygpath.exe -w $perl_path`; 
          chomp $perl_path;
          $perl_path =~ s#\\#/#g;

          # Several paths breaks lwmake.exe while running under cygwin, See bug 200036239 for more details
          $ELW{'PATH'} = "";
          dvsLwdaSamples::exec "$lwmake -C $samples_rootdir_cygpath -f $samples_rootdir_cygpath/scripts/windows.mk USE_P4=1 DRIVER_ROOT=$driver_root PERL=$perl_path";
          $ELW{'PATH'} = $user_path;
       }
       else
       {
          $driver_root =~ s|\\drivers\\gpgpu$||;
          $driver_root =~ s|\\drivers\\gpgpu\\$||;
          dvsLwdaSamples::exec "$lwmake -C $samples_scripts_dir -f $samples_scripts_dir\\windows.mk DRIVER_ROOT=$driver_root USE_P4=1 PERL=$p4_tools_dir\\win32\\ActivePerl\\5.10.0.1004\\bin\\perl";
       }
    }
  }

  start_timer("Build LWCA Samples");

  # temporary 
  $parallel = 1 if ($os eq "win32");
 
  if (!$package_sources_only) {
    if ($parallel > 1) {
      build_samples_parallel ($samples, $make, $msbuild);
    } 
    else {
      build_samples_sequential ($samples, $make, $msbuild);
    }
  }

  stop_timer("Build LWCA Samples");
}

sub build_samples_sequential
{
  my ($samples, $make, $msbuild) = @_;

  foreach my $sample (@$samples)
    {
      print_test_preambule ("BUILDING", $sample);
      my $ret = build_sample ($sample, $make, $msbuild);
      print_test_summary($sample, $ret);
   }
}

sub build_samples_parallel
{
  my ($samples, $make, $msbuild) = @_;
  my $ret = 0;

  print_test_preambule ("BUILDING", "all");

  foreach my $sample (sort (@$samples))
    {
      $ret = verify_sample($sample);
      last if $ret;
    }

  if ($os eq "Linux" or $os eq "Darwin")
    {
      my @projects = map $_ . "/Makefile", @$samples;
      my $lwstom_headers_path = "$samples_rootdir/../scripts/dvs/test_requisites";
      my $lwstom_libs_path = "$samples_rootdir/../scripts/dvs/test_requisites/x86_64_Linux";
      my $headers_libs_path = " DFLT_PATH=$lwstom_libs_path";
      $headers_libs_path .= " HEADER_SEARCH_PATH=$lwstom_headers_path ";
      $headers_libs_path .= " EXTRA_CCFLAGS=\"-I$lwstom_headers_path -L$lwstom_libs_path\" ";
      $ret = dvsLwdaSamples::exec "$make clean -k -C $samples_rootdir/.. -j $parallel  $headers_libs_path" if (!$ret);
      $ret = dvsLwdaSamples::exec "$make all -k -C $samples_rootdir/.. -j $parallel  $headers_libs_path" if (!$ret);
    }

  if ($os eq "win32")
    {
      $samples_rootdir =~ s#/#\\#g; 
      my $project = basename("Samples_vs2010.sln");
      my $max_cpu_count = $parallel - 1;
      my $processor_number = $parallel;
      my @targets_arr = map $_ . ":Rebuild", @$samples;
      my $targets = join (";", @targets_arr);
      $targets =~ s#/#\\#g; 
 
      if ($^O eq "cygwin")
        {
          $project =~ s#\\#/#g;
          $project =~ s#([a-zA-Z])://*#/cygdrive/$1/#g;
          $samples_rootdir =~ s#\\#/#g;
          $samples_rootdir =~ s#([a-zA-Z])://*#/cygdrive/$1/#g;
        }

      $ret = dvsLwdaSamples::exec "cd $samples_rootdir && "
                                . "$msbuild /m:$max_cpu_count "
                                         . "/p:ProcessorNumber=$processor_number "
                                         . "/t:$targets "
                                         . "$project" if (!$ret);
    }
  
  # A build error will show up as a make exit code 2. Needs to be colwerted to 1.
  $ret = 1 if $ret > 0; 

  print_test_summary("all", $ret);
}

sub run_samples
{
  my ($samples) = @_;

  start_timer ("Run LWCA Samples");

  foreach my $sample (@$samples)
    {
      print_test_preambule ("RUNNING", $sample);
      my $ret = run_sample ($sample);
      print_test_summary ($sample, $ret);
    }

  stop_timer ("Run LWCA Samples");
}

sub build_and_run_samples
{
  my ($samples, $make, $msbuild) = @_;
  my $ret = 0;

  start_timer("Build & Run LWCA Samples");

  foreach my $sample (@$samples)
    {
      print_test_preambule ("RUNNING", $sample);

      $ret = build_sample ($sample, $make, $msbuild);
      $ret = run_sample ($sample) if not $ret;

      print_test_summary($sample, $ret);
    }

  stop_timer("Build & Run LWCA Samples");
}

sub package_samples
{
  my ($samples_list) = @_;
  my @files = ();

  start_timer("Packaging");

  dvsLwdaSamples::exec "$delete_pkg_cmd";

  if ($os ne "win32") {
      dvsLwdaSamples::exec "rm -f $toolkit_rootdir/bin/${target_arch}_${os}_${build}/liblwda.so*";
  }

  if ($os eq "win32") {
      my $samples_bin = "$samples_dir\\bin";
      $samples_bin .= "\\win32" if ($target_arch eq "i686");
      $samples_bin .= "\\win64" if ($target_arch eq "x86_64");
      $samples_bin .= "\\Debug" if ($build eq "debug");
      $samples_bin .= "\\Release" if ($build eq "release");

      foreach my $sample (sort (@$samples_list))
        {
          push @files, "$samples_dir\\Samples\\$sample";
        }
      dvsLwdaSamples::exec "$pkg_cmd ".join(" ", @files);
      dvsLwdaSamples::exec "$pkg_cmd $samples_bin\\*";
      dvsLwdaSamples::exec "$pkg_cmd $samples_dir\\Common\\*";
      dvsLwdaSamples::exec "$pkg_cmd $samples_dir\\scripts\\*";
      dvsLwdaSamples::exec "$pkg_cmd build\\scripts\\get_define_val.pl";
      dvsLwdaSamples::exec "$pkg_cmd build\\scripts\\max.pl";
      dvsLwdaSamples::exec "$pkg_cmd bin\\${target_arch}_${os}${abi}_${build}\\*.dll";
  } else {
      my $samples_bin = "$samples_dir/bin/$target_arch/linux";
      $samples_bin .= "/debug" if ($build eq "debug");
      $samples_bin .= "/release" if ($build eq "release");

      foreach my $sample (sort (@$samples_list))
        {
          push @files, "$samples_dir/Samples/$sample";
        }
      dvsLwdaSamples::exec "$pkg_cmd ".join(" ", @files);
      dvsLwdaSamples::exec "$pkg_cmd $samples_bin";
      dvsLwdaSamples::exec "$pkg_cmd $samples_dir/Common";
      dvsLwdaSamples::exec "$pkg_cmd $samples_dir/scripts";
      dvsLwdaSamples::exec "$pkg_cmd --exclude='*.exe' --exclude='*test*' --exclude='*Test*' bin/${target_arch}_${os}${abi}_${build}"; 
      dvsLwdaSamples::exec "$pkg_cmd build/make-3.81"; 
  }

  stop_timer("Packaging");
}

1;
