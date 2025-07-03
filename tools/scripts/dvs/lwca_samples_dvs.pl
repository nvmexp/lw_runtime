#!/usr/bin/elw perl

use strict;
use warnings;
use Getopt::Long;
use Cwd;
use Cwd 'abs_path';

my $debug = 1;
my $build = "release";
my $clean = 0;
my $test_name = "";
my $timeout = 1;
my $os = "";
my $arch = "";
my $make = "";
my $make_options = "";
my $make_targets = "";
my $test_output_limit = 2000;
my $toolkit_rootdir = "/sw/gpgpu";
my $samples_rootdir = "/sw/devrel/SDK10/Compute/C/noqatest";
my $samples_windir = "\\sw\\devrel\\SDK10\\Compute\\C\\noqatest";
my $windows_vs_target = 'vs2008';
my $driver_rootdir  = "";
my $passed = 0;
my $failed = 0;
my $waived = 0;
my $start_time = "";
my $stop_time = "";

my @lwda_samples_list = ("vectorAdd",
                         "boxFilter",
                         "matrixMul",
                         "matrixMulLWBLAS",
                         #"simpleD3D9",
                         "simpleGL",
                        );
sub print_usage()
{
    print STDERR "Usage:  lwda_samples_dvs.pl <options>\n";
    print STDERR "Options:\n";
    print STDERR "  -debug                   : Print extra debug messages\n";
    print STDERR "  -help                    : Print help message\n";
    print STDERR "  -timeout <min>           : timeout in minutes for each individual test (default: $timeout)\n";
    print STDERR "  -toolkit <dir>           : toolkit root directory\n";
    print STDERR "  -samples <dir>           : samples root directory (for dev elw only)\n";
    print STDERR "  -driver <dir>            : driver root directory (for dev elw only)\n";
    print STDERR "  -testname <name>         : run only the tests whose name include <name>\n";
    print STDERR "  -build debug|release     : build type (default: $build)\n";
    print STDERR "  -arch i686|x86_64        : override architecture type\n";
    print STDERR "  -os win32|Darwin|Linux   : override OS\n";
    print STDERR "  -clean                   : force rebuilding each sample\n";
}

sub print_results_summary
{
  my $dvs_score = 0;

  $dvs_score = 100 * ($passed / ($passed + $failed)) if ($passed + $failed) != 0;

  print "\n";
  print "RESULTS\n";
  print "Passed : $passed\n";
  print "Waived : $waived\n";
  print "Failed : $failed\n";
  printf "LWCA DVS BASIC SANITY SCORE: %.1f\n", $dvs_score;
}

sub print_timing_summary
{
  print "\n";
  print "Start time : $start_time\n";
  print "Stop time  : $stop_time\n";
}

sub read_command_line_options()
{
  GetOptions(
     "debug"                    => \$debug,
     "help"                     => sub { print_usage() and exit 0 },
     "timeout=i"                => \$timeout,
     "testname=s"               => \$test_name,
     "toolkit=s"                => \$toolkit_rootdir,
     "samples=s"                => \$samples_rootdir,
     "driver=s"                 => \$driver_rootdir,
     "build=s"                  => \$build,
     "arch=s"                   => \$arch,
     "os=s"                     => \$os,
     "clean"                    => \$clean,
     );
}

sub get_lwrrent_time()
{
   my ($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime(time);
   $year += 1900;
   $mon += 1;
   return sprintf ("%04d-%02d-%02d %02d:%02d:%02d", $year, $mon, $mday, $hour, $min, $sec);
}

sub set_elwironment()
{
  my $native_os = $^O;
  my $native_arch = "";

  if ($native_os eq "MSWin32" || $native_os =~ m/CYGWIN/ig) {
      $native_os = "win32";
      $ELW{'PROCESSOR_ARCHITECTURE'} ||= "";
      $ELW{'PROCESSOR_ARCHITEW6432'} ||= "";
      if ((lc($ELW{PROCESSOR_ARCHITECTURE}) ne "x86") ||
          (lc($ELW{PROCESSOR_ARCHITECTURE}) eq "amd64") ||
          (lc($ELW{PROCESSOR_ARCHITEW6432}) eq "amd64"))
        {
          $native_arch = "x86_64";
        }
      else {
          $native_arch = "i686";
      }
  } else {
      $native_os   = `uname`;    chomp($native_os);
      $native_arch = `uname -m`; chomp($native_arch);
  }

  $os = $native_os if $os eq "";
  $arch = $native_arch if $arch eq "";

  $ELW{'LIBLWDA_PATH'} ||= "";

  my $toolkit_bin_path = "${toolkit_rootdir}/bin/${arch}_${os}_${build}";
  my $driver_bin_path = "${driver_rootdir}/bin/${arch}_${os}_${build}";
  my $build_path = "${toolkit_rootdir}/build";
  my $user_path = $ELW{'PATH'};

  $samples_rootdir = "${toolkit_rootdir}/samples" if (${samples_rootdir} eq "");

  my $lib_path = "$toolkit_bin_path";
  if ($driver_rootdir ne "") {
      $lib_path = "$lib_path:${driver_bin_path}";
  }

  if ($os eq "win32") {
      $make = "msbuild";
  } elsif ($os eq "Linux") {
      $make = "${build_path}/make-3.81";
      $ELW{'LD_LIBRARY_PATH'} = ${lib_path};
  } elsif ($os eq "Darwin") {
      $make = "/usr/bin/make";
      $ELW{'DYLD_LIBRARY_PATH'} = ${lib_path};
  }

  $ELW{'PATH'} = "${toolkit_bin_path}:${build_path}:${user_path}";

  my $extra_lwccflags = "";
  $extra_lwccflags .= "\"";
  $extra_lwccflags .= "-I${toolkit_rootdir}/lwca/import ";
  $extra_lwccflags .= "-I${toolkit_rootdir}/lwca/tools/lwdart ";
  $extra_lwccflags .= "-I${toolkit_rootdir}/lwfft/branches/TOT/liblwfft ";
  $extra_lwccflags .= "-I${toolkit_rootdir}/lwblas/src ";
  $extra_lwccflags .= "-I${toolkit_rootdir}/lwsparse/src ";
  $extra_lwccflags .= "-I${toolkit_rootdir}/lwrand/src ";
  $extra_lwccflags .= "-I${toolkit_rootdir}/NPP/npp/include ";
  $extra_lwccflags .= "\"";

  my $extra_ldflags = "";
  $extra_ldflags .= "\"";
  $extra_ldflags .= "-L${toolkit_bin_path} ";
  $extra_ldflags .= "-L${driver_bin_path} ";
  $extra_ldflags .= "\""; 

  if ($os eq "win32") {
    $ELW{'LWDA_PATH'} = ${toolkit_rootdir};
	
	$make_options .= "";
	
    $make_targets .= "/t:Clean " if $clean;
    $make_targets .= "/t:Rebuild /p:Configuration=Release /p:Platform=x64";
  } else {
    $make_options .= "-s ";
    $make_options .= "LWDA_INC_PATH=${toolkit_bin_path} ";
    $make_options .= "LWDA_LIB_PATH=${toolkit_bin_path} ";
    $make_options .= "LWDA_BIN_PATH=${toolkit_bin_path} ";
    $make_options .= "EXTRA_LWCCFLAGS=${extra_lwccflags} ";
    $make_options .= "EXTRA_CCFLAGS=${extra_lwccflags} ";
    $make_options .= "EXTRA_LDFLAGS=${extra_ldflags} ";

    $make_targets .= "clean " if $clean;
    $make_targets .= "build ";
  }

  if ($debug) {
      print "\n";
      print "Environment:\n";
      print "  toolkit_bin_path = $toolkit_bin_path\n";
      print "  driver_bin_path = $driver_bin_path\n";
      print "  build_path = $build_path\n";
      print "  lib_path = $lib_path\n";
      print "  user_path = $user_path\n";
      print "  make  = $make\n";
      print "  make_options  = $make_options\n";
      print "  PATH = $ELW{'PATH'}\n";
      print "  LD_LIBRARY_PATH = $ELW{'LD_LIBRARY_PATH'}\n" if ($os eq "Linux");
      print "  DYLD_LIBRARY_PATH = $ELW{'DYLD_LIBRARY_PATH'}\n" if ($os eq "Darwin");
  }
}

# Wrapper for system that logs the commands so you can see what it did
sub exelwte_command
{
    my ($cmd) = @_;
    my  $ret = 0;

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
        printf "Application error code: $ret\n" if $debug;

        alarm 0;
    };

    if ($@) {
        if ($@ eq "alarm") {
            my @exelwtable = split(' ', $cmd, 2);
            printf "\n Application timed out. Killing $exelwtable[0].\n";
            system ("killall ".$exelwtable[0]);
        }
        else {
            my $err = $@;
            printf "\n Application died unexpectedly : $err\n";
        }
        $ret = 1;
    }

    return $ret if ($ret == 0 || $ret == 1 || $ret == 2);

    my $signals  = $ret & 127;
    my $app_exit = $ret >> 8;
    my $dumped_core = $ret & 0x80;

    if (($app_exit != 0) && ($app_exit != 0)) {
        printf "\n Application exited with status $app_exit\n";
    }
    if ($signals != 0) {
        printf "\n Application received signal $signals\n";
    }
    if ($dumped_core != 0) {
        printf "\n Application generated a core dump\n";
    }

    return 1;
}

sub print_test_preambule
{
  my $sample = $_[0];
  my $lwrrent_time= get_lwrrent_time();

  print "\n";
  print "&&&& INFO ilwoking LWCA sample [${sample}] at $lwrrent_time\n";
  print "&&&& RUNNING ${sample}\n";
  print "\n";
}

sub print_test_summary
{
  my $sample = $_[0];
  my $ret = $_[1];

  $passed += 1 if $ret == 0;
  $failed += 1 if $ret == 1;
  $waived += 1 if $ret == 2;

  print "\n";
  print "&&&& SUCCESS $sample\n" if $ret == 0;
  print "&&&& FAILED $sample\n"  if $ret == 1;
  print "&&&& WAIVED $sample\n"  if $ret == 2;
}

sub get_sample_ilwocations
{
  my $sample = $_[0];
  my @ilwocations = ();
  my $num_ilwocations = 0;
  my $info_file = "${samples_rootdir}/${sample}/info.xml";

  open(INFO, $info_file) || die "Unable to open $info_file";

  # extract one invocation 
  while (<INFO>) {
      next if $_ !~ m/<qatest>.*<\/qatest>/;
      my $invocation= $_;
      $invocation=~ s/.*<qatest>(.*)<\/qatest>.*/$1/; 
      chomp $invocation;
      $num_ilwocations = push(@ilwocations, $invocation);
  }

  close (INFO);

  # no invocation means empty invocation
  push(@ilwocations, "") if !$num_ilwocations;

  return @ilwocations;
}

sub run_tests()
{
  $start_time = get_lwrrent_time();

  foreach my $sample (@lwda_samples_list)
    {
      next if $sample !~ m/$test_name/;

      print_test_preambule $sample;

	  my $cmd = "";
      if ($os eq "win32") {
         $cmd = "$make $make_options $make_targets ${samples_windir}\\${sample}\\${sample}_${windows_vs_target}.sln" ;
	  } else {
         $cmd = "$make $make_options $make_targets -C ${samples_rootdir}/${sample}" ;
	  }
	  
      print "$cmd\n" if $debug;
      my $ret = exelwte_command $cmd;
		 
      if ($ret gt 0) {
        print_test_summary($sample, $ret);
        next;
      }

      my @ilwocations = get_sample_ilwocations(${sample});
      for my $invocation (@ilwocations) {
          $cmd = "cd ${samples_rootdir}/${sample} && ${sample} $invocation";
          print "$cmd\n" if $debug;
          $ret = exelwte_command $cmd;
      }

      print_test_summary($sample, $ret);
   }

  $stop_time = get_lwrrent_time();
}

sub main()
{
  read_command_line_options();
  set_elwironment();

  run_tests();

  print_timing_summary();
  print_results_summary();
}

main();
