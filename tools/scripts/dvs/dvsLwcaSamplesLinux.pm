package dvsLwdaSamplesLinux;

use strict;
use Exporter;
use vars qw($VERSION @ISA @EXPORT @EXPORT_OK);
use File::Copy;

$VERSION     = 1.00;
@ISA         = qw(Exporter);
@EXPORT      = qw(set_elwironment_linux set_paths_linux install_files_linux);
@EXPORT_OK   = @EXPORT;

sub set_paths_linux
{
  my ($mode, $target_arch, $os, $abi, $build, $environment, $toolkit_rootdir, $driver_rootdir, $p4_toolkit_dir, $debug) = @_;

  if ($environment eq "dev")
    {
      my $toolkit_bin_path = "$toolkit_rootdir/bin/${target_arch}_${os}${abi}_${build}";
      my $driver_bin_path = "$driver_rootdir/bin/${target_arch}_${os}${abi}_${build}";
      my $build_path = "$p4_toolkit_dir/build";

      $driver_bin_path = "$driver_rootdir/bin/phat_${os}_$build" if ($os eq "Darwin");
    
      my $lib_path = "$toolkit_bin_path:$driver_bin_path";
      my $lwstom_libs_path = "$toolkit_rootdir/samples/github-lwca-samples/scripts/dvs/test_requisites/${target_arch}_${os}";

      $ELW{'PATH'} = "$toolkit_bin_path:$build_path:$ELW{'PATH'}";
      $ELW{'LD_LIBRARY_PATH'} = "$lib_path:$lwstom_libs_path" if ($os eq "Linux");
      $ELW{'DYLD_LIBRARY_PATH'} = $lib_path if ($os eq "Darwin");
      $ELW{'DYLD_FRAMEWORK_PATH'} = $lib_path if ($os eq "Darwin");
    }

  if ($environment eq "rel")
    {
      my $toolkit_bin_path = "$toolkit_rootdir/bin";

      my $toolkit_lib_path = "$toolkit_rootdir/lib";
      $toolkit_lib_path .= ":$toolkit_rootdir/lib64" if ($os eq "Linux" && $target_arch eq "x86_64");
    
      $ELW{'PATH'} = "$toolkit_bin_path:$ELW{'PATH'}";
      $ELW{'LD_LIBRARY_PATH'} = $toolkit_lib_path if ($os eq "Linux");
      $ELW{'DYLD_LIBRARY_PATH'} = $toolkit_lib_path if ($os eq "Darwin");
      $ELW{'DYLD_FRAMEWORK_PATH'} = $toolkit_lib_path if ($os eq "Darwin");
    }

  if ($debug)
    {
      print "\n";
      print "PATHS:\n";
      print "  PATH = $ELW{'PATH'}\n";
      print "  LD_LIBRARY_PATH = $ELW{'LD_LIBRARY_PATH'}\n" if ($os eq "Linux");
      print "  DYLD_LIBRARY_PATH = $ELW{'DYLD_LIBRARY_PATH'}\n" if ($os eq "Darwin");
      print "  DYLD_FRAMEWORK_PATH = $ELW{'DYLD_FRAMEWORK_PATH'}\n" if ($os eq "Darwin");
      print "\n";
    }
}

sub set_elwironment_linux
{
  my ($host_arch, $target_arch, $os, $abi, $build, $environment, $toolkit_rootdir, $driver_rootdir, $debug) = @_;

  # empty by design

  if ($debug)
    {
      print "\n";
      print "ENVIRONMENT:\n";
      print "  <empty>\n";
      print "\n";
    }
}

sub install_files_linux
{
  my ($src, $dst) = @_;
  copy ($src, $dst) or warn "Cannot copy file $src to $dst ($!)\n";
}

1;
