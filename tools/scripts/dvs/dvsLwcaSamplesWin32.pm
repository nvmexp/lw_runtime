package dvsLwdaSamplesWin32;

use strict;
use Exporter;
use File::Copy;
use File::Path qw(mkpath);
use vars qw($VERSION @ISA @EXPORT @EXPORT_OK);
use Win32::TieRegistry (Delimiter=>"/");
use Win32;

$VERSION     = 1.00;
@ISA         = qw(Exporter);
@EXPORT      = qw(set_elwironment_win32 set_paths_win32
                  copy_installation_win32 install_files_win32);
@EXPORT_OK   = @EXPORT;

sub set_paths_win32
{
  my ($mode, $target_arch, $os, $build, $environment, $toolkit_rootdir, $driver_rootdir, $debug) = @_;

  my $user_path = $ELW{'PATH'};
  $ELW{'PATH'} = "";
  $ELW{'PATH'} .= "$toolkit_rootdir/bin/${target_arch}_${os}_${build}/local/bin;" if ($environment eq "dev") ;
  $ELW{'PATH'} .= "$toolkit_rootdir/bin;" if ($environment eq "rel");
  $ELW{'PATH'} .= "$user_path";

  if ($^O eq "cygwin")
    {
      $ELW{'PATH'}           =~ s#\\#/#g;
      $ELW{'PATH'}           =~ s#[^a-zA-Z][a-zA-Z]://*#/cygdrive/c/#g;
      $ELW{'PATH'}           =~ s#;#:#g;
    }

  if ($debug)
    {
      print "\n";
      print "PATHS:\n";
      print "  PATH = $ELW{'PATH'}\n";
      print "\n";
    }
}

sub set_elwironment_win32
{
  my ($host_arch, $target_arch, $os, $build, $environment, $toolkit_rootdir, $driver_rootdir, $p4_tools_dir, $p4_toolkit_dir, $vulcan_toolkit_dir, $msvc_version, $debug) = @_;

  my $windows_sdk_dir; 
  my $ms_dot_net_dir;
  my $ms_dot_net_dir64;
  my $msvc_dir;

  if ($environment eq "rel" && $vulcan_toolkit_dir ne "")
  {
    $windows_sdk_dir = "$p4_tools_dir/windsdk/7.1" if ($msvc_version =~ m/vc10/i);
    $windows_sdk_dir = "$p4_tools_dir/windsdk/8.1" if ($msvc_version  =~ m/vc11/i);
    $ms_dot_net_dir  = "$p4_tools_dir/dotNet/4.0/Framework/v4.0.30319";
    $ms_dot_net_dir64 = "$p4_tools_dir/dotNet/4.0/Framework64/v4.0.30319";
    $msvc_dir = "$p4_tools_dir/msvc100sp1" if ($msvc_version =~ m/vc10/i);
    $msvc_dir = "$p4_tools_dir/msvc110" if ($msvc_version =~ m/vc11/i);
  }
  else
  {
    $windows_sdk_dir = "$p4_tools_dir/sdk/WinSDK/7.1" if ($msvc_version =~ m/vc10/i);
    $windows_sdk_dir = "$p4_tools_dir/sdk/WinSDK/8.1" if ($msvc_version  =~ m/vc11/i);
    $ms_dot_net_dir  = "$p4_tools_dir/sdk/dotNet/4.0/Framework/v4.0.30319";
    $ms_dot_net_dir64 = "$p4_tools_dir/sdk/dotNet/4.0/Framework64/v4.0.30319";
    $msvc_dir = "$p4_tools_dir/win32/msvc100sp1" if ($msvc_version =~ m/vc10/i);
    $msvc_dir = "$p4_tools_dir/win32/msvc110" if ($msvc_version =~ m/vc11/i);
  }

  $msvc_dir =~ s#\/#\\#g;
  $msvc_dir =~ s#^\\\\#\\#g;

  my $build_path = "$p4_toolkit_dir\\build";

  my $vc_bin_path = "";
  $vc_bin_path = "$msvc_dir/VC/bin" if $host_arch eq "i686" and $target_arch eq "i686";
  $vc_bin_path = "$msvc_dir/VC/bin" if $host_arch eq "x86_64" and $target_arch eq "i686";
  $vc_bin_path = "$msvc_dir/VC/bin/amd64" if $host_arch eq "x86_64" and $target_arch eq "x86_64";
  $vc_bin_path = "$msvc_dir/VC/bin/x86_amd64" if $host_arch eq "i686" and $target_arch eq "x86_64";

  my $vc_packages_path = "$msvc_dir/VC/VCPackages";
  my $vs_ide_path = "$msvc_dir/Common7/IDE";
  my $vs_tools_path = "$msvc_dir/Common7/Tools";

  my $sdk_bin_path = "$windows_sdk_dir/x64" if $target_arch eq "x86_64";

  my $user_path = $ELW{'PATH'};
  $ELW{'PATH'} = "";
  $ELW{'PATH'} .= "$vc_bin_path;";
  $ELW{'PATH'} .= "$ms_dot_net_dir64;";
  # Path to Microsoft Visual C++ Redistributable required for VC11
  $ELW{'PATH'} .= "$msvc_dir\\VC\\redist\\x64\\Microsoft.VC110.CRT;" if ($msvc_version =~ m/vc11/i);
  $ELW{'PATH'} .= "$p4_tools_dir\\sdk\\dotNet\\4.0\\MSBuild\\Microsoft.Cpp\\v4.0\\V110;" if ($msvc_version =~ m/vc11/i);

  my @split_paths = split(/;/, $user_path);
  my @spliced_arr;
  foreach (@split_paths)
  {
    if (!($_ =~ m/Program\ Files/i))
    {
      push @spliced_arr, $_;
    }
  }
  $user_path = join(';', @spliced_arr); 

  $ELW{'PATH'} .= "$vc_packages_path;";
  $ELW{'PATH'} .= "$vs_ide_path;";
  $ELW{'PATH'} .= "$vs_tools_path;";
  $ELW{'PATH'} .= "$sdk_bin_path;";
  $ELW{'PATH'} .= "$build_path;";
  $ELW{'PATH'} .= "$windows_sdk_dir/Bin;"if ($msvc_version =~ m/vc10/i);
  $ELW{'PATH'} .= "$windows_sdk_dir/bin/x64;" if ($msvc_version =~ m/vc11/i);
  $ELW{'PATH'} .= "$user_path;";

  $ELW{'INCLUDE'} .= "$msvc_dir/VC/INCLUDE;";
  $ELW{'INCLUDE'} .= "$windows_sdk_dir/INCLUDE;" if ($msvc_version =~ m/vc10/i);
  $ELW{'INCLUDE'} .= "$windows_sdk_dir/INCLUDE/um;" if ($msvc_version =~ m/vc11/i);
  $ELW{'INCLUDE'} .= "$windows_sdk_dir/INCLUDE/shared;" if ($msvc_version =~ m/vc11/i);
  $ELW{'INCLUDE'} .= "$windows_sdk_dir/INCLUDE/gl;" if ($msvc_version =~ m/vc10/i);
  $ELW{'INCLUDE'} .= "$windows_sdk_dir/INCLUDE/um/gl;" if ($msvc_version =~ m/vc11/i);

  if ($target_arch eq "x86_64")
    {
      $ELW{'LIB'} .= "$msvc_dir/VC/Lib/amd64;";
      $ELW{'LIB'} .= "$windows_sdk_dir/Lib/X64;"if ($msvc_version =~ m/vc10/i);
      $ELW{'LIB'} .= "$windows_sdk_dir/Lib/wilw6.3/um/x64;"if ($msvc_version =~ m/vc11/i);
      $ELW{'LIBPATH'} .= "$ms_dot_net_dir64;";
      $ELW{'LIBPATH'} .= "$msvc_dir/VC/Lib/amd64;";
    }
  else
    {
      $ELW{'LIB'} .= "$msvc_dir/VC/Lib;";
      $ELW{'LIB'} .= "$windows_sdk_dir/Lib;";
      $ELW{'LIBPATH'} .= "$ms_dot_net_dir;";
      $ELW{'LIBPATH'} .= "$msvc_dir/VC/Lib;";
    }
    
  if ($environment eq "dev")
    {
      my $toolkit_bin_path = "$toolkit_rootdir\\bin\\${target_arch}_${os}_${build}";
      my $driver_bin_path = "$driver_rootdir\\bin\\${target_arch}_${os}_${build}";
      $ELW{'LIBLWDA_PATH'} = "$driver_bin_path";

      if ($^O eq "cygwin")
        {
          $ELW{'LIBLWDA_PATH'}   =~ s#\\#/#g;
          $ELW{'LIBLWDA_PATH'}   =~ s#[a-zA-Z]://*#/cygdrive/c/#g;
          $ELW{'LIBLWDA_PATH'}   =~ s#;#:#g;

          $ELW{'LWDA_PATH'}      = "C:\\cygwin$toolkit_bin_path\\local";
          $ELW{'LWDA_PATH_V9_0'} = "C:\\cygwin$toolkit_bin_path\\local";
          $ELW{'LWDA_PATH_V9_1'} = "C:\\cygwin$toolkit_bin_path\\local";
          $ELW{'LWDA_PATH_V9_2'} = "C:\\cygwin$toolkit_bin_path\\local";
          $ELW{'LWDAPropsPath'}  = "C:\\cygwin$toolkit_bin_path\\MsBuildExtensions";
        }

      if ($^O eq "MSWin32")
        {
          $ELW{'LWDA_PATH'}      = "$toolkit_bin_path/local";
          $ELW{'LWDA_PATH_V9_0'} = "$toolkit_bin_path/local";
          $ELW{'LWDA_PATH_V9_1'} = "$toolkit_bin_path/local";
          $ELW{'LWDA_PATH_V9_2'} = "$toolkit_bin_path/local";
          $ELW{'LWDA_PATH_V9_2'} = "$toolkit_bin_path/local";
          $ELW{'LWDA_PATH_V10_0'} = "$toolkit_bin_path/local";
          $ELW{'LWDA_PATH_V10_1'} = "$toolkit_bin_path/local";
          $ELW{'LWDA_PATH_V10_2'} = "$toolkit_bin_path/local";
          $ELW{'LWDAPropsPath'}  = "$toolkit_bin_path\\MsBuildExtensions";
        } 

      $ELW{'LIBLWDA_PATH'}   =~ s#/#\\#g;
      $ELW{'LWDA_PATH'}      =~ s#/#\\#g;
      $ELW{'LWDA_PATH_V9_0'} =~ s#/#\\#g;
      $ELW{'LWDA_PATH_V9_1'} =~ s#/#\\#g;
      $ELW{'LWDA_PATH_V9_2'} =~ s#/#\\#g;
      $ELW{'LWDA_PATH_V10_0'} =~ s#/#\\#g;
      $ELW{'LWDA_PATH_V10_1'} =~ s#/#\\#g;
      $ELW{'LWDA_PATH_V10_2'} =~ s#/#\\#g;
      $ELW{'LWDAPropsPath'}  =~ s#/#\\#g;
    }

  if ($environment eq "rel" && $vulcan_toolkit_dir ne "")
    {
       $ELW{'LWDA_PATH'}      =  $vulcan_toolkit_dir;
       $ELW{'LWDA_PATH_V9_0'} =  $vulcan_toolkit_dir;
       $ELW{'LWDA_PATH_V9_1'} =  $vulcan_toolkit_dir;
       $ELW{'LWDA_PATH_V9_2'} =  $vulcan_toolkit_dir;
       $ELW{'LWDA_PATH_V10_0'} =  $vulcan_toolkit_dir;
       $ELW{'LWDA_PATH_V10_1'} =  $vulcan_toolkit_dir;
       $ELW{'LWDA_PATH_V10_2'} =  $vulcan_toolkit_dir;
       $ELW{'LWDAPropsPath'}  =  "$toolkit_rootdir\\dotNet\\4.0\\MSBuild\\Microsoft.Cpp\\v4.0\\BuildLwstomizations";
       $ELW{'MSBuildToolsPath'} =  "$ms_dot_net_dir64";
       $ELW{'MSBuildExtensionsPath'} = "$p4_tools_dir\\dotNet\\4.0\\MSBuild";
       $ELW{'MSBuildExtensionsPath32'} = "$p4_tools_dir\\dotNet\\4.0\\MSBuild";
       $ELW{'MSBuildExtensionsPath64'} = "$p4_tools_dir\\dotNet\\4.0\\MSBuild";
    }
 
  if ($environment eq "dev" || $environment eq "rel")
    {
      if ($^O eq "cygwin")
        {
          $ELW{'PATH'}           =~ s#\\#/#g;
          $ELW{'PATH'}           =~ s#[a-zA-Z]://*#/cygdrive/c/#g;
          $ELW{'PATH'}           =~ s#;#:#g;
          $ELW{'DXSDK_DIR'}      = "C:\\cygwin\\home\\lwda0\\lwca\\sw\\tools\\sdk\\DirectX_Jun2010";
        }

      if ($^O eq "MSWin32")
        {
          $ELW{'PATH'}           =~ s#/#\\#g;
          $ELW{'DXSDK_DIR'}      = "$p4_tools_dir\\sdk\\DirectX_Jun2010";
        }

       $ELW{'DXSDK_DIR'}      =~ s#/#\\#g;
       # Added to use MSBuild from p4 depot
       $ELW{'MSBuildToolsPath'} =  "$ms_dot_net_dir64";
       $ELW{'MSBuildExtensionsPath'} = "$p4_tools_dir\\sdk\\dotNet\\4.0\\MSBuild";
       $ELW{'MSBuildExtensionsPath32'} = "$p4_tools_dir\\sdk\\dotNet\\4.0\\MSBuild";
       $ELW{'MSBuildExtensionsPath64'} = "$p4_tools_dir\\sdk\\dotNet\\4.0\\MSBuild";
    }
 
  # Override PROCESSOR_ARCHITECTURE so that LWCC uses the right --ccbin option
  # The variable is normally set by the OS based on the bitness of the
  # application being run (here Perl), not the bitness of the machine.
  # Either way, we decide what LWCC should do, not Perl.
  $ELW{'PROCESSOR_ARCHITECTURE'} = "x86" if $target_arch eq "i686";
  $ELW{'PROCESSOR_ARCHITECTURE'} = "AMD64" if $target_arch eq "x86_64";

  if ($debug)
    {
      print "\n";
      print "ENVIRONMENT:\n";
      print "  PATH = $ELW{'PATH'}\n";
      print "  LIBLWDA_PATH = $ELW{'LIBLWDA_PATH'}\n";
      print "  INCLUDE = $ELW{'INCLUDE'}\n";
      print "  LIB = $ELW{'LIB'}\n";
      print "  LIBPATH = $ELW{'LIBPATH'}\n";
      print "  LWDA_PATH = $ELW{'LWDA_PATH'}\n";
      print "  LWDA_PATH_V10_0 = $ELW{'LWDA_PATH_V10_0'}\n";
      print "  LWDA_PATH_V10_1 = $ELW{'LWDA_PATH_V10_1'}\n";
      print "  LWDA_PATH_V10_2 = $ELW{'LWDA_PATH_V10_2'}\n";
      print "  LWDAPropsPath = $ELW{'LWDAPropsPath'}\n";
      print "  PROCESSOR_ARCHITECTURE = $ELW{'PROCESSOR_ARCHITECTURE'}\n";

      if ($vulcan_toolkit_dir ne "")
      {
        print "  MSBuildToolsPath = $ELW{'MSBuildToolsPath'}\n";
        print "  MSBuildExtensionsPath = $ELW{'MSBuildExtensionsPath'}\n";
        print "  MSBuildExtensionsPath32 = $ELW{'MSBuildExtensionsPath32'}\n";
        print "  MSBuildExtensionsPath64 = $ELW{'MSBuildExtensionsPath64'}\n";
        print "\n";
      }
      print "  MSBuildToolsPath = $ELW{'MSBuildToolsPath'}\n";
      print "  MSBuildExtensionsPath = $ELW{'MSBuildExtensionsPath'}\n";
      print "  MSBuildExtensionsPath32 = $ELW{'MSBuildExtensionsPath32'}\n";
      print "  MSBuildExtensionsPath64 = $ELW{'MSBuildExtensionsPath64'}\n";
      print "\n";
    }
}

sub install_files_win32
{
  my ($src, $dst) = @_;

  if ($^O eq "cygwin")
    {
      system "/usr/bin/cp", "-p", "-f", "$src", "$dst";

      if ($? == -1) {
        print "failed to execute: $!\n";
      } elsif ($? & 127) {
        printf "child died with signal %d, %s coredump\n", ($? & 127),  ($? & 128) ? 'with' : 'without';
      } elsif (($? >> 8) != 0) {
        printf "child exited with value %d\n", $? >> 8;
      }
    }
  elsif ($^O eq "MSWin32")
    {
      Win32::CopyFile ($src, $dst, 1) or warn "Cannot copy file $src to $dst ($!)\n";
    }
}

1;
