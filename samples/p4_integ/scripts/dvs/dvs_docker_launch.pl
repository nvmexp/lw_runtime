#!/usr/bin/elw perl

use strict;
use warnings;
use FindBin;
use lib $FindBin::RealBin;
use Getopt::Long;
use Cwd;
use dvsLwdaSamples;

my $toolkit_rootdir = "";
my $driver_rootdir = "";
my $samples_rootdir = "";
my $samples_scriptdir = "";
my $os = "";
my $arch = "";
my $host_arch = "";
my $target_arch = "";
my $abi = "";
my $build = "";
my $p4_build_rootdir = "";
my $package = "";
my $keywords_include_string = "";
my $keywords_exclude_string = "";
my $help = 0;
my $debug = 0;
my $action = "";
my $p4_tools_dir = "";
my $p4_toolkit_dir = "";
my $msvc_version = "";
my $perl_override = "";
my $docker_image_url = "";
my $docker_cmd = "docker";
my $mode = ""; # 2 modes supported 1.) normal 2.) mig

sub launch_docker_build
{
    my $keywords_exclude_param = "--keywords-exclude $keywords_exclude_string" if ($keywords_exclude_string);
    my $keywords_include_param = "--keywords-include $keywords_include_string" if ($keywords_include_string);
    if (($os eq "win32") and ($^O eq "MSWin32"))
    {
        my $mount_p4dir    = "-v \"$p4_build_rootdir:c:\\p4dir\" ";
        my $perl_exe       = "C:\\p4dir\\sw\\tools\\win32\\ActivePerl\\5.10.0.1004\\bin\\perl.exe";
        my $set_perl_elw   = "-e PERL=$perl_exe";
        my $samples_build_script = "C:\\p4dir\\$toolkit_rootdir\\$samples_scriptdir\\dvs_build_lwda_samples.pl";
        #my $keywords_exclude_param = "--keywords-exclude $keywords_exclude_string" if ($keywords_exclude_string);
        #my $keywords_include_param = "--keywords-include $keywords_include_string" if ($keywords_include_string);
        my $dvs_build_lwda_samples_call = "$perl_exe $samples_build_script ";
        $dvs_build_lwda_samples_call .= "$keywords_exclude_param " if ($keywords_exclude_param);
        $dvs_build_lwda_samples_call .= "$keywords_include_param " if ($keywords_include_param);
        $dvs_build_lwda_samples_call .= "--mode dvs ";
        $dvs_build_lwda_samples_call .= "--p4build c:\\p4dir ";
        $dvs_build_lwda_samples_call .= "--driver $driver_rootdir ";
        $dvs_build_lwda_samples_call .= "--toolkit $toolkit_rootdir ";
        $dvs_build_lwda_samples_call .= "--package $package ";
        $dvs_build_lwda_samples_call .= "--samples-rootdir $samples_rootdir " if ($samples_rootdir);
        $dvs_build_lwda_samples_call .= "--arch $arch --os $os --build $build --use-docker --debug ";
        $dvs_build_lwda_samples_call .= "--msvc-version=$msvc_version " if ($msvc_version);
        my $docker_run_cmd = "$docker_cmd run -m 2gb $mount_p4dir $set_perl_elw  $docker_image_url ";
        $docker_run_cmd   .= "\" $dvs_build_lwda_samples_call\"";
        print "Exelwting $docker_run_cmd\n";
        system($docker_run_cmd);
    }
    if ($os eq "Linux")
    {
        my $mount_p4dir    = "-v \"$p4_build_rootdir:/p4dir\" ";
        my $perl_exe       = "/p4dir/sw/tools/linux/perl-5.18.1/bin/perl ";
        my $set_perl_elw   = "-e PERL=$perl_exe";
        my $samples_build_script = "/p4dir/$toolkit_rootdir/$samples_scriptdir/dvs_build_lwda_samples.pl";

        my $dvs_build_lwda_samples_call = "$perl_exe $samples_build_script ";
        $dvs_build_lwda_samples_call .= "$keywords_exclude_param " if ($keywords_exclude_param);
        $dvs_build_lwda_samples_call .= "$keywords_include_param " if ($keywords_include_param);
        $dvs_build_lwda_samples_call .= "--mode dvs ";
        $dvs_build_lwda_samples_call .= "--p4build /p4dir ";
        $dvs_build_lwda_samples_call .= "--driver $driver_rootdir ";
        $dvs_build_lwda_samples_call .= "--toolkit $toolkit_rootdir ";
        $dvs_build_lwda_samples_call .= "--package $package ";
        $dvs_build_lwda_samples_call .= "--samples-rootdir $samples_rootdir " if ($samples_rootdir);
        $dvs_build_lwda_samples_call .= "--arch $arch --os $os --build $build --use-docker --debug ";
        my $docker_run_cmd = "$docker_cmd run $mount_p4dir $set_perl_elw --rm $docker_image_url /bin/sh -c ";
        $docker_run_cmd   .= "\" $dvs_build_lwda_samples_call\"";
        print "Exelwting $docker_run_cmd\n";
        system($docker_run_cmd);
    }
}

sub launch_docker_run
{
    my $keywords_exclude_param = "--keywords-exclude $keywords_exclude_string" if ($keywords_exclude_string);
    my $keywords_include_param = "--keywords-include $keywords_include_string" if ($keywords_include_string);
    if (($os eq "win32") and ($^O eq "MSWin32"))
    {
        # my $mount_p4dir    = "-v \"$p4_build_rootdir:c:\\p4dir\" ";
        # my $perl_exe       = "C:\\p4dir\\sw\\tools\\win32\\ActivePerl\\5.10.0.1004\\bin\\perl.exe";
        # my $set_perl_elw   = "-e PERL=$perl_exe";
        # my $samples_build_script = "C:\\p4dir\\$samples_scriptdir\\dvs_build_lwda_samples.pl";
        # #my $keywords_exclude_param = "--keywords-exclude $keywords_exclude_string" if ($keywords_exclude_string);
        # #my $keywords_include_param = "--keywords-include $keywords_include_string" if ($keywords_include_string);
        # my $dvs_build_lwda_samples_call = "$perl_exe $samples_build_script ";
        # $dvs_build_lwda_samples_call .= "$keywords_exclude_param " if ($keywords_exclude_param);
        # $dvs_build_lwda_samples_call .= "$keywords_include_param " if ($keywords_include_param);
        # $dvs_build_lwda_samples_call .= "--mode dvs ";
        # $dvs_build_lwda_samples_call .= "--p4build c:\\p4dir ";
        # $dvs_build_lwda_samples_call .= "--driver $driver_rootdir ";
        # $dvs_build_lwda_samples_call .= "--toolkit $toolkit_rootdir ";
        # $dvs_build_lwda_samples_call .= "--package $package ";
        # $dvs_build_lwda_samples_call .= "--arch $arch --os $os --build $build --use-docker --debug ";
        # $dvs_build_lwda_samples_call .= "--msvc-version=$msvc_version " if ($msvc_version);
        # my $docker_run_cmd = "$docker_cmd run $mount_p4dir $set_perl_elw -it --rm $docker_image_url ";
        # $docker_run_cmd   .= "\" $dvs_build_lwda_samples_call\"";
        # print "Exelwting $docker_run_cmd\n";
        # dvsLwdaSamples::exec "$docker_run_cmd";
    }
    if ($os eq "Linux")
    {
        my $mount_p4dir    = "-v \"$toolkit_rootdir:/p4dir\" ";
        my $samples_build_script = "/p4dir/$samples_scriptdir/dvs_run_lwda_samples.pl";
        my $x11_and_lwda_support = " --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY -e XAUTHORITY -e LWIDIA_DRIVER_CAPABILITIES=all ";
        my $dvs_build_lwda_samples_call = "$samples_build_script ";
        $dvs_build_lwda_samples_call .= "$keywords_exclude_param " if ($keywords_exclude_param);
        $dvs_build_lwda_samples_call .= "$keywords_include_param " if ($keywords_include_param);
        $dvs_build_lwda_samples_call .= "--mode dvs ";
        $dvs_build_lwda_samples_call .= "--toolkit-rootdir /p4dir ";
        $dvs_build_lwda_samples_call .= "--samples-rootdir $samples_rootdir " if ($samples_rootdir);
        $dvs_build_lwda_samples_call .= "--arch $arch --os $os --build $build --debug ";
        my $docker_run_cmd = "$docker_cmd run $x11_and_lwda_support $mount_p4dir --rm $docker_image_url /bin/sh -c ";
        $docker_run_cmd   .= "\" $dvs_build_lwda_samples_call\"";
        print "Exelwting $docker_run_cmd\n";
        system($docker_run_cmd);
    }
}

sub pull_docker_image
{
    my ($docker_image_name) =  @_;
    my $docker_username = "lwda_ro";
    my $docker_password = "Lwidia3d!";
    my $docker_registry = "urm.lwpu.com/";

    my $cmd = "$docker_cmd login -u $docker_username -p $docker_password $docker_registry";
    print "Exelwting $cmd \n";
    system($cmd);
    $docker_image_url = $docker_registry . "sw-gpu-lwca-installer-docker-local/samples:" . $docker_image_name;
    $cmd = "$docker_cmd pull $docker_image_url";
    print "Exelwting $cmd \n";
    system($cmd);
}

sub remove_docker_image
{
    my $cmd = "$docker_cmd image rm -f $docker_image_url";
    print "Exelwting $cmd \n";
    system($cmd);
    $cmd = "$docker_cmd system prune -a -f";
    print "Exelwting $cmd \n";
    system($cmd);
}

sub main
{
    my $docker_image_name = "";

    GetOptions(
    "docker-image=s"            => \$docker_image_name,
    "toolkit-rootdir=s"         => \$toolkit_rootdir,
    "driver=s"                  => \$driver_rootdir,
    "arch=s"                    => \$arch,
    "mode=s"                    => \$mode,
    "host-arch=s"               => \$host_arch,
    "target-arch=s"             => \$target_arch,
    "os=s"                      => \$os,
    "abi=s"                     => \$abi,
    "build=s"                   => \$build,
    "p4build=s"                 => \$p4_build_rootdir,
    "package=s"                 => \$package,
    "help|h"                    => \$help,
    "action=s"                  => \$action,
    "keywords-include=s"        => \$keywords_include_string,
    "keywords-exclude=s"        => \$keywords_exclude_string,
    "p4-toolkit-dir=s"          => \$p4_toolkit_dir,
    "p4-tools-dir=s"            => \$p4_tools_dir,
    "samples-rootdir=s"         => \$samples_rootdir,
    "samples-scriptdir=s"       => \$samples_scriptdir,
    "msvc-version=s"            => \$msvc_version,
    "perl-override=s"           => \$perl_override,
    ) or die "Error while processing the command-line options. Exiting.";

    if ($help) {
        print_usage();
        exit 0;
    }

    $host_arch = $target_arch = $arch if ($arch);
    if ($toolkit_rootdir eq "" ||
      ($host_arch ne "x86_64" &&  $host_arch ne "ARMv7" && $host_arch ne "aarch64" && $host_arch ne "ppc64le") ||
      ($target_arch ne "x86_64" &&  $target_arch ne "ARMv7" && $target_arch ne "aarch64" && $target_arch ne "ppc64le") ||
      ($os ne "win32" && $os ne "Linux") ||
      ($host_arch eq "ARMv7" && $abi ne "" && $abi ne "gnueabi" && $abi ne "gnueabihf") ||
      ($target_arch eq "ARMv7" && $abi ne "" && $abi ne "gnueabi" && $abi ne "gnueabihf") ||
      ($build ne "debug" && $build ne "release") ||
      ($action ne "build" && $action ne "run" && $action ne "all" && $action ne "clobber"))
    {
        print STDERR "Error: missing argument\n\n";
        print_usage();
        exit 1;
    }

    if ($msvc_version eq "" and $os eq "win32")
    {
        $msvc_version = "VS2019";
    }

    if ($samples_scriptdir eq "")
    {
        $samples_scriptdir = "samples/scripts";
    }
    if ($mode eq "")
    {
        # TODO: add support for MIG mode
        # default mode is normal
        $mode = "normal";
    }

    if ($os eq "Linux")
    {
      $docker_cmd = "sudo docker";
    }

    if ($action eq "build")
    {
        pull_docker_image($docker_image_name);
        launch_docker_build();
        remove_docker_image();
    }
    elsif ($action eq "run")
    {
        if ($os eq "Linux")
        {
            $docker_image_name = "ubuntu_18.04_x86_64";
        }
        pull_docker_image($docker_image_name);
        launch_docker_run();
        remove_docker_image();
    }
}

sub print_usage
{
  my $script_name = basename ($0);

  print STDERR "Usage:  $script_name <arguments> <options>\n";
  print STDERR "\n";
  print STDERR "Required Arguments:\n";
  print STDERR "  --toolkit-rootdir <dir>        : toolkit root directory\n";
  print STDERR "  --driver-rootdir <dir>         : driver root directory\n";
  print STDERR "  --arch x86_64                  : architecture type for --host-arch and --target-arch\n";
  print STDERR "  --os win32|Linux               : operating system\n";
  print STDERR "  --build debug|release          : build type\n";
  print STDERR "  --p4build <dir>                : p4 build directory\n";
  print STDERR "  --package <pathname>           : full pathname of the result package\n";
  print STDERR "  --samples-scriptdir <dir>      : samples scripts directory.\n";
  print STDERR "\n";
  print STDERR "Options:\n";
  print STDERR "  --help                         : Print this help message\n";
  print STDERR "  --host-arch x86_64|ARMv7|aarch64|ppc64le   : architecture of the build machine\n";
  print STDERR "  --target-arch x86_64|ARMv7|aarch64|ppc64le : architecture of the target machine\n";
  print STDERR "  --abi gnueabi|gnueabihf        : ABI for ARMv7. Default is gnueabi\n";
  print STDERR "  --action clobber|build|run|all : build and/or run the samples.\n";
  print STDERR "  --keywords-include <k>[,<k>]*  : only include the samples that match any of the keywords.\n";
  print STDERR "  --keywords-exclude <k>[,<k>]*  : exclude the samples that match any of the keywords.\n";
  print STDERR "  --samples-rootdir <dir>        : samples root directory. Default is <toolkit_dir>/samples.\n";
  print STDERR "  --msvc-version                 : Permitted values are VS2019. Default is VS2019. Only usable when --os=win32. \n";
  print STDERR "  --perl-override                : Overrides perl binary used for generating Makefiles. Only usable when --os=Linux. \n";
}
main();

exit 0;