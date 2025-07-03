#!/usr/bin/elw perl

use strict;
use warnings;
use FindBin;
use lib $FindBin::RealBin;
use dvsLwdaSamples;

sub main
{
  my %keyword_samples = ();
  my %filtered_samples = ();
  my %blacklist = ();
  my @samples = ();
  my $make = "";
  my $lwmake = "";
  my $msbuild = "";
  my $error_code = 0;
  my %keywords_include = ();
  my %keywords_exclude = ();
  my $mode = "";
  my $action = "all";
  my $environment = "dev";

  read_command_line_options (\$mode, \$action, \$environment, \%keywords_include, \%keywords_exclude);

  set_paths ($mode);
  set_elwironment ();
  set_commands (\$make, \$lwmake, \$msbuild);

  build_samples_blacklist (\%blacklist);
  build_filtered_samples_list (\%filtered_samples);
  build_keyword_samples (\%keyword_samples, \%keywords_include, \%keywords_exclude);
  build_samples_list (\@samples, \%blacklist, \%filtered_samples, \%keyword_samples);

  if ($environment eq "dev")
    {
      copy_header_files (\%keywords_include, \%keywords_exclude);
      copy_library_files ();
      copy_msvc_build_extensions ();
      copy_installation ();
    }

  if ($action eq "clobber") {
      clobber_samples ($make, $msbuild, $lwmake);
    exit 0;
  } elsif ($action eq "build") {
    build_samples (\@samples, $make, $msbuild, $lwmake);
  } elsif ($action eq "run") {
    run_samples (\@samples);
  } elsif ($action eq "all") {
    build_and_run_samples (\@samples, $make, $msbuild);
  }

  print_timing_summary();
  print_results_summary();

  $error_code = get_num_failures () > 0 ? 1 : 0;
  exit $error_code;
}

main ();
exit 1;

