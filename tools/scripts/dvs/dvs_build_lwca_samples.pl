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
  my $mode = "dvs";
  my $action = "build";
  my $environment = "dev";
 
  read_command_line_options (\$mode, \$action, \$environment, \%keywords_include, \%keywords_exclude);

  build_samples_blacklist (\%blacklist);
  build_filtered_samples_list (\%filtered_samples);
  build_keyword_samples (\%keyword_samples, \%keywords_include, \%keywords_exclude);
  build_samples_list (\@samples, \%blacklist, \%filtered_samples, \%keyword_samples);

  if ($mode eq "dvs" and $action eq "clobber")
    {
      set_paths ($mode);
      set_elwironment ();
      set_commands (\$make, \$lwmake, \$msbuild);
      set_compression_commands ();

      if ($environment eq "dev")
        {
          copy_header_files (\%keywords_include, \%keywords_exclude);
          copy_msvc_build_extensions ();
          copy_installation ();
          copy_library_files(\%keywords_include, \%keywords_exclude);
        }

      clobber_samples ($make, $msbuild, $lwmake);
    }

  if ($mode eq "dvs" and $action eq "build")
    {
      set_paths ($mode);
      set_elwironment ();
      set_commands (\$make, \$lwmake, \$msbuild);
      set_compression_commands ();

      merge_required_packages();

      if ($environment eq "dev")
        {
          copy_header_files (\%keywords_include, \%keywords_exclude);
          copy_msvc_build_extensions ();
          copy_installation ();
          copy_library_files(\%keywords_include, \%keywords_exclude);
        }

      build_samples (\@samples, $make, $msbuild, $lwmake);
    
      package_samples (\@samples);
    }

  if ($mode eq "dvs" and $action eq "run")
    {
      set_run_elwironment();
      run_samples(\@samples);
    }

  print_timing_summary();
  print_results_summary();
}

main();
exit 0;
