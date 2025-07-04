#!/usr/bin/elw perl
#
# Copyright (c) 2015      Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# Copyright (c) 2015-2020 Cisco Systems, Inc.  All rights reserved.
# $COPYRIGHT$
#
# Subroutine to generate a bunch of Fortran declarations and symbols
#

use strict;

use Getopt::Long;

my $caps_arg;
my $plain_arg;
my $single_underscore_arg;
my $double_underscore_arg;
my $help_arg = 0;

&Getopt::Long::Configure("bundling");
my $ok = Getopt::Long::GetOptions("caps=i" => \$caps_arg,
                                  "plain=i" => \$plain_arg,
                                  "single=i" => \$single_underscore_arg,
                                  "double=i" => \$double_underscore_arg,
                                  "help|h" => \$help_arg);

if ($help_arg || !$ok) {
    print "Usage: $0 [--caps|--plain|--single|--double] [--help]\n";
    exit(1 - $ok);
}

my $file_c_constants_decl = "mpif-c-constants-decl.h";
my $file_c_constants = "mpif-c-constants.h";
my $file_f08_types = "mpif-f08-types.h";

# If we are not building fortran, then just make empty files
if ($caps_arg + $plain_arg + $single_underscore_arg +
    $double_underscore_arg == 0) {
    system("touch $file_c_constants_decl");
    system("touch $file_c_constants");
    system("touch $file_f08_types");
    exit(0);
}

###############################################################

# Declare a hash of all the Fortran sentinel values

my $fortran;

$fortran->{bottom} = {
    c_type => "int",
    c_name => "mpi_fortran_bottom",
    f_type => "integer",
    f_name => "MPI_BOTTOM",
};
$fortran->{in_place} = {
    c_type => "int",
    c_name => "mpi_fortran_in_place",
    f_type => "integer",
    f_name => "MPI_IN_PLACE",
};
$fortran->{unweighted} = {
    c_type => "int",
    c_name => "mpi_fortran_unweighted",
    f_type => "integer, dimension(1)",
    f_name => "MPI_UNWEIGHTED",
};
$fortran->{weights_empty} = {
    c_type => "int",
    c_name => "mpi_fortran_weights_empty",
    f_type => "integer, dimension(1)",
    f_name => "MPI_WEIGHTS_EMPTY",
};

$fortran->{argv_null} = {
    c_type => "char",
    c_name => "mpi_fortran_argv_null",
    f_type => "character, dimension(1)",
    f_name => "MPI_ARGV_NULL",
};
$fortran->{argvs_null} = {
    c_type => "char",
    c_name => "mpi_fortran_argvs_null",
    f_type => "character, dimension(1, 1)",
    f_name => "MPI_ARGVS_NULL",
};

$fortran->{errcodes_ignore} = {
    c_type => "int",
    c_name => "mpi_fortran_errcodes_ignore",
    f_type => "integer, dimension(1)",
    f_name => "MPI_ERRCODES_IGNORE",
};
$fortran->{status_ignore} = {
    c_type => "int *",
    c_name => "mpi_fortran_status_ignore",
    f_type => "type(MPI_STATUS)",
    f_name => "MPI_STATUS_IGNORE",
};
$fortran->{statuses_ignore} = {
    c_type => "int *",
    c_name => "mpi_fortran_statuses_ignore",
    f_type => "type(MPI_STATUS)",
    f_name => "MPI_STATUSES_IGNORE(1)",
};

###############################################################

sub mangle {
    my $name = shift;

    if ($plain_arg) {
        return $name;
    } elsif ($caps_arg) {
        return uc($name);
    } elsif ($single_underscore_arg) {
        return $name . "_";
    } elsif ($double_underscore_arg) {
        return $name . "__";
    } else {
        die "Unknown name mangling type";
    }
}

sub gen_c_constants_decl {
    open(OUT, ">$file_c_constants_decl") ||
        die "Can't write to $file_c_constants_decl";

    print OUT "/* WARNING: This is a generated file!  Edits will be lost! */
/*
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2015      Cisco Systems, Inc.  All rights reserved.
 * \$COPYRIGHT\$
 *
 * This file was generated by gen-mpi-mangling.pl
 */

/* Note that the rationale for the types of each of these variables is
   dislwssed in ompi/include/mpif-common.h.  Do not change the types
   without also changing ompi/runtime/ompi_mpi_init.c and
   ompi/include/mpif-common.h. */\n\n";

    foreach my $key (sort(keys(%{$fortran}))) {
        my $f = $fortran->{$key};
        my $m = mangle($f->{c_name});
        print OUT "extern $f->{c_type} $m;
#define OMPI_IS_FORTRAN_" . uc($key) . "(addr) \\
        (addr == (void*) &$m)\n\n";
    }

    close(OUT);
}

sub gen_c_constants {
    open(OUT, ">$file_c_constants") ||
        die "Can't write to $file_c_constants";

    print OUT "/* WARNING: This is a generated file!  Edits will be lost! */
/*
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2015      Cisco Systems, Inc.  All rights reserved.
 * \$COPYRIGHT\$
 *
 * This file was generated by gen-mpi-mangling.pl
 */\n\n";

    foreach my $key (sort(keys(%{$fortran}))) {
        my $f = $fortran->{$key};
        my $m = mangle($f->{c_name});
        print OUT "$f->{c_type} $m;\n";
    }

    close (OUT);
}

sub gen_f08_types {
    open(OUT, ">$file_f08_types") ||
        die "Can't write to $file_f08_types";

    print OUT "! WARNING: This is a generated file!  Edits will be lost! */
!
! Copyright (c) 2015      Research Organization for Information Science
!                         and Technology (RIST). All rights reserved.
! Copyright (c) 2015      Cisco Systems, Inc.  All rights reserved.
! \$COPYRIGHT\$
!
! This file was generated by gen-mpi-mangling.pl
!\n\n";

    foreach my $key (sort(keys(%{$fortran}))) {
        my $f = $fortran->{$key};
        print OUT "$f->{f_type}, bind(C, name=\"".mangle($f->{c_name})."\") :: $f->{f_name}\n";
    }

    close (OUT);
}

gen_c_constants_decl();
gen_c_constants();
gen_f08_types();

exit(0);
