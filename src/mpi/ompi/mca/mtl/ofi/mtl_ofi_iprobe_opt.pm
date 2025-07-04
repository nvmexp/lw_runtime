#!/usr/bin/elw perl
#
# Copyright (c) 2013-2018 Intel, Inc. All rights reserved
#
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

use strict;
use warnings;
use opt_common::mtl_ofi_opt_common;
package mtl_ofi_iprobe_opt;

my @true_false = ("false", "true");

sub gen_funcs {
    my $gen_file = $_[0];
    my $gen_type = $_[1];
    my $OFI_CQ_DATA_EN = "false";

    foreach $OFI_CQ_DATA_EN (@true_false) {
        my @flags = ($OFI_CQ_DATA_EN);
        if (($gen_type cmp "FUNC") == 0) {
            my $FUNC = gen_iprobe_function(\@flags);
            print $gen_file "$FUNC\n\n";
        }
        if (($gen_type cmp "SYM") == 0) {
            my $SYM = gen_iprobe_sym_init(\@flags);
            print $gen_file "$SYM\n";
        }
    }
}

sub gen_iprobe_function {
    my @op_flags = @{$_[0]};
    my $MTL_OFI_NAME_EXT = opt_common::mtl_ofi_opt_common::gen_flags_ext(\@op_flags);
    my $OFI_CQ_DATA_EN = $op_flags[0];

    my $IPROBE_FUNCTION =
"__opal_attribute_always_inline__ static inline int
ompi_mtl_ofi_iprobe_" . $MTL_OFI_NAME_EXT . "(struct mca_mtl_base_module_t *mtl,
                struct ompi_communicator_t *comm,
                int src,
                int tag,
                int *flag,
                struct ompi_status_public_t *status)
{
    const bool OFI_CQ_DATA = " . $OFI_CQ_DATA_EN . ";

    return ompi_mtl_ofi_iprobe_generic(mtl, comm, src, tag,
                                    flag, status,
                                    OFI_CQ_DATA);
}";
    return $IPROBE_FUNCTION;
}

sub gen_iprobe_sym_init {
    my @op_flags = @{$_[0]};
    my $MTL_OFI_FUNC_NAME = "ompi_mtl_ofi_iprobe_" . opt_common::mtl_ofi_opt_common::gen_flags_ext(\@op_flags) . "";
    my $OFI_CQ_DATA_EN = $op_flags[0];
    my $symbol_init =
"
    sym_table->ompi_mtl_ofi_iprobe[".$OFI_CQ_DATA_EN."]
        = ".$MTL_OFI_FUNC_NAME.";
";
    return $symbol_init;
}

1;
