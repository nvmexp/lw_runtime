/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2018      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_topo.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/op/op.h"
#include "coll_tuned.h"

/* scan algorithm variables */
static int coll_tuned_scan_forced_algorithm = 0;

/* valid values for coll_tuned_scan_forced_algorithm */
static mca_base_var_enum_value_t scan_algorithms[] = {
    {0, "ignore"},
    {1, "linear"},
    {2, "relwrsive_doubling"},
    {0, NULL}
};

/**
 * The following are used by dynamic and forced rules
 *
 * publish details of each algorithm and if its forced/fixed/locked in
 * as you add methods/algorithms you must update this and the query/map routines
 *
 * this routine is called by the component only
 * this makes sure that the mca parameters are set to their initial values and
 * perms module does not call this they call the forced_getvalues routine
 * instead.
 */

int ompi_coll_tuned_scan_intra_check_forced_init (coll_tuned_force_algorithm_mca_param_indices_t *mca_param_indices)
{
    mca_base_var_enum_t*new_enum;
    int cnt;

    for( cnt = 0; NULL != scan_algorithms[cnt].string; cnt++ );
    ompi_coll_tuned_forced_max_algorithms[SCAN] = cnt;

    (void) mca_base_component_var_register(&mca_coll_tuned_component.super.collm_version,
                                           "scan_algorithm_count",
                                           "Number of scan algorithms available",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                           MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                           OPAL_INFO_LVL_5,
                                           MCA_BASE_VAR_SCOPE_CONSTANT,
                                           &ompi_coll_tuned_forced_max_algorithms[SCAN]);

    /* MPI_T: This variable should eventually be bound to a communicator */
    coll_tuned_scan_forced_algorithm = 0;
    (void) mca_base_var_enum_create("coll_tuned_scan_algorithms", scan_algorithms, &new_enum);
    mca_param_indices->algorithm_param_index =
        mca_base_component_var_register(&mca_coll_tuned_component.super.collm_version,
                                        "scan_algorithm",
                                        "Which scan algorithm is used. Can be locked down to choice of: 0 ignore, 1 linear, 2 relwrsive_doubling",
                                        MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                        OPAL_INFO_LVL_5,
                                        MCA_BASE_VAR_SCOPE_ALL,
                                        &coll_tuned_scan_forced_algorithm);
    OBJ_RELEASE(new_enum);
    if (mca_param_indices->algorithm_param_index < 0) {
        return mca_param_indices->algorithm_param_index;
    }

    return (MPI_SUCCESS);
}

int ompi_coll_tuned_scan_intra_do_this(const void *sbuf, void* rbuf, int count,
                                         struct ompi_datatype_t *dtype,
                                         struct ompi_op_t *op,
                                         struct ompi_communicator_t *comm,
                                         mca_coll_base_module_t *module,
                                         int algorithm)
{
    OPAL_OUTPUT((ompi_coll_tuned_stream,"coll:tuned:scan_intra_do_this selected algorithm %d",
                 algorithm));

    switch (algorithm) {
    case (0):
    case (1):  return ompi_coll_base_scan_intra_linear(sbuf, rbuf, count, dtype,
                                                       op, comm, module);
    case (2):  return ompi_coll_base_scan_intra_relwrsivedoubling(sbuf, rbuf, count, dtype,
                                                                  op, comm, module);
    } /* switch */
    OPAL_OUTPUT((ompi_coll_tuned_stream,"coll:tuned:scan_intra_do_this attempt to select algorithm %d when only 0-%d is valid?",
                 algorithm, ompi_coll_tuned_forced_max_algorithms[SCAN]));
    return (MPI_ERR_ARG);
}
