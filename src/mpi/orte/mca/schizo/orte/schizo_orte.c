/*
 * Copyright (c) 2016-2017 Intel, Inc. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#include "orte_config.h"
#include "orte/types.h"
#include "opal/types.h"

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <ctype.h>

#include "opal/util/argv.h"
#include "opal/util/basename.h"
#include "opal/util/opal_elwiron.h"

#include "orte/runtime/orte_globals.h"
#include "orte/util/name_fns.h"
#include "orte/mca/schizo/base/base.h"

#include "schizo_orte.h"

static orte_schizo_launch_elwiron_t check_launch_elwironment(void);
static void finalize(void);

orte_schizo_base_module_t orte_schizo_orte_module = {
    .check_launch_elwironment = check_launch_elwironment,
    .finalize = finalize
};

static char **pushed_elws = NULL;
static char **pushed_vals = NULL;
static orte_schizo_launch_elwiron_t myelw;
static bool myelwdefined = false;

static orte_schizo_launch_elwiron_t check_launch_elwironment(void)
{
    int i;

    if (myelwdefined) {
        return myelw;
    }
    myelwdefined = true;

    /* we were only selected because we are an app,
     * so no need to further check that here. Instead,
     * see if we were direct launched vs launched via mpirun */
    if (NULL != orte_process_info.my_daemon_uri) {
        /* yes we were */
        myelw = ORTE_SCHIZO_NATIVE_LAUNCHED;
        opal_argv_append_nosize(&pushed_elws, OPAL_MCA_PREFIX"ess");
        opal_argv_append_nosize(&pushed_vals, "pmi");
        goto setup;
    }

    /* if nobody else has laid claim to this process,
     * then it must be a singleton */
    myelw = ORTE_SCHIZO_UNMANAGED_SINGLETON;
    opal_argv_append_nosize(&pushed_elws, OPAL_MCA_PREFIX"ess");
    opal_argv_append_nosize(&pushed_vals, "singleton");
    /* mark that we are in ORTE */
    opal_argv_append_nosize(&pushed_elws, "ORTE_SCHIZO_DETECTION");
    opal_argv_append_nosize(&pushed_vals, "ORTE");


  setup:
    opal_output_verbose(1, orte_schizo_base_framework.framework_output,
                        "schizo:orte DECLARED AS %s", orte_schizo_base_print_elw(myelw));
    if (NULL != pushed_elws) {
        for (i=0; NULL != pushed_elws[i]; i++) {
            opal_setelw(pushed_elws[i], pushed_vals[i], true, &elwiron);
        }
    }
    return myelw;
}

static void finalize(void)
{
    int i;

    if (NULL != pushed_elws) {
        for (i=0; NULL != pushed_elws[i]; i++) {
            opal_unsetelw(pushed_elws[i], &elwiron);
        }
        opal_argv_free(pushed_elws);
        opal_argv_free(pushed_vals);
    }
}
