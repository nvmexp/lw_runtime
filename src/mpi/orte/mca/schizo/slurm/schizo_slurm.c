/*
 * Copyright (c) 2016-2017 Intel, Inc. All rights reserved.
 * Copyright (c) 2016      Mellanox Technologies Ltd.  All rights reserved.
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

#include "schizo_slurm.h"

static orte_schizo_launch_elwiron_t check_launch_elwironment(void);
static int get_remaining_time(uint32_t *timeleft);
static void finalize(void);

orte_schizo_base_module_t orte_schizo_slurm_module = {
    .check_launch_elwironment = check_launch_elwironment,
    .get_remaining_time = get_remaining_time,
    .finalize = finalize
};

static char **pushed_elws = NULL;
static char **pushed_vals = NULL;
static orte_schizo_launch_elwiron_t myelw;
static bool myelwdefined = false;

static orte_schizo_launch_elwiron_t check_launch_elwironment(void)
{
    char *bind, *list, *ptr;
    int i;

    if (myelwdefined) {
        return myelw;
    }
    myelwdefined = true;

    /* we were only selected because SLURM was detected
     * and we are an app, so no need to further check
     * that here. Instead, see if we were direct launched
     * vs launched via mpirun */
    if (NULL != orte_process_info.my_daemon_uri) {
        /* nope */
        myelw = ORTE_SCHIZO_NATIVE_LAUNCHED;
        opal_argv_append_nosize(&pushed_elws, OPAL_MCA_PREFIX"ess");
        opal_argv_append_nosize(&pushed_vals, "pmi");
        /* mark that we are native */
        opal_argv_append_nosize(&pushed_elws, "ORTE_SCHIZO_DETECTION");
        opal_argv_append_nosize(&pushed_vals, "NATIVE");
        goto setup;
    }

    /* see if we are in a SLURM allocation */
    if (NULL == getelw("SLURM_NODELIST")) {
        /* nope */
        myelw = ORTE_SCHIZO_UNDETERMINED;
        return myelw;
    }

    /* mark that we are in SLURM */
    opal_argv_append_nosize(&pushed_elws, "ORTE_SCHIZO_DETECTION");
    opal_argv_append_nosize(&pushed_vals, "SLURM");

    /* we are in an allocation, but were we direct launched
     * or are we a singleton? */
    if (NULL == getelw("SLURM_STEP_ID")) {
        /* not in a job step - ensure we select the
         * correct things */
        opal_argv_append_nosize(&pushed_elws, OPAL_MCA_PREFIX"ess");
        opal_argv_append_nosize(&pushed_vals, "singleton");
        myelw = ORTE_SCHIZO_MANAGED_SINGLETON;
        goto setup;
    }
    myelw = ORTE_SCHIZO_DIRECT_LAUNCHED;
    opal_argv_append_nosize(&pushed_elws, OPAL_MCA_PREFIX"ess");
    opal_argv_append_nosize(&pushed_vals, "pmi");

    /* if we are direct-launched by SLURM, then disable binding */
    opal_argv_append_nosize(&pushed_elws, OPAL_MCA_PREFIX"hwloc_base_binding_policy");
    opal_argv_append_nosize(&pushed_vals, "none");
    /* indicate we are externally bound so we won't try to do it ourselves */
    opal_argv_append_nosize(&pushed_elws, OPAL_MCA_PREFIX"orte_externally_bound");
    opal_argv_append_nosize(&pushed_vals, "1");

  setup:
      opal_output_verbose(1, orte_schizo_base_framework.framework_output,
                          "schizo:slurm DECLARED AS %s", orte_schizo_base_print_elw(myelw));
    if (NULL != pushed_elws) {
        for (i=0; NULL != pushed_elws[i]; i++) {
            opal_setelw(pushed_elws[i], pushed_vals[i], true, &elwiron);
        }
    }
    return myelw;
}

static int get_remaining_time(uint32_t *timeleft)
{
    char output[256], *cmd, *jobid, **res;
    FILE *fp;
    uint32_t tleft;
    size_t cnt;

    /* set the default */
    *timeleft = UINT32_MAX;

    if (NULL == (jobid = getelw("SLURM_JOBID"))) {
        return ORTE_ERR_TAKE_NEXT_OPTION;
    }
    if (0 > asprintf(&cmd, "squeue -h -j %s -o %%L", jobid)) {
        return ORTE_ERR_OUT_OF_RESOURCE;
    }
    fp = popen(cmd, "r");
    if (NULL == fp) {
        free(cmd);
        return ORTE_ERR_FILE_OPEN_FAILURE;
    }
    if (NULL == fgets(output, 256, fp)) {
        free(cmd);
        pclose(fp);
        return ORTE_ERR_FILE_READ_FAILURE;
    }
    free(cmd);
    pclose(fp);
    /* the output is returned in a colon-delimited set of fields */
    res = opal_argv_split(output, ':');
    cnt =  opal_argv_count(res);
    tleft = strtol(res[cnt-1], NULL, 10); // has to be at least one field
    /* the next field would be minutes */
    if (1 < cnt) {
        tleft += 60 * strtol(res[cnt-2], NULL, 10);
    }
    /* next field would be hours */
    if (2 < cnt) {
        tleft += 3600 * strtol(res[cnt-3], NULL, 10);
    }
    /* next field is days */
    if (3 < cnt) {
        tleft += 24*3600 * strtol(res[cnt-4], NULL, 10);
    }
    /* if there are more fields than that, then it is infinite */
    if (4 < cnt) {
        tleft = UINT32_MAX;
    }
    opal_argv_free(res);

    *timeleft = tleft;
    return ORTE_SUCCESS;
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
