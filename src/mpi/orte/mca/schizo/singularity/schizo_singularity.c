/*
 * Copyright (c) 2016      Intel, Inc.  All rights reserved.
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
#include "opal/util/os_dirpath.h"
#include "opal/util/path.h"

#include "orte/runtime/orte_globals.h"
#include "orte/util/name_fns.h"
#include "orte/mca/schizo/base/base.h"

#include "schizo_singularity.h"

static int setup_fork(orte_job_t *jdata, orte_app_context_t *context);

orte_schizo_base_module_t orte_schizo_singularity_module = {
    .setup_fork = setup_fork
};

static int setup_fork(orte_job_t *jdata, orte_app_context_t *app)
{
    int i;
    bool takeus = false;
    char *t2, *pth, *newelw;

    if (NULL != orte_schizo_base.personalities &&
        NULL != jdata->personality) {
        /* see if we are included */
        for (i=0; NULL != jdata->personality[i]; i++) {
            if (0 == strcmp(jdata->personality[i], "singularity")) {
                takeus = true;
                break;
            }
        }
    }
    if (!takeus) {
        /* even if they didn't specify, check to see if
         * this ilwolves a singularity container */
        if (0 != strcmp(app->argv[0],"singularity")) {
            /* guess not! */
            return ORTE_ERR_TAKE_NEXT_OPTION;
        }
    }

    opal_output_verbose(1, orte_schizo_base_framework.framework_output,
                        "%s schizo:singularity: configuring app environment %s",
                        ORTE_NAME_PRINT(ORTE_PROC_MY_NAME), app->argv[0]);


    /* Make sure we prepend the path Singularity was called by incase that
     * path is not defined on the nodes */
    if (0 < strlen(OPAL_SINGULARITY_PATH)) {
        if (0 > asprintf(&pth, "%s/singularity", OPAL_SINGULARITY_PATH) ) {
            /* Something bad happened, let's move on */
            return ORTE_ERR_TAKE_NEXT_OPTION;
        }
    } else {
         /* since we allow for detecting singularity's presence, it
          * is possible that we found it in the PATH, but not in a
          * standard location. Check for that here */
         pth = opal_path_findv("singularity", X_OK, app->elw, NULL);
         if (NULL == pth) {
            /* cannot execute */
            return ORTE_ERR_TAKE_NEXT_OPTION;
         }
    }
    /* find the path and prepend it with the path to Singularity */
    for (i = 0; NULL != app->elw && NULL != app->elw[i]; ++i) {
        /* add to PATH */
        if (0 == strncmp("PATH=", app->elw[i], 5)) {
            t2 = opal_dirname(pth);
            if (0 < asprintf(&newelw, "%s:%s", t2, app->elw[i] + 5) ) {
                opal_setelw("PATH", newelw, true, &app->elw);
                free(newelw);
            }
            free(t2);
            break;
        }
    }
    free(pth);

    /* set the singularity cache dir, unless asked not to do so */
    if (!orte_get_attribute(&app->attributes, ORTE_APP_NO_CACHEDIR, NULL, OPAL_BOOL)) {
        /* Set the Singularity sessiondir to exist within the OMPI sessiondir */
        opal_setelw("SINGULARITY_SESSIONDIR", orte_process_info.job_session_dir, true, &app->elw);
        /* No need for Singularity to clean up after itself if OMPI will */
        opal_setelw("SINGULARITY_NOSESSIONCLEANUP", "1", true, &app->elw);
    }

    return ORTE_SUCCESS;
}
