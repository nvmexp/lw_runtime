/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2010      Oracle and/or its affiliates.  All rights reserved.
 * Copyright (c) 2011      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013-2018 Intel, Inc.  All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2016-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#include "orte_config.h"
#include "orte/constants.h"

#include <string.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <signal.h>
#include <errno.h>

#include "opal/hash_string.h"
#include "opal/util/arch.h"
#include "opal/util/argv.h"
#include "opal/util/opal_elwiron.h"
#include "opal/util/path.h"
#include "opal/util/timings.h"
#include "opal/runtime/opal_progress_threads.h"
#include "opal/mca/installdirs/installdirs.h"
#include "opal/mca/pmix/base/base.h"
#include "opal/mca/pmix/pmix.h"

#include "orte/util/show_help.h"
#include "orte/util/proc_info.h"
#include "orte/mca/errmgr/base/base.h"
#include "orte/mca/filem/base/base.h"
#include "orte/mca/plm/base/base.h"
#include "orte/mca/rml/base/rml_contact.h"
#include "orte/mca/state/base/base.h"
#include "orte/util/name_fns.h"
#include "orte/runtime/orte_globals.h"
#include "orte/util/session_dir.h"
#include "orte/util/pre_condition_transports.h"

#include "orte/mca/ess/ess.h"
#include "orte/mca/ess/base/base.h"
#include "orte/mca/ess/singleton/ess_singleton.h"


static int rte_init(void);
static int rte_finalize(void);
static void rte_abort(int status, bool report);

orte_ess_base_module_t orte_ess_singleton_module = {
    rte_init,
    rte_finalize,
    rte_abort,
    NULL /* ft_event */
};

extern char *orte_ess_singleton_server_uri;
static bool added_transport_keys=false;
static bool added_num_procs = false;
static bool added_app_ctx = false;
static bool added_pmix_elws = false;
static bool progress_thread_running = false;

static int fork_hnp(void);

static int rte_init(void)
{
    int rc, ret;
    char *error = NULL;
    int u32, *u32ptr;
    uint16_t u16, *u16ptr;
    orte_process_name_t name;

    /* run the prolog */
    if (ORTE_SUCCESS != (rc = orte_ess_base_std_prolog())) {
        ORTE_ERROR_LOG(rc);
        return rc;
    }
    u32ptr = &u32;
    u16ptr = &u16;

    if (NULL != mca_ess_singleton_component.server_uri) {
        /* we are going to connect to a server HNP */
        if (0 == strncmp(mca_ess_singleton_component.server_uri, "file", strlen("file")) ||
            0 == strncmp(mca_ess_singleton_component.server_uri, "FILE", strlen("FILE"))) {
            char input[1024], *filename;
            FILE *fp;

            /* it is a file - get the filename */
            filename = strchr(mca_ess_singleton_component.server_uri, ':');
            if (NULL == filename) {
                /* filename is not correctly formatted */
                orte_show_help("help-orterun.txt", "orterun:ompi-server-filename-bad", true,
                               "singleton", mca_ess_singleton_component.server_uri);
                return ORTE_ERROR;
            }
            ++filename; /* space past the : */

            if (0 >= strlen(filename)) {
                /* they forgot to give us the name! */
                orte_show_help("help-orterun.txt", "orterun:ompi-server-filename-missing", true,
                               "singleton", mca_ess_singleton_component.server_uri);
                return ORTE_ERROR;
            }

            /* open the file and extract the uri */
            fp = fopen(filename, "r");
            if (NULL == fp) { /* can't find or read file! */
                orte_show_help("help-orterun.txt", "orterun:ompi-server-filename-access", true,
                               "singleton", mca_ess_singleton_component.server_uri);
                return ORTE_ERROR;
            }
            memset(input, 0, 1024);  // initialize the array to ensure a NULL termination
            if (NULL == fgets(input, 1023, fp)) {
                /* something malformed about file */
                fclose(fp);
                orte_show_help("help-orterun.txt", "orterun:ompi-server-file-bad", true,
                               "singleton", mca_ess_singleton_component.server_uri, "singleton");
                return ORTE_ERROR;
            }
            fclose(fp);
            input[strlen(input)-1] = '\0';  /* remove newline */
            orte_process_info.my_hnp_uri = strdup(input);
        } else {
            orte_process_info.my_hnp_uri = strdup(mca_ess_singleton_component.server_uri);
        }
        /* save the daemon uri - we will process it later */
        orte_process_info.my_daemon_uri = strdup(orte_process_info.my_hnp_uri);
        /* construct our name - we are in their job family, so we know that
         * much. However, we cannot know how many other singletons and jobs
         * this HNP is running. Oh well - if someone really wants to use this
         * option, they can try to figure it out. For now, we'll just assume
         * we are the only ones */
        ORTE_PROC_MY_NAME->jobid = ORTE_CONSTRUCT_LOCAL_JOBID(ORTE_PROC_MY_HNP->jobid, 1);
        /* obviously, we are vpid=0 for this job */
        ORTE_PROC_MY_NAME->vpid = 0;

        /* for colwenience, push the pubsub version of this param into the elwiron */
        opal_setelw (OPAL_MCA_PREFIX"pubsub_orte_server", orte_process_info.my_hnp_uri, true, &elwiron);
    } else if (NULL != getelw("SINGULARITY_CONTAINER") ||
               mca_ess_singleton_component.isolated) {
        /* ensure we use the isolated pmix component */
        opal_setelw(OPAL_MCA_PREFIX"pmix", "isolated", true, &elwiron);
    } else {
        /* we want to use PMIX_NAMESPACE that will be sent by the hnp as a jobid */
        opal_setelw(OPAL_MCA_PREFIX"orte_launch", "1", true, &elwiron);
        /* spawn our very own HNP to support us */
        if (ORTE_SUCCESS != (rc = fork_hnp())) {
            ORTE_ERROR_LOG(rc);
            return rc;
        }
        /* our name was given to us by the HNP */
        opal_setelw(OPAL_MCA_PREFIX"pmix", "^s1,s2,cray,isolated", true, &elwiron);
    }

    /* get an async event base - we use the opal_async one so
     * we don't startup extra threads if not needed */
    orte_event_base = opal_progress_thread_init(NULL);
    progress_thread_running = true;

    /* open and setup pmix */
    if (OPAL_SUCCESS != (ret = mca_base_framework_open(&opal_pmix_base_framework, 0))) {
        error = "opening pmix";
        goto error;
    }
    if (OPAL_SUCCESS != (ret = opal_pmix_base_select())) {
        error = "select pmix";
        goto error;
    }
    /* set the event base */
    opal_pmix_base_set_evbase(orte_event_base);
    /* initialize the selected module */
    if (!opal_pmix.initialized() && (OPAL_SUCCESS != (ret = opal_pmix.init(NULL)))) {
        /* we cannot run */
        error = "pmix init";
        goto error;
    }

    /* pmix.init set our process name down in the OPAL layer,
     * so carry it forward here */
    ORTE_PROC_MY_NAME->jobid = OPAL_PROC_MY_NAME.jobid;
    ORTE_PROC_MY_NAME->vpid = OPAL_PROC_MY_NAME.vpid;
    name.jobid = OPAL_PROC_MY_NAME.jobid;
    name.vpid = ORTE_VPID_WILDCARD;

    /* get our local rank from PMI */
    OPAL_MODEX_RECV_VALUE(ret, OPAL_PMIX_LOCAL_RANK,
                          ORTE_PROC_MY_NAME, &u16ptr, OPAL_UINT16);
    if (OPAL_SUCCESS != ret) {
        error = "getting local rank";
        goto error;
    }
    orte_process_info.my_local_rank = u16;

    /* get our node rank from PMI */
    OPAL_MODEX_RECV_VALUE(ret, OPAL_PMIX_NODE_RANK,
                          ORTE_PROC_MY_NAME, &u16ptr, OPAL_UINT16);
    if (OPAL_SUCCESS != ret) {
        error = "getting node rank";
        goto error;
    }
    orte_process_info.my_node_rank = u16;

    /* get max procs */
    OPAL_MODEX_RECV_VALUE(ret, OPAL_PMIX_MAX_PROCS,
                          &name, &u32ptr, OPAL_UINT32);
    if (OPAL_SUCCESS != ret) {
        error = "getting max procs";
        goto error;
    }
    orte_process_info.max_procs = u32;

    /* we are a singleton, so there is only one proc in the job */
    orte_process_info.num_procs = 1;
    /* push into the elwiron for pickup in MPI layer for
     * MPI-3 required info key
     */
    if (NULL == getelw(OPAL_MCA_PREFIX"orte_ess_num_procs")) {
        char * num_procs;
        asprintf(&num_procs, "%d", orte_process_info.num_procs);
        opal_setelw(OPAL_MCA_PREFIX"orte_ess_num_procs", num_procs, true, &elwiron);
        free(num_procs);
        added_num_procs = true;
    }
    if (NULL == getelw("OMPI_APP_CTX_NUM_PROCS")) {
        char * num_procs;
        asprintf(&num_procs, "%d", orte_process_info.num_procs);
        opal_setelw("OMPI_APP_CTX_NUM_PROCS", num_procs, true, &elwiron);
        free(num_procs);
        added_app_ctx = true;
    }


    /* get our app number from PMI - ok if not found */
    OPAL_MODEX_RECV_VALUE_OPTIONAL(ret, OPAL_PMIX_APPNUM,
                          ORTE_PROC_MY_NAME, &u32ptr, OPAL_UINT32);
    if (OPAL_SUCCESS == ret) {
        orte_process_info.app_num = u32;
    } else {
        orte_process_info.app_num = 0;
    }
    /* set some other standard values */
    orte_process_info.num_local_peers = 0;

    /* setup transport keys in case the MPI layer needs them -
     * we can use the jobfam and stepid as unique keys
     * because they are unique values assigned by the RM
     */
    if (NULL == getelw(OPAL_MCA_PREFIX"orte_precondition_transports")) {
        char *key;
        ret = orte_pre_condition_transports(NULL, &key);
        if (ORTE_SUCCESS == ret) {
            opal_setelw(OPAL_MCA_PREFIX"orte_precondition_transports", key, true, &elwiron);
            free(key);
        }
    }

    /* now that we have all required info, complete the setup */
    /*
     * stdout/stderr buffering
     * If the user requested to override the default setting then do
     * as they wish.
     */
    if( orte_ess_base_std_buffering > -1 ) {
        if( 0 == orte_ess_base_std_buffering ) {
            setvbuf(stdout, NULL, _IONBF, 0);
            setvbuf(stderr, NULL, _IONBF, 0);
        }
        else if( 1 == orte_ess_base_std_buffering ) {
            setvbuf(stdout, NULL, _IOLBF, 0);
            setvbuf(stderr, NULL, _IOLBF, 0);
        }
        else if( 2 == orte_ess_base_std_buffering ) {
            setvbuf(stdout, NULL, _IOFBF, 0);
            setvbuf(stderr, NULL, _IOFBF, 0);
        }
    }

    /* if I am an MPI app, we will let the MPI layer define and
     * control the opal_proc_t structure. Otherwise, we need to
     * do so here */
    if (ORTE_PROC_NON_MPI) {
        orte_process_info.super.proc_name = *(opal_process_name_t*)ORTE_PROC_MY_NAME;
        orte_process_info.super.proc_hostname = orte_process_info.nodename;
        orte_process_info.super.proc_flags = OPAL_PROC_ALL_LOCAL;
        orte_process_info.super.proc_arch = opal_local_arch;
        opal_proc_local_set(&orte_process_info.super);
    }

    /* open and setup the state machine */
    if (ORTE_SUCCESS != (ret = mca_base_framework_open(&orte_state_base_framework, 0))) {
        ORTE_ERROR_LOG(ret);
        error = "orte_state_base_open";
        goto error;
    }
    if (ORTE_SUCCESS != (ret = orte_state_base_select())) {
        ORTE_ERROR_LOG(ret);
        error = "orte_state_base_select";
        goto error;
    }

    /* open the errmgr */
    if (ORTE_SUCCESS != (ret = mca_base_framework_open(&orte_errmgr_base_framework, 0))) {
        ORTE_ERROR_LOG(ret);
        error = "orte_errmgr_base_open";
        goto error;
    }

    /* setup my session directory */
    if (orte_create_session_dirs) {
        OPAL_OUTPUT_VERBOSE((2, orte_ess_base_framework.framework_output,
                             "%s setting up session dir with\n\ttmpdir: %s\n\thost %s",
                             ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                             (NULL == orte_process_info.tmpdir_base) ? "UNDEF" : orte_process_info.tmpdir_base,
                             orte_process_info.nodename));
        if (ORTE_SUCCESS != (ret = orte_session_dir(true, ORTE_PROC_MY_NAME))) {
            ORTE_ERROR_LOG(ret);
            error = "orte_session_dir";
            goto error;
        }
        /* Once the session directory location has been established, set
           the opal_output elw file location to be in the
           proc-specific session directory. */
        opal_output_set_output_file_info(orte_process_info.proc_session_dir,
                                         "output-", NULL, NULL);
        /* register the directory for cleanup */
        if (NULL != opal_pmix.register_cleanup) {
            if (orte_standalone_operation) {
                if (OPAL_SUCCESS != (ret = opal_pmix.register_cleanup(orte_process_info.top_session_dir, true, false, true))) {
                    ORTE_ERROR_LOG(ret);
                    error = "register cleanup";
                    goto error;
                }
            } else {
                if (OPAL_SUCCESS != (ret = opal_pmix.register_cleanup(orte_process_info.job_session_dir, true, false, false))) {
                    ORTE_ERROR_LOG(ret);
                    error = "register cleanup";
                    goto error;
                }
            }
        }
    }

    /* if we have info on the HNP and local daemon, process it */
    if (NULL != orte_process_info.my_hnp_uri) {
        /* we have to set the HNP's name, even though we won't route messages directly
         * to it. This is required to ensure that we -do- send messages to the correct
         * HNP name
         */
        if (ORTE_SUCCESS != (ret = orte_rml_base_parse_uris(orte_process_info.my_hnp_uri,
                                                            ORTE_PROC_MY_HNP, NULL))) {
            ORTE_ERROR_LOG(ret);
            error = "orte_rml_parse_HNP";
            goto error;
        }
    }
    if (NULL != orte_process_info.my_daemon_uri) {
        opal_value_t val;

        /* extract the daemon's name so we can update the routing table */
        if (ORTE_SUCCESS != (ret = orte_rml_base_parse_uris(orte_process_info.my_daemon_uri,
                                                            ORTE_PROC_MY_DAEMON, NULL))) {
            ORTE_ERROR_LOG(ret);
            error = "orte_rml_parse_daemon";
            goto error;
        }
        /* Set the contact info in the database - this won't actually establish
         * the connection, but just tells us how to reach the daemon
         * if/when we attempt to send to it
         */
        OBJ_CONSTRUCT(&val, opal_value_t);
        val.key = OPAL_PMIX_PROC_URI;
        val.type = OPAL_STRING;
        val.data.string = orte_process_info.my_daemon_uri;
        if (OPAL_SUCCESS != (ret = opal_pmix.store_local(ORTE_PROC_MY_DAEMON, &val))) {
            ORTE_ERROR_LOG(ret);
            val.key = NULL;
            val.data.string = NULL;
            OBJ_DESTRUCT(&val);
            error = "store DAEMON URI";
            goto error;
        }
        val.key = NULL;
        val.data.string = NULL;
        OBJ_DESTRUCT(&val);
    }

    /* setup the errmgr */
    if (ORTE_SUCCESS != (ret = orte_errmgr_base_select())) {
        ORTE_ERROR_LOG(ret);
        error = "orte_errmgr_base_select";
        goto error;
    }

    /* setup process binding */
    if (ORTE_SUCCESS != (ret = orte_ess_base_proc_binding())) {
        error = "proc_binding";
        goto error;
    }

    /* this needs to be set to enable debugger use when direct launched */
    if (NULL == orte_process_info.my_daemon_uri) {
        orte_standalone_operation = true;
    }

    /* set max procs */
    if (orte_process_info.max_procs < orte_process_info.num_procs) {
        orte_process_info.max_procs = orte_process_info.num_procs;
    }

    /* push our hostname so others can find us, if they need to - the
     * native PMIx component will ignore this request as the hostname
     * is provided by the system */
    OPAL_MODEX_SEND_VALUE(ret, OPAL_PMIX_GLOBAL, OPAL_PMIX_HOSTNAME, orte_process_info.nodename, OPAL_STRING);
    if (ORTE_SUCCESS != ret) {
        error = "db store hostname";
        goto error;
    }

    /* if we are an ORTE app - and not an MPI app - then
     * we need to exchange our connection info here.
     * MPI_Init has its own modex, so we don't need to do
     * two of them. However, if we don't do a modex at all,
     * then processes have no way to communicate
     *
     * NOTE: only do this when the process originally launches.
     * Cannot do this on a restart as the rest of the processes
     * in the job won't be exelwting this step, so we would hang
     */
    if (ORTE_PROC_IS_NON_MPI && !orte_do_not_barrier) {
        /* need to commit the data before we fence */
        opal_pmix.commit();
        if (ORTE_SUCCESS != (ret = opal_pmix.fence(NULL, 0))) {
            error = "opal_pmix.fence() failed";
            goto error;
        }
    }

    return ORTE_SUCCESS;

  error:
    if (ORTE_ERR_SILENT != ret && !orte_report_silent_errors) {
        orte_show_help("help-orte-runtime.txt",
                       "orte_init:startup:internal-failure",
                       true, error, ORTE_ERROR_NAME(ret), ret);
    }
    return ret;
}

static int rte_finalize(void)
{
    /* remove the elwars that we pushed into elwiron
     * so we leave that structure intact
     */
    if (added_transport_keys) {
        unsetelw(OPAL_MCA_PREFIX"orte_precondition_transports");
    }
    if (added_num_procs) {
        unsetelw(OPAL_MCA_PREFIX"orte_ess_num_procs");
    }
    if (added_app_ctx) {
        unsetelw("OMPI_APP_CTX_NUM_PROCS");
    }
    if (added_pmix_elws) {
        unsetelw("PMIX_NAMESPACE");
        unsetelw("PMIX_RANK");
        unsetelw("PMIX_SERVER_URI");
        unsetelw("PMIX_SELWRITY_MODE");
    }
    /* close frameworks */
    (void) mca_base_framework_close(&orte_filem_base_framework);
    (void) mca_base_framework_close(&orte_errmgr_base_framework);

    /* mark us as finalized */
    if (NULL != opal_pmix.finalize) {
        opal_pmix.finalize();
        (void) mca_base_framework_close(&opal_pmix_base_framework);
    }

    (void) mca_base_framework_close(&orte_state_base_framework);
    orte_session_dir_finalize(ORTE_PROC_MY_NAME);

    /* cleanup the process info */
    orte_proc_info_finalize();

    /* release the event base */
    if (progress_thread_running) {
        opal_progress_thread_finalize(NULL);
        progress_thread_running = false;
    }
    return ORTE_SUCCESS;
}

#define ORTE_URI_MSG_LGTH   256

static void set_handler_default(int sig)
{
    struct sigaction act;

    act.sa_handler = SIG_DFL;
    act.sa_flags = 0;
    sigemptyset(&act.sa_mask);

    sigaction(sig, &act, (struct sigaction *)0);
}

static int fork_hnp(void)
{
    int p[2], death_pipe[2];
    char *cmd;
    char **argv = NULL;
    int argc;
    char *param, *cptr;
    sigset_t sigs;
    int buffer_length, num_chars_read, chunk;
    char *orted_uri;
    int rc, i;

    /* A pipe is used to communicate between the parent and child to
       indicate whether the exec ultimately succeeded or failed.  The
       child sets the pipe to be close-on-exec; the child only ever
       writes anything to the pipe if there is an error (e.g.,
       exelwtable not found, exec() fails, etc.).  The parent does a
       blocking read on the pipe; if the pipe closed with no data,
       then the exec() succeeded.  If the parent reads something from
       the pipe, then the child was letting us know that it failed.
    */
    if (pipe(p) < 0) {
        ORTE_ERROR_LOG(ORTE_ERR_SYS_LIMITS_PIPES);
        return ORTE_ERR_SYS_LIMITS_PIPES;
    }

    /* we also have to give the HNP a pipe it can watch to know when
     * we terminated. Since the HNP is going to be a child of us, it
     * can't just use waitpid to see when we leave - so it will watch
     * the pipe instead
     */
    if (pipe(death_pipe) < 0) {
        ORTE_ERROR_LOG(ORTE_ERR_SYS_LIMITS_PIPES);
        return ORTE_ERR_SYS_LIMITS_PIPES;
    }

    /* find the orted binary using the install_dirs support - this also
     * checks to ensure that we can see this exelwtable and it *is* exelwtable by us
     */
    cmd = opal_path_access("orted", opal_install_dirs.bindir, X_OK);
    if (NULL == cmd) {
        /* guess we couldn't do it - best to abort */
        ORTE_ERROR_LOG(ORTE_ERR_FILE_NOT_EXELWTABLE);
        close(p[0]);
        close(p[1]);
        return ORTE_ERR_FILE_NOT_EXELWTABLE;
    }

    /* okay, setup an appropriate argv */
    opal_argv_append(&argc, &argv, "orted");

    /* tell the daemon it is to be the HNP */
    opal_argv_append(&argc, &argv, "--hnp");

    /* tell the daemon to get out of our process group */
    opal_argv_append(&argc, &argv, "--set-sid");

    /* tell the daemon to report back its uri so we can connect to it */
    opal_argv_append(&argc, &argv, "--report-uri");
    asprintf(&param, "%d", p[1]);
    opal_argv_append(&argc, &argv, param);
    free(param);

    /* give the daemon a pipe it can watch to tell when we have died */
    opal_argv_append(&argc, &argv, "--singleton-died-pipe");
    asprintf(&param, "%d", death_pipe[0]);
    opal_argv_append(&argc, &argv, param);
    free(param);

    /* add any debug flags */
    if (orte_debug_flag) {
        opal_argv_append(&argc, &argv, "--debug");
    }

    if (orte_debug_daemons_flag) {
        opal_argv_append(&argc, &argv, "--debug-daemons");
    }

    if (orte_debug_daemons_file_flag) {
        if (!orte_debug_daemons_flag) {
            opal_argv_append(&argc, &argv, "--debug-daemons");
        }
        opal_argv_append(&argc, &argv, "--debug-daemons-file");
    }

    /* indicate that it must use the novm state machine */
    opal_argv_append(&argc, &argv, "-"OPAL_MCA_CMD_LINE_ID);
    opal_argv_append(&argc, &argv, "state_novm_select");
    opal_argv_append(&argc, &argv, "1");

    /* direct the selection of the ess component */
    opal_argv_append(&argc, &argv, "-"OPAL_MCA_CMD_LINE_ID);
    opal_argv_append(&argc, &argv, "ess");
    opal_argv_append(&argc, &argv, "hnp");

    /* direct the selection of the pmix component */
    opal_argv_append(&argc, &argv, "-"OPAL_MCA_CMD_LINE_ID);
    opal_argv_append(&argc, &argv, "pmix");
    opal_argv_append(&argc, &argv, "^s1,s2,cray,isolated");

    /* Fork off the child */
    orte_process_info.hnp_pid = fork();
    if(orte_process_info.hnp_pid < 0) {
        ORTE_ERROR_LOG(ORTE_ERR_SYS_LIMITS_CHILDREN);
        close(p[0]);
        close(p[1]);
        close(death_pipe[0]);
        close(death_pipe[1]);
        free(cmd);
        opal_argv_free(argv);
        return ORTE_ERR_SYS_LIMITS_CHILDREN;
    }

    if (orte_process_info.hnp_pid == 0) {
        close(p[0]);
        close(death_pipe[1]);
        /* I am the child - exec me */

        /* Set signal handlers back to the default.  Do this close
           to the execve() because the event library may (and likely
           will) reset them.  If we don't do this, the event
           library may have left some set that, at least on some
           OS's, don't get reset via fork() or exec().  Hence, the
           orted could be unkillable (for example). */
        set_handler_default(SIGTERM);
        set_handler_default(SIGINT);
        set_handler_default(SIGHUP);
        set_handler_default(SIGPIPE);
        set_handler_default(SIGCHLD);

        /* Unblock all signals, for many of the same reasons that
           we set the default handlers, above.  This is noticable
           on Linux where the event library blocks SIGTERM, but we
           don't want that blocked by the orted (or, more
           specifically, we don't want it to be blocked by the
           orted and then inherited by the ORTE processes that it
           forks, making them unkillable by SIGTERM). */
        sigprocmask(0, 0, &sigs);
        sigprocmask(SIG_UNBLOCK, &sigs, 0);

        execv(cmd, argv);

        /* if I get here, the execv failed! */
        orte_show_help("help-ess-base.txt", "ess-base:execv-error",
                       true, cmd, strerror(errno));
        exit(1);

    } else {
        int count;

        free(cmd);
        /* I am the parent - wait to hear something back and
         * report results
         */
        close(p[1]);  /* parent closes the write - orted will write its contact info to it*/
        close(death_pipe[0]);  /* parent closes the death_pipe's read */
        opal_argv_free(argv);

        /* setup the buffer to read the HNP's uri */
        buffer_length = ORTE_URI_MSG_LGTH;
        chunk = ORTE_URI_MSG_LGTH-1;
        num_chars_read = 0;
        orted_uri = (char*)malloc(buffer_length);
        memset(orted_uri, 0, buffer_length);

        while (0 != (rc = read(p[0], &orted_uri[num_chars_read], chunk))) {
            if (rc < 0 && (EAGAIN == errno || EINTR == errno)) {
                continue;
            } else if (rc < 0) {
                num_chars_read = -1;
                break;
            }
            /* we read something - better get more */
            num_chars_read += rc;
            chunk -= rc;
            if (0 == chunk) {
                chunk = ORTE_URI_MSG_LGTH;
                orted_uri = realloc((void*)orted_uri, buffer_length+chunk);
                memset(&orted_uri[buffer_length], 0, chunk);
                buffer_length += chunk;
            }
        }
        close(p[0]);

        if (num_chars_read <= 0) {
            /* we didn't get anything back - this is bad */
            ORTE_ERROR_LOG(ORTE_ERR_HNP_COULD_NOT_START);
            free(orted_uri);
            return ORTE_ERR_HNP_COULD_NOT_START;
        }

        /* parse the sysinfo from the returned info - must
         * start from the end of the string as the uri itself
         * can contain brackets */
        if (NULL == (param = strrchr(orted_uri, '['))) {
            ORTE_ERROR_LOG(ORTE_ERR_COMM_FAILURE);
            free(orted_uri);
            return ORTE_ERR_COMM_FAILURE;
        }
        *param = '\0'; /* terminate the uri string */
        ++param;  /* point to the start of the sysinfo */

        /* find the end of the sysinfo */
        if (NULL == (cptr = strchr(param, ']'))) {
            ORTE_ERROR_LOG(ORTE_ERR_COMM_FAILURE);
            free(orted_uri);
            return ORTE_ERR_COMM_FAILURE;
        }
        *cptr = '\0';  /* terminate the sysinfo string */
        ++cptr;  /* point to the start of the pmix uri */

        /* colwert the sysinfo string */
        if (ORTE_SUCCESS != (rc = orte_util_colwert_string_to_sysinfo(&orte_local_cpu_type,
                                      &orte_local_cpu_model, param))) {
            ORTE_ERROR_LOG(rc);
            free(orted_uri);
            return rc;
        }

        /* save the daemon uri - we will process it later */
        orte_process_info.my_daemon_uri = strdup(orted_uri);
        /* likewise, since this is also the HNP, set that uri too */
        orte_process_info.my_hnp_uri = orted_uri;

        /* split the pmix_uri into its parts */
        argv = opal_argv_split(cptr, '*');
        count = opal_argv_count(argv);
        /* push each piece into the environment */
        for (i=0; i < count; i++) {
            char *c = strchr(argv[i], '=');
            assert(NULL != c);
            *c++ = '\0';
            opal_setelw(argv[i], c, true, &elwiron);
        }
        opal_argv_free(argv);
        added_pmix_elws = true;

        /* all done - report success */
        return ORTE_SUCCESS;
    }
}

static void rte_abort(int status, bool report)
{
    struct timespec tp = {0, 100000};

    OPAL_OUTPUT_VERBOSE((1, orte_ess_base_framework.framework_output,
                         "%s ess:singleton:abort: abort with status %d",
                         ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),
                         status));

    /* PMI doesn't like NULL messages, but our interface
     * doesn't provide one - so rig one up here
     */
    opal_pmix.abort(status, "N/A", NULL);

    /* provide a little delay for the PMIx thread to
     * get the info out */
    nanosleep(&tp, NULL);

    /* Now Exit */
    _exit(status);
}
