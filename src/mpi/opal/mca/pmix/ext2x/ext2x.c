/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2014-2018 Intel, Inc. All rights reserved.
 * Copyright (c) 2014-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2014-2015 Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2016      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2017      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"
#include "opal/constants.h"
#include "opal/types.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "opal/dss/dss.h"
#include "opal/mca/event/event.h"
#include "opal/mca/hwloc/base/base.h"
#include "opal/runtime/opal.h"
#include "opal/runtime/opal_progress_threads.h"
#include "opal/threads/threads.h"
#include "opal/util/argv.h"
#include "opal/util/error.h"
#include "opal/util/opal_elwiron.h"
#include "opal/util/output.h"
#include "opal/util/proc.h"
#include "opal/util/show_help.h"

#include "ext2x.h"
#include "opal/mca/pmix/base/base.h"
#include "opal/mca/pmix/pmix_types.h"

#include <pmix_common.h>
#include <pmix.h>

/****    C.O.M.M.O.N   I.N.T.E.R.F.A.C.E.S     ****/

/* These are functions used by both client and server to
 * access common functions in the embedded PMIx library */
static bool legacy_get(void);
static const char *ext2x_get_nspace(opal_jobid_t jobid);
static void ext2x_register_jobid(opal_jobid_t jobid, const char *nspace);
static void register_handler(opal_list_t *event_codes,
                             opal_list_t *info,
                             opal_pmix_notification_fn_t evhandler,
                             opal_pmix_evhandler_reg_cbfunc_t cbfunc,
                             void *cbdata);
static void deregister_handler(size_t evhandler,
                               opal_pmix_op_cbfunc_t cbfunc,
                               void *cbdata);
static int notify_event(int status,
                        const opal_process_name_t *source,
                        opal_pmix_data_range_t range,
                        opal_list_t *info,
                        opal_pmix_op_cbfunc_t cbfunc, void *cbdata);
static void ext2x_query(opal_list_t *queries,
                         opal_pmix_info_cbfunc_t cbfunc, void *cbdata);
static void ext2x_log(opal_list_t *info,
                       opal_pmix_op_cbfunc_t cbfunc, void *cbdata);

const opal_pmix_base_module_t opal_pmix_ext2x_module = {
    .legacy_get = legacy_get,
    /* client APIs */
    .init = ext2x_client_init,
    .finalize = ext2x_client_finalize,
    .initialized = ext2x_initialized,
    .abort = ext2x_abort,
    .commit = ext2x_commit,
    .fence = ext2x_fence,
    .fence_nb = ext2x_fencenb,
    .put = ext2x_put,
    .get = ext2x_get,
    .get_nb = ext2x_getnb,
    .publish = ext2x_publish,
    .publish_nb = ext2x_publishnb,
    .lookup = ext2x_lookup,
    .lookup_nb = ext2x_lookupnb,
    .unpublish = ext2x_unpublish,
    .unpublish_nb = ext2x_unpublishnb,
    .spawn = ext2x_spawn,
    .spawn_nb = ext2x_spawnnb,
    .connect = ext2x_connect,
    .connect_nb = ext2x_connectnb,
    .disconnect = ext2x_disconnect,
    .disconnect_nb = ext2x_disconnectnb,
    .resolve_peers = ext2x_resolve_peers,
    .resolve_nodes = ext2x_resolve_nodes,
    .query = ext2x_query,
    .log = ext2x_log,
    .allocate = ext2x_allocate,
    .job_control = ext2x_job_control,
    /* server APIs */
    .server_init = ext2x_server_init,
    .server_finalize = ext2x_server_finalize,
    .generate_regex = ext2x_server_gen_regex,
    .generate_ppn = ext2x_server_gen_ppn,
    .server_register_nspace = ext2x_server_register_nspace,
    .server_deregister_nspace = ext2x_server_deregister_nspace,
    .server_register_client = ext2x_server_register_client,
    .server_deregister_client = ext2x_server_deregister_client,
    .server_setup_fork = ext2x_server_setup_fork,
    .server_dmodex_request = ext2x_server_dmodex,
    .server_notify_event = ext2x_server_notify_event,
    /* tool APIs */
    .tool_init = ext2x_tool_init,
    .tool_finalize = ext2x_tool_fini,
    /* utility APIs */
    .get_version = PMIx_Get_version,
    .register_evhandler = register_handler,
    .deregister_evhandler = deregister_handler,
    .notify_event = notify_event,
    .store_local = ext2x_store_local,
    .get_nspace = ext2x_get_nspace,
    .register_jobid = ext2x_register_jobid
};

static bool legacy_get(void)
{
    return mca_pmix_ext2x_component.legacy_get;
}

static void opcbfunc(pmix_status_t status, void *cbdata)
{
    ext2x_opcaddy_t *op = (ext2x_opcaddy_t*)cbdata;

    OPAL_ACQUIRE_OBJECT(op);

    if (NULL != op->opcbfunc) {
        op->opcbfunc(ext2x_colwert_rc(status), op->cbdata);
    }
    OBJ_RELEASE(op);
}


static const char *ext2x_get_nspace(opal_jobid_t jobid)
{
    opal_ext2x_jobid_trkr_t *jptr;

    OPAL_PMIX_ACQUIRE_THREAD(&opal_pmix_base.lock);

    OPAL_LIST_FOREACH(jptr, &mca_pmix_ext2x_component.jobids, opal_ext2x_jobid_trkr_t) {
        if (jptr->jobid == jobid) {
            OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);
            return jptr->nspace;
        }
    }
    OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);
    return NULL;
}

static void ext2x_register_jobid(opal_jobid_t jobid, const char *nspace)
{
    opal_ext2x_jobid_trkr_t *jptr;

    OPAL_PMIX_ACQUIRE_THREAD(&opal_pmix_base.lock);

    /* if we don't already have it, add this to our jobid tracker */
    OPAL_LIST_FOREACH(jptr, &mca_pmix_ext2x_component.jobids, opal_ext2x_jobid_trkr_t) {
        if (jptr->jobid == jobid) {
            OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);
            return;
        }
    }
    jptr = OBJ_NEW(opal_ext2x_jobid_trkr_t);
    (void)strncpy(jptr->nspace, nspace, PMIX_MAX_NSLEN);
    jptr->jobid = jobid;
    opal_list_append(&mca_pmix_ext2x_component.jobids, &jptr->super);
    OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);
}

static void event_hdlr_complete(pmix_status_t status, void *cbdata)
{
    ext2x_opcaddy_t *op = (ext2x_opcaddy_t*)cbdata;

    OBJ_RELEASE(op);
}

static void return_local_event_hdlr(int status, opal_list_t *results,
                                    opal_pmix_op_cbfunc_t cbfunc, void *thiscbdata,
                                    void *notification_cbdata)
{
    ext2x_threadshift_t *cd = (ext2x_threadshift_t*)notification_cbdata;
    ext2x_opcaddy_t *op;
    opal_value_t *kv;
    pmix_status_t pstatus;
    size_t n;

    OPAL_ACQUIRE_OBJECT(cd);
    if (NULL != cd->pmixcbfunc) {
        op = OBJ_NEW(ext2x_opcaddy_t);

        if (NULL != results && 0 < (op->ninfo = opal_list_get_size(results))) {
            /* colwert the list of results to an array of info */
            PMIX_INFO_CREATE(op->info, op->ninfo);
            n=0;
            OPAL_LIST_FOREACH(kv, cd->info, opal_value_t) {
                (void)strncpy(op->info[n].key, kv->key, PMIX_MAX_KEYLEN);
                ext2x_value_load(&op->info[n].value, kv);
                ++n;
            }
        }
        /* colwert the status */
        pstatus = ext2x_colwert_opalrc(status);
        /* call the library's callback function */
        cd->pmixcbfunc(pstatus, op->info, op->ninfo, event_hdlr_complete, op, cd->cbdata);
    }

    /* release the threadshift object */
    if (NULL != cd->info) {
        OPAL_LIST_RELEASE(cd->info);
    }
    OBJ_RELEASE(cd);

    /* release the caller */
    if (NULL != cbfunc) {
        cbfunc(OPAL_SUCCESS, thiscbdata);
    }
}

/* this function will be called by the PMIx client library
 * whenever it receives notification of an event. The
 * notification can come from an ORTE daemon (when launched
 * by mpirun), directly from a RM (when direct launched), or
 * from another process (via the local daemon).
 * The call will occur in the PMIx event base */
void ext2x_event_hdlr(size_t evhdlr_registration_id,
                       pmix_status_t status, const pmix_proc_t *source,
                       pmix_info_t info[], size_t ninfo,
                       pmix_info_t results[], size_t nresults,
                       pmix_event_notification_cbfunc_fn_t cbfunc,
                       void *cbdata)
{
    ext2x_threadshift_t *cd;
    int rc;
    opal_value_t *iptr;
    size_t n;
    opal_ext2x_event_t *event;

    opal_output_verbose(2, opal_pmix_base_framework.framework_output,
                        "%s RECEIVED NOTIFICATION OF STATUS %d",
                        OPAL_NAME_PRINT(OPAL_PROC_MY_NAME), status);

    OPAL_PMIX_ACQUIRE_THREAD(&opal_pmix_base.lock);

    cd = OBJ_NEW(ext2x_threadshift_t);
    cd->id = evhdlr_registration_id;
    cd->pmixcbfunc = cbfunc;
    cd->cbdata = cbdata;

    /* colwert the incoming status */
    cd->status = ext2x_colwert_rc(status);
    opal_output_verbose(2, opal_pmix_base_framework.framework_output,
                        "%s COLWERTED STATUS %d TO STATUS %d",
                        OPAL_NAME_PRINT(OPAL_PROC_MY_NAME), status, cd->status);

    /* colwert the nspace/rank to an opal_process_name_t */
    if (NULL == source) {
        cd->pname.jobid = OPAL_NAME_ILWALID->jobid;
        cd->pname.vpid = OPAL_NAME_ILWALID->vpid;
    } else {
        if (OPAL_SUCCESS != (rc = opal_colwert_string_to_jobid(&cd->pname.jobid, source->nspace))) {
            OPAL_ERROR_LOG(rc);
            cd->pname.jobid = OPAL_NAME_ILWALID->jobid;
        }
        cd->pname.vpid = ext2x_colwert_rank(source->rank);
    }

    /* colwert the array of info */
    if (NULL != info) {
        cd->info = OBJ_NEW(opal_list_t);
        for (n=0; n < ninfo; n++) {
            iptr = OBJ_NEW(opal_value_t);
            iptr->key = strdup(info[n].key);
            if (OPAL_SUCCESS != (rc = ext2x_value_unload(iptr, &info[n].value))) {
                OPAL_ERROR_LOG(rc);
                OBJ_RELEASE(iptr);
                continue;
            }
            opal_list_append(cd->info, &iptr->super);
        }
    }

    /* colwert the array of prior results */
    if (NULL != results) {
        for (n=0; n < nresults; n++) {
            iptr = OBJ_NEW(opal_value_t);
            iptr->key = strdup(results[n].key);
            if (OPAL_SUCCESS != (rc = ext2x_value_unload(iptr, &results[n].value))) {
                OPAL_ERROR_LOG(rc);
                OBJ_RELEASE(iptr);
                continue;
            }
            opal_list_append(&cd->results, &iptr->super);
        }
    }

    /* cycle thru the registrations */
    OPAL_LIST_FOREACH(event, &mca_pmix_ext2x_component.events, opal_ext2x_event_t) {
        if (evhdlr_registration_id == event->index) {
            /* found it - ilwoke the handler, pointing its
             * callback function to our callback function */
            opal_output_verbose(2, opal_pmix_base_framework.framework_output,
                                "%s _EVENT_HDLR CALLING EVHDLR",
                                OPAL_NAME_PRINT(OPAL_PROC_MY_NAME));
            if (NULL != event->handler) {
                OBJ_RETAIN(event);
                OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);
                event->handler(cd->status, &cd->pname,
                               cd->info, &cd->results,
                               return_local_event_hdlr, cd);
                OBJ_RELEASE(event);
                return;
            }
        }
    }

    OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);

    /* if we didn't find a match, we still have to call their final callback */
    if (NULL != cbfunc) {
        cbfunc(PMIX_SUCCESS, NULL, 0, NULL, NULL, cbdata);
    }
    OPAL_LIST_RELEASE(cd->info);
    OBJ_RELEASE(cd);
    return;
}

opal_vpid_t ext2x_colwert_rank(pmix_rank_t rank)
{
    switch(rank) {
    case PMIX_RANK_UNDEF:
        return OPAL_VPID_ILWALID;
    case PMIX_RANK_WILDCARD:
        return OPAL_VPID_WILDCARD;
    default:
        return (opal_vpid_t)rank;
    }
}

pmix_rank_t ext2x_colwert_opalrank(opal_vpid_t vpid)
{
    switch(vpid) {
    case OPAL_VPID_WILDCARD:
        return PMIX_RANK_WILDCARD;
    case OPAL_VPID_ILWALID:
        return PMIX_RANK_UNDEF;
    default:
        return (pmix_rank_t)vpid;
    }
}

pmix_status_t ext2x_colwert_opalrc(int rc)
{
    switch (rc) {
    case OPAL_ERR_DEBUGGER_RELEASE:
        return PMIX_ERR_DEBUGGER_RELEASE;

    case OPAL_ERR_HANDLERS_COMPLETE:
        return PMIX_EVENT_ACTION_COMPLETE;

    case OPAL_ERR_PROC_ABORTED:
        return PMIX_ERR_PROC_ABORTED;

    case OPAL_ERR_PROC_REQUESTED_ABORT:
        return PMIX_ERR_PROC_REQUESTED_ABORT;

    case OPAL_ERR_PROC_ABORTING:
        return PMIX_ERR_PROC_ABORTING;

    case OPAL_ERR_NODE_DOWN:
        return PMIX_ERR_NODE_DOWN;

    case OPAL_ERR_NODE_OFFLINE:
        return PMIX_ERR_NODE_OFFLINE;

    case OPAL_ERR_JOB_TERMINATED:
        return PMIX_ERR_JOB_TERMINATED;

    case OPAL_ERR_PROC_RESTART:
        return PMIX_ERR_PROC_RESTART;

    case OPAL_ERR_PROC_CHECKPOINT:
        return PMIX_ERR_PROC_CHECKPOINT;

    case OPAL_ERR_PROC_MIGRATE:
        return PMIX_ERR_PROC_MIGRATE;

    case OPAL_ERR_EVENT_REGISTRATION:
        return PMIX_ERR_EVENT_REGISTRATION;

    case OPAL_ERR_NOT_IMPLEMENTED:
    case OPAL_ERR_NOT_SUPPORTED:
        return PMIX_ERR_NOT_SUPPORTED;

    case OPAL_ERR_NOT_FOUND:
        return PMIX_ERR_NOT_FOUND;

    case OPAL_ERR_PERM:
    case OPAL_ERR_UNREACH:
    case OPAL_ERR_SERVER_NOT_AVAIL:
        return PMIX_ERR_UNREACH;

    case OPAL_ERR_BAD_PARAM:
        return PMIX_ERR_BAD_PARAM;

    case OPAL_ERR_OUT_OF_RESOURCE:
        return PMIX_ERR_OUT_OF_RESOURCE;

    case OPAL_ERR_DATA_VALUE_NOT_FOUND:
        return PMIX_ERR_DATA_VALUE_NOT_FOUND;

    case OPAL_ERR_TIMEOUT:
        return PMIX_ERR_TIMEOUT;

    case OPAL_ERR_WOULD_BLOCK:
        return PMIX_ERR_WOULD_BLOCK;

    case OPAL_EXISTS:
        return PMIX_EXISTS;

    case OPAL_ERR_PARTIAL_SUCCESS:
        return PMIX_QUERY_PARTIAL_SUCCESS;

    case OPAL_ERR_MODEL_DECLARED:
        return PMIX_MODEL_DECLARED;

    case OPAL_ERROR:
        return PMIX_ERROR;
    case OPAL_SUCCESS:
    case OPAL_OPERATION_SUCCEEDED:
        return PMIX_SUCCESS;
    default:
        return rc;
    }
}

int ext2x_colwert_rc(pmix_status_t rc)
{
    switch (rc) {
    case PMIX_ERR_DEBUGGER_RELEASE:
        return OPAL_ERR_DEBUGGER_RELEASE;

    case PMIX_EVENT_ACTION_COMPLETE:
        return OPAL_ERR_HANDLERS_COMPLETE;

    case PMIX_ERR_PROC_ABORTED:
        return OPAL_ERR_PROC_ABORTED;

    case PMIX_ERR_PROC_REQUESTED_ABORT:
        return OPAL_ERR_PROC_REQUESTED_ABORT;

    case PMIX_ERR_PROC_ABORTING:
        return OPAL_ERR_PROC_ABORTING;

    case PMIX_ERR_NODE_DOWN:
        return OPAL_ERR_NODE_DOWN;

    case PMIX_ERR_NODE_OFFLINE:
        return OPAL_ERR_NODE_OFFLINE;

    case PMIX_ERR_JOB_TERMINATED:
        return OPAL_ERR_JOB_TERMINATED;

    case PMIX_ERR_PROC_RESTART:
        return OPAL_ERR_PROC_RESTART;

    case PMIX_ERR_PROC_CHECKPOINT:
        return OPAL_ERR_PROC_CHECKPOINT;

    case PMIX_ERR_PROC_MIGRATE:
        return OPAL_ERR_PROC_MIGRATE;

    case PMIX_ERR_EVENT_REGISTRATION:
        return OPAL_ERR_EVENT_REGISTRATION;

    case PMIX_ERR_NOT_SUPPORTED:
        return OPAL_ERR_NOT_SUPPORTED;

    case PMIX_ERR_NOT_FOUND:
        return OPAL_ERR_NOT_FOUND;

    case PMIX_ERR_OUT_OF_RESOURCE:
        return OPAL_ERR_OUT_OF_RESOURCE;

    case PMIX_ERR_INIT:
        return OPAL_ERROR;

    case PMIX_ERR_BAD_PARAM:
        return OPAL_ERR_BAD_PARAM;

    case PMIX_ERR_UNREACH:
    case PMIX_ERR_NO_PERMISSIONS:
        return OPAL_ERR_UNREACH;

    case PMIX_ERR_TIMEOUT:
        return OPAL_ERR_TIMEOUT;

    case PMIX_ERR_WOULD_BLOCK:
        return OPAL_ERR_WOULD_BLOCK;

    case PMIX_ERR_LOST_CONNECTION_TO_SERVER:
    case PMIX_ERR_LOST_PEER_CONNECTION:
    case PMIX_ERR_LOST_CONNECTION_TO_CLIENT:
        return OPAL_ERR_COMM_FAILURE;

    case PMIX_EXISTS:
        return OPAL_EXISTS;

    case PMIX_QUERY_PARTIAL_SUCCESS:
        return OPAL_ERR_PARTIAL_SUCCESS;

    case PMIX_MONITOR_HEARTBEAT_ALERT:
        return OPAL_ERR_HEARTBEAT_ALERT;

    case PMIX_MONITOR_FILE_ALERT:
        return OPAL_ERR_FILE_ALERT;

    case PMIX_MODEL_DECLARED:
        return OPAL_ERR_MODEL_DECLARED;


    case PMIX_ERROR:
        return OPAL_ERROR;
    case PMIX_SUCCESS:
        return OPAL_SUCCESS;
    default:
        return rc;
    }
}

opal_pmix_scope_t ext2x_colwert_scope(pmix_scope_t scope)
{
    switch(scope) {
        case PMIX_SCOPE_UNDEF:
            return OPAL_PMIX_SCOPE_UNDEF;
        case PMIX_LOCAL:
            return OPAL_PMIX_LOCAL;
        case PMIX_REMOTE:
            return OPAL_PMIX_REMOTE;
        case PMIX_GLOBAL:
            return OPAL_PMIX_GLOBAL;
        default:
            return OPAL_PMIX_SCOPE_UNDEF;
    }
}

pmix_scope_t ext2x_colwert_opalscope(opal_pmix_scope_t scope) {
    switch(scope) {
    case OPAL_PMIX_LOCAL:
        return PMIX_LOCAL;
    case OPAL_PMIX_REMOTE:
        return PMIX_REMOTE;
    case OPAL_PMIX_GLOBAL:
        return PMIX_GLOBAL;
    default:
        return PMIX_SCOPE_UNDEF;
    }
}

pmix_data_range_t ext2x_colwert_opalrange(opal_pmix_data_range_t range) {
    switch(range) {
    case OPAL_PMIX_RANGE_UNDEF:
        return PMIX_RANGE_UNDEF;
    case OPAL_PMIX_RANGE_LOCAL:
        return PMIX_RANGE_LOCAL;
    case OPAL_PMIX_RANGE_NAMESPACE:
        return PMIX_RANGE_NAMESPACE;
    case OPAL_PMIX_RANGE_SESSION:
        return PMIX_RANGE_SESSION;
    case OPAL_PMIX_RANGE_GLOBAL:
        return PMIX_RANGE_GLOBAL;
    case OPAL_PMIX_RANGE_LWSTOM:
        return PMIX_RANGE_LWSTOM;
    default:
        return PMIX_SCOPE_UNDEF;
    }
}

opal_pmix_data_range_t ext2x_colwert_range(pmix_data_range_t range) {
    switch(range) {
    case PMIX_RANGE_UNDEF:
        return OPAL_PMIX_RANGE_UNDEF;
    case PMIX_RANGE_LOCAL:
        return OPAL_PMIX_RANGE_LOCAL;
    case PMIX_RANGE_NAMESPACE:
        return OPAL_PMIX_RANGE_NAMESPACE;
    case PMIX_RANGE_SESSION:
        return OPAL_PMIX_RANGE_SESSION;
    case PMIX_RANGE_GLOBAL:
        return OPAL_PMIX_RANGE_GLOBAL;
    case PMIX_RANGE_LWSTOM:
        return OPAL_PMIX_RANGE_LWSTOM;
    default:
        return OPAL_PMIX_RANGE_UNDEF;
    }
}

opal_pmix_persistence_t ext2x_colwert_persist(pmix_persistence_t persist)
{
    switch(persist) {
        case PMIX_PERSIST_INDEF:
            return OPAL_PMIX_PERSIST_INDEF;
        case PMIX_PERSIST_FIRST_READ:
            return OPAL_PMIX_PERSIST_FIRST_READ;
        case PMIX_PERSIST_PROC:
            return OPAL_PMIX_PERSIST_PROC;
        case PMIX_PERSIST_APP:
            return OPAL_PMIX_PERSIST_APP;
        case PMIX_PERSIST_SESSION:
            return OPAL_PMIX_PERSIST_SESSION;
        default:
        return OPAL_PMIX_PERSIST_INDEF;
    }
}

pmix_persistence_t ext2x_colwert_opalpersist(opal_pmix_persistence_t persist)
{
    switch(persist) {
        case OPAL_PMIX_PERSIST_INDEF:
            return PMIX_PERSIST_INDEF;
        case OPAL_PMIX_PERSIST_FIRST_READ:
            return PMIX_PERSIST_FIRST_READ;
        case OPAL_PMIX_PERSIST_PROC:
            return PMIX_PERSIST_PROC;
        case OPAL_PMIX_PERSIST_APP:
            return PMIX_PERSIST_APP;
        case OPAL_PMIX_PERSIST_SESSION:
            return PMIX_PERSIST_SESSION;
        default:
            return PMIX_PERSIST_INDEF;
    }
}

char* ext2x_colwert_jobid(opal_jobid_t jobid)
{
    opal_ext2x_jobid_trkr_t *jptr;

    /* look thru our list of jobids and find the
     * corresponding nspace */
    OPAL_LIST_FOREACH(jptr, &mca_pmix_ext2x_component.jobids, opal_ext2x_jobid_trkr_t) {
        if (jptr->jobid == jobid) {
            return jptr->nspace;
        }
    }
    return NULL;
}

/****   RHC: NEED TO ADD SUPPORT FOR NEW PMIX DATA TYPES, INCLUDING
 ****   COLWERSION OF PROC STATES    ****/

void ext2x_value_load(pmix_value_t *v,
                       opal_value_t *kv)
{
    opal_ext2x_jobid_trkr_t *job;
    bool found;
    opal_list_t *list;
    opal_value_t *val;
    pmix_info_t *info;
    size_t n;

    switch(kv->type) {
        case OPAL_UNDEF:
            v->type = PMIX_UNDEF;
            break;
        case OPAL_BOOL:
            v->type = PMIX_BOOL;
            memcpy(&(v->data.flag), &kv->data.flag, 1);
            break;
        case OPAL_BYTE:
            v->type = PMIX_BYTE;
            memcpy(&(v->data.byte), &kv->data.byte, 1);
            break;
        case OPAL_STRING:
            v->type = PMIX_STRING;
            if (NULL != kv->data.string) {
                v->data.string = strdup(kv->data.string);
            } else {
                v->data.string = NULL;
            }
            break;
        case OPAL_SIZE:
            v->type = PMIX_SIZE;
            memcpy(&(v->data.size), &kv->data.size, sizeof(size_t));
            break;
        case OPAL_PID:
            v->type = PMIX_PID;
            memcpy(&(v->data.pid), &kv->data.pid, sizeof(pid_t));
            break;
        case OPAL_INT:
            v->type = PMIX_INT;
            memcpy(&(v->data.integer), &kv->data.integer, sizeof(int));
            break;
        case OPAL_INT8:
            v->type = PMIX_INT8;
            memcpy(&(v->data.int8), &kv->data.int8, 1);
            break;
        case OPAL_INT16:
            v->type = PMIX_INT16;
            memcpy(&(v->data.int16), &kv->data.int16, 2);
            break;
        case OPAL_INT32:
            v->type = PMIX_INT32;
            memcpy(&(v->data.int32), &kv->data.int32, 4);
            break;
        case OPAL_INT64:
            v->type = PMIX_INT64;
            memcpy(&(v->data.int64), &kv->data.int64, 8);
            break;
        case OPAL_UINT:
            v->type = PMIX_UINT;
            memcpy(&(v->data.uint), &kv->data.uint, sizeof(int));
            break;
        case OPAL_UINT8:
            v->type = PMIX_UINT8;
            memcpy(&(v->data.uint8), &kv->data.uint8, 1);
            break;
        case OPAL_UINT16:
            v->type = PMIX_UINT16;
            memcpy(&(v->data.uint16), &kv->data.uint16, 2);
            break;
        case OPAL_UINT32:
            v->type = PMIX_UINT32;
            memcpy(&(v->data.uint32), &kv->data.uint32, 4);
            break;
        case OPAL_UINT64:
            v->type = PMIX_UINT64;
            memcpy(&(v->data.uint64), &kv->data.uint64, 8);
            break;
        case OPAL_FLOAT:
            v->type = PMIX_FLOAT;
            memcpy(&(v->data.fval), &kv->data.fval, sizeof(float));
            break;
        case OPAL_DOUBLE:
            v->type = PMIX_DOUBLE;
            memcpy(&(v->data.dval), &kv->data.dval, sizeof(double));
            break;
        case OPAL_TIMEVAL:
            v->type = PMIX_TIMEVAL;
            memcpy(&(v->data.tv), &kv->data.tv, sizeof(struct timeval));
            break;
        case OPAL_TIME:
            v->type = PMIX_TIME;
            memcpy(&(v->data.time), &kv->data.time, sizeof(time_t));
            break;
        case OPAL_STATUS:
            v->type = PMIX_STATUS;
            v->data.status = ext2x_colwert_opalrc(kv->data.status);
            break;
        case OPAL_VPID:
            v->type = PMIX_PROC_RANK;
            v->data.rank = ext2x_colwert_opalrank(kv->data.name.vpid);
            break;
        case OPAL_NAME:
            v->type = PMIX_PROC;
            /* have to stringify the jobid */
            PMIX_PROC_CREATE(v->data.proc, 1);
            /* see if this job is in our list of known nspaces */
            found = false;
            OPAL_LIST_FOREACH(job, &mca_pmix_ext2x_component.jobids, opal_ext2x_jobid_trkr_t) {
                if (job->jobid == kv->data.name.jobid) {
                    (void)strncpy(v->data.proc->nspace, job->nspace, PMIX_MAX_NSLEN);
                    found = true;
                    break;
                }
            }
            if (!found) {
                (void)opal_snprintf_jobid(v->data.proc->nspace, PMIX_MAX_NSLEN, kv->data.name.jobid);
            }
            v->data.proc->rank = ext2x_colwert_opalrank(kv->data.name.vpid);
            break;
        case OPAL_BYTE_OBJECT:
            v->type = PMIX_BYTE_OBJECT;
            if (NULL != kv->data.bo.bytes) {
                v->data.bo.bytes = (char*)malloc(kv->data.bo.size);
                memcpy(v->data.bo.bytes, kv->data.bo.bytes, kv->data.bo.size);
                v->data.bo.size = (size_t)kv->data.bo.size;
            } else {
                v->data.bo.bytes = NULL;
                v->data.bo.size = 0;
            }
            break;
        case OPAL_PERSIST:
            v->type = PMIX_PERSIST;
            v->data.persist = ext2x_colwert_opalpersist((opal_pmix_persistence_t)kv->data.uint8);
            break;
        case OPAL_SCOPE:
            v->type = PMIX_SCOPE;
            v->data.scope = ext2x_colwert_opalscope((opal_pmix_scope_t)kv->data.uint8);
            break;
        case OPAL_DATA_RANGE:
            v->type = PMIX_DATA_RANGE;
            v->data.range = ext2x_colwert_opalrange((opal_pmix_data_range_t)kv->data.uint8);
            break;
        case OPAL_PROC_STATE:
            v->type = PMIX_PROC_STATE;
            /* the OPAL layer doesn't have any concept of proc state,
             * so the ORTE layer is responsible for colwerting it */
            memcpy(&v->data.state, &kv->data.uint8, sizeof(uint8_t));
            break;
        case OPAL_PTR:
            /* if someone returned a pointer, it must be to a list of
             * opal_value_t's that we need to colwert to a pmix_data_array
             * of pmix_info_t structures */
            list = (opal_list_t*)kv->data.ptr;
            v->type = PMIX_DATA_ARRAY;
            v->data.darray = (pmix_data_array_t*)malloc(sizeof(pmix_data_array_t));
            v->data.darray->type = PMIX_INFO;
            v->data.darray->size = opal_list_get_size(list);
            if (0 < v->data.darray->size) {
                PMIX_INFO_CREATE(info, v->data.darray->size);
                v->data.darray->array = info;
                n=0;
                OPAL_LIST_FOREACH(val, list, opal_value_t) {
                    (void)strncpy(info[n].key, val->key, PMIX_MAX_KEYLEN);
                    ext2x_value_load(&info[n].value, val);
                    ++n;
                }
            } else {
                v->data.darray->array = NULL;
            }
            break;
        default:
            /* silence warnings */
            break;
    }
}

int ext2x_value_unload(opal_value_t *kv,
                       const pmix_value_t *v)
{
    int rc=OPAL_SUCCESS;
    bool found;
    opal_ext2x_jobid_trkr_t *job;
    opal_list_t *lt;
    opal_value_t *ival;
    size_t n;

    switch(v->type) {
    case PMIX_UNDEF:
        kv->type = OPAL_UNDEF;
        break;
    case PMIX_BOOL:
        kv->type = OPAL_BOOL;
        memcpy(&kv->data.flag, &(v->data.flag), 1);
        break;
    case PMIX_BYTE:
        kv->type = OPAL_BYTE;
        memcpy(&kv->data.byte, &(v->data.byte), 1);
        break;
    case PMIX_STRING:
        kv->type = OPAL_STRING;
        if (NULL != v->data.string) {
            kv->data.string = strdup(v->data.string);
        }
        break;
    case PMIX_SIZE:
        kv->type = OPAL_SIZE;
        memcpy(&kv->data.size, &(v->data.size), sizeof(size_t));
        break;
    case PMIX_PID:
        kv->type = OPAL_PID;
        memcpy(&kv->data.pid, &(v->data.pid), sizeof(pid_t));
        break;
    case PMIX_INT:
        kv->type = OPAL_INT;
        memcpy(&kv->data.integer, &(v->data.integer), sizeof(int));
        break;
    case PMIX_INT8:
        kv->type = OPAL_INT8;
        memcpy(&kv->data.int8, &(v->data.int8), 1);
        break;
    case PMIX_INT16:
        kv->type = OPAL_INT16;
        memcpy(&kv->data.int16, &(v->data.int16), 2);
        break;
    case PMIX_INT32:
        kv->type = OPAL_INT32;
        memcpy(&kv->data.int32, &(v->data.int32), 4);
        break;
    case PMIX_INT64:
        kv->type = OPAL_INT64;
        memcpy(&kv->data.int64, &(v->data.int64), 8);
        break;
    case PMIX_UINT:
        kv->type = OPAL_UINT;
        memcpy(&kv->data.uint, &(v->data.uint), sizeof(int));
        break;
    case PMIX_UINT8:
        kv->type = OPAL_UINT8;
        memcpy(&kv->data.uint8, &(v->data.uint8), 1);
        break;
    case PMIX_UINT16:
        kv->type = OPAL_UINT16;
        memcpy(&kv->data.uint16, &(v->data.uint16), 2);
        break;
    case PMIX_UINT32:
        kv->type = OPAL_UINT32;
        memcpy(&kv->data.uint32, &(v->data.uint32), 4);
        break;
    case PMIX_UINT64:
        kv->type = OPAL_UINT64;
        memcpy(&kv->data.uint64, &(v->data.uint64), 8);
        break;
    case PMIX_FLOAT:
        kv->type = OPAL_FLOAT;
        memcpy(&kv->data.fval, &(v->data.fval), sizeof(float));
        break;
    case PMIX_DOUBLE:
        kv->type = OPAL_DOUBLE;
        memcpy(&kv->data.dval, &(v->data.dval), sizeof(double));
        break;
    case PMIX_TIMEVAL:
        kv->type = OPAL_TIMEVAL;
        memcpy(&kv->data.tv, &(v->data.tv), sizeof(struct timeval));
        break;
    case PMIX_TIME:
        kv->type = OPAL_TIME;
        memcpy(&kv->data.time, &(v->data.time), sizeof(time_t));
        break;
    case PMIX_STATUS:
        kv->type = OPAL_STATUS;
        kv->data.status = ext2x_colwert_rc(v->data.status);
        break;
    case PMIX_PROC_RANK:
        kv->type = OPAL_VPID;
        kv->data.name.vpid = ext2x_colwert_rank(v->data.rank);
        break;
    case PMIX_PROC:
        kv->type = OPAL_NAME;
        /* see if this job is in our list of known nspaces */
        found = false;
        OPAL_LIST_FOREACH(job, &mca_pmix_ext2x_component.jobids, opal_ext2x_jobid_trkr_t) {
            if (0 == strncmp(job->nspace, v->data.proc->nspace, PMIX_MAX_NSLEN)) {
                kv->data.name.jobid = job->jobid;
                found = true;
                break;
            }
        }
        if (!found) {
            if (OPAL_SUCCESS != (rc = opal_colwert_string_to_jobid(&kv->data.name.jobid, v->data.proc->nspace))) {
                return ext2x_colwert_opalrc(rc);
            }
        }
        kv->data.name.vpid = ext2x_colwert_rank(v->data.proc->rank);
        break;
    case PMIX_BYTE_OBJECT:
        kv->type = OPAL_BYTE_OBJECT;
        if (NULL != v->data.bo.bytes && 0 < v->data.bo.size) {
            kv->data.bo.bytes = (uint8_t*)malloc(v->data.bo.size);
            memcpy(kv->data.bo.bytes, v->data.bo.bytes, v->data.bo.size);
            kv->data.bo.size = (int)v->data.bo.size;
        } else {
            kv->data.bo.bytes = NULL;
            kv->data.bo.size = 0;
        }
        break;
    case PMIX_PERSIST:
        kv->type = OPAL_PERSIST;
        kv->data.uint8 = ext2x_colwert_persist(v->data.persist);
        break;
    case PMIX_SCOPE:
        kv->type = OPAL_SCOPE;
        kv->data.uint8 = ext2x_colwert_scope(v->data.scope);
        break;
    case PMIX_DATA_RANGE:
        kv->type = OPAL_DATA_RANGE;
        kv->data.uint8 = ext2x_colwert_range(v->data.range);
        break;
    case PMIX_PROC_STATE:
        kv->type = OPAL_PROC_STATE;
        /* the OPAL layer doesn't have any concept of proc state,
         * so the ORTE layer is responsible for colwerting it */
        memcpy(&kv->data.uint8, &v->data.state, sizeof(uint8_t));
        break;
    case PMIX_POINTER:
        kv->type = OPAL_PTR;
        kv->data.ptr = v->data.ptr;
        break;
    case PMIX_DATA_ARRAY:
        if (NULL == v->data.darray || NULL == v->data.darray->array) {
            kv->data.ptr = NULL;
            break;
        }
        lt = OBJ_NEW(opal_list_t);
        kv->type = OPAL_PTR;
        kv->data.ptr = (void*)lt;
        for (n=0; n < v->data.darray->size; n++) {
            ival = OBJ_NEW(opal_value_t);
            opal_list_append(lt, &ival->super);
            /* handle the various types */
            if (PMIX_INFO == v->data.darray->type) {
                pmix_info_t *iptr = (pmix_info_t*)v->data.darray->array;
                ival->key = strdup(iptr[n].key);
                rc = ext2x_value_unload(ival, &iptr[n].value);
                if (OPAL_SUCCESS != rc) {
                    OPAL_LIST_RELEASE(lt);
                    kv->type = OPAL_UNDEF;
                    kv->data.ptr = NULL;
                    break;
                }
            }
        }
        break;
    default:
        /* silence warnings */
        rc = OPAL_ERROR;
        break;
    }
    return rc;
}

static void errreg_cbfunc (pmix_status_t status,
                           size_t errhandler_ref,
                           void *cbdata)
{
    ext2x_opcaddy_t *op = (ext2x_opcaddy_t*)cbdata;

    OPAL_ACQUIRE_OBJECT(op);
    op->event->index = errhandler_ref;
    opal_output_verbose(5, opal_pmix_base_framework.framework_output,
                        "PMIX2x errreg_cbfunc - error handler registered status=%d, reference=%lu",
                        status, (unsigned long)errhandler_ref);
    if (NULL != op->evregcbfunc) {
        op->evregcbfunc(ext2x_colwert_rc(status), errhandler_ref, op->cbdata);
    }
    OBJ_RELEASE(op);
}

static void register_handler(opal_list_t *event_codes,
                             opal_list_t *info,
                             opal_pmix_notification_fn_t evhandler,
                             opal_pmix_evhandler_reg_cbfunc_t cbfunc,
                             void *cbdata)
{
    ext2x_opcaddy_t *op = (ext2x_opcaddy_t*)cbdata;
    size_t n;
    opal_value_t *kv;

    OPAL_PMIX_ACQUIRE_THREAD(&opal_pmix_base.lock);
    if (0 >= opal_pmix_base.initialized) {
        OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);
        if (NULL != cbfunc) {
            cbfunc(OPAL_ERR_NOT_INITIALIZED, 0, cbdata);
        }
        return;
    }

    op = OBJ_NEW(ext2x_opcaddy_t);
    op->evregcbfunc = cbfunc;
    op->cbdata = cbdata;

    /* colwert the event codes */
    if (NULL != event_codes) {
        op->ncodes = opal_list_get_size(event_codes);
        op->pcodes = (pmix_status_t*)malloc(op->ncodes * sizeof(pmix_status_t));
        n=0;
        OPAL_LIST_FOREACH(kv, event_codes, opal_value_t) {
            op->pcodes[n] = ext2x_colwert_opalrc(kv->data.integer);
            ++n;
        }
    }

    /* colwert the list of info to an array of pmix_info_t */
    if (NULL != info && 0 < (op->ninfo = opal_list_get_size(info))) {
        PMIX_INFO_CREATE(op->info, op->ninfo);
        n=0;
        OPAL_LIST_FOREACH(kv, info, opal_value_t) {
            (void)strncpy(op->info[n].key, kv->key, PMIX_MAX_KEYLEN);
            ext2x_value_load(&op->info[n].value, kv);
            ++n;
        }
    }

    /* register the event */
    op->event = OBJ_NEW(opal_ext2x_event_t);
    op->event->handler = evhandler;
    opal_list_append(&mca_pmix_ext2x_component.events, &op->event->super);
    OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);

    PMIx_Register_event_handler(op->pcodes, op->ncodes,
                                op->info, op->ninfo,
                                ext2x_event_hdlr, errreg_cbfunc, op);
    return;
}

static void deregister_handler(size_t evhandler,
                               opal_pmix_op_cbfunc_t cbfunc,
                               void *cbdata)
{
    ext2x_opcaddy_t *op;
    opal_ext2x_event_t *event;

    OPAL_PMIX_ACQUIRE_THREAD(&opal_pmix_base.lock);
    if (0 >= opal_pmix_base.initialized) {
        OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);
        if (NULL != cbfunc) {
            cbfunc(OPAL_ERR_NOT_INITIALIZED, cbdata);
        }
        return;
    }

    /* look for this event */
    OPAL_LIST_FOREACH(event, &mca_pmix_ext2x_component.events, opal_ext2x_event_t) {
        if (evhandler == event->index) {
            opal_list_remove_item(&mca_pmix_ext2x_component.events, &event->super);
            OBJ_RELEASE(event);
            break;
        }
    }

    OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);

    op = OBJ_NEW(ext2x_opcaddy_t);
    op->opcbfunc = cbfunc;
    op->cbdata = cbdata;

    /* tell the library to deregister this handler */
    PMIx_Deregister_event_handler(evhandler, opcbfunc, op);
    return;
}

static void notify_complete(pmix_status_t status, void *cbdata)
{
    ext2x_opcaddy_t *op = (ext2x_opcaddy_t*)cbdata;
    if (NULL != op->opcbfunc) {
        op->opcbfunc(ext2x_colwert_rc(status), op->cbdata);
    }
    OBJ_RELEASE(op);
}

static int notify_event(int status,
                        const opal_process_name_t *source,
                        opal_pmix_data_range_t range,
                        opal_list_t *info,
                        opal_pmix_op_cbfunc_t cbfunc, void *cbdata)
{
    ext2x_opcaddy_t *op;
    opal_value_t *kv;
    pmix_proc_t p, *pptr;
    pmix_status_t pstatus;
    size_t n;
    pmix_data_range_t prange;
    char *nsptr;

    OPAL_PMIX_ACQUIRE_THREAD(&opal_pmix_base.lock);
    if (0 >= opal_pmix_base.initialized) {
        OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);
        return OPAL_ERR_NOT_INITIALIZED;
    }

    op = OBJ_NEW(ext2x_opcaddy_t);
    op->opcbfunc = cbfunc;
    op->cbdata = cbdata;

    /* colwert the status */
    pstatus = ext2x_colwert_opalrc(status);

    /* colwert the source */
    if (NULL == source) {
        pptr = NULL;
    } else {
        if (NULL == (nsptr = ext2x_colwert_jobid(source->jobid))) {
            OBJ_RELEASE(op);
            OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);
            return OPAL_ERR_NOT_FOUND;
        }
        (void)strncpy(p.nspace, nsptr, PMIX_MAX_NSLEN);
        p.rank = ext2x_colwert_opalrank(source->vpid);
        pptr = &p;
    }
    OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);

    /* colwert the range */
    prange = ext2x_colwert_opalrange(range);

    /* colwert the list of info */
    if (NULL != info && 0 < (op->ninfo = opal_list_get_size(info))) {
        PMIX_INFO_CREATE(op->info, op->ninfo);
        n=0;
        OPAL_LIST_FOREACH(kv, info, opal_value_t) {
            (void)strncpy(op->info[n].key, kv->key, PMIX_MAX_KEYLEN);
            /* little dicey here as we need to colwert a status, if
             * provided, and it will be an int coming down to us */
            if (0 == strcmp(kv->key, OPAL_PMIX_JOB_TERM_STATUS)) {
                op->info[n].value.type = PMIX_STATUS;
                op->info[n].value.data.status = ext2x_colwert_opalrc(kv->data.integer);
            } else {
                ext2x_value_load(&op->info[n].value, kv);
            }
            ++n;
        }
    }

    /* ask the library to notify our clients */
    pstatus = PMIx_Notify_event(pstatus, pptr, prange, op->info, op->ninfo, notify_complete, op);

    return ext2x_colwert_rc(pstatus);
}

static void relcbfunc(void *cbdata)
{
    opal_list_t *results = (opal_list_t*)cbdata;
    if (NULL != results) {
        OPAL_LIST_RELEASE(results);
    }
}

static void infocbfunc(pmix_status_t status,
                       pmix_info_t *info, size_t ninfo,
                       void *cbdata,
                       pmix_release_cbfunc_t release_fn,
                       void *release_cbdata)
{
    ext2x_opcaddy_t *cd = (ext2x_opcaddy_t*)cbdata;
    int rc = OPAL_SUCCESS;
    opal_list_t *results = NULL;
    opal_value_t *iptr;
    size_t n;

    OPAL_ACQUIRE_OBJECT(cd);

    /* colwert the array of pmix_info_t to the list of info */
    if (NULL != info) {
        results = OBJ_NEW(opal_list_t);
        for (n=0; n < ninfo; n++) {
            iptr = OBJ_NEW(opal_value_t);
            opal_list_append(results, &iptr->super);
            iptr->key = strdup(info[n].key);
            if (OPAL_SUCCESS != (rc = ext2x_value_unload(iptr, &info[n].value))) {
                OPAL_LIST_RELEASE(results);
                results = NULL;
                break;
            }
        }
    }

    if (NULL != release_fn) {
        release_fn(release_cbdata);
    }

    /* return the values to the original requestor */
    if (NULL != cd->qcbfunc) {
        cd->qcbfunc(rc, results, cd->cbdata, relcbfunc, results);
    }
    OBJ_RELEASE(cd);
}

static void ext2x_query(opal_list_t *queries,
                         opal_pmix_info_cbfunc_t cbfunc, void *cbdata)
{
    int rc;
    opal_value_t *ival;
    size_t n, nqueries, nq;
    ext2x_opcaddy_t *cd;
    pmix_status_t prc;
    opal_pmix_query_t *q;

    OPAL_PMIX_ACQUIRE_THREAD(&opal_pmix_base.lock);
    if (0 >= opal_pmix_base.initialized) {
        OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);
        if (NULL != cbfunc) {
            cbfunc(OPAL_ERR_NOT_INITIALIZED, NULL, cbdata, NULL, NULL);
        }
        return;
    }
    OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);

    /* create the caddy */
    cd = OBJ_NEW(ext2x_opcaddy_t);

    /* bozo check */
    if (NULL == queries || 0 == (nqueries = opal_list_get_size(queries))) {
        rc = OPAL_ERR_BAD_PARAM;
        goto CLEANUP;
    }

    /* setup the operation */
    cd->qcbfunc = cbfunc;
    cd->cbdata = cbdata;
    cd->nqueries = nqueries;

    /* colwert the list to an array of query objects */
    PMIX_QUERY_CREATE(cd->queries, cd->nqueries);
    n=0;
    OPAL_LIST_FOREACH(q, queries, opal_pmix_query_t) {
        cd->queries[n].keys = opal_argv_copy(q->keys);
        cd->queries[n].nqual = opal_list_get_size(&q->qualifiers);
        if (0 < cd->queries[n].nqual) {
            PMIX_INFO_CREATE(cd->queries[n].qualifiers, cd->queries[n].nqual);
            nq = 0;
            OPAL_LIST_FOREACH(ival, &q->qualifiers, opal_value_t) {
                (void)strncpy(cd->queries[n].qualifiers[nq].key, ival->key, PMIX_MAX_KEYLEN);
                ext2x_value_load(&cd->queries[n].qualifiers[nq].value, ival);
                ++nq;
            }
        }
        ++n;
    }

    /* pass it down */
    if (PMIX_SUCCESS != (prc = PMIx_Query_info_nb(cd->queries, cd->nqueries,
                                                  infocbfunc, cd))) {
        /* do not hang! */
        rc = ext2x_colwert_rc(prc);
        goto CLEANUP;
    }

    return;

  CLEANUP:
    if (NULL != cbfunc) {
        cbfunc(rc, NULL, cbdata, NULL, NULL);
    }
    OBJ_RELEASE(cd);
    return;
}

static void ext2x_log(opal_list_t *info,
                       opal_pmix_op_cbfunc_t cbfunc, void *cbdata)
{
    int rc;
    opal_value_t *ival;
    size_t n, ninfo;
    ext2x_opcaddy_t *cd;
    pmix_status_t prc;

    OPAL_PMIX_ACQUIRE_THREAD(&opal_pmix_base.lock);
    if (0 >= opal_pmix_base.initialized) {
        OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);
        if (NULL != cbfunc) {
            cbfunc(OPAL_ERR_NOT_INITIALIZED, cbdata);
        }
        return;
    }
    OPAL_PMIX_RELEASE_THREAD(&opal_pmix_base.lock);

    /* create the caddy */
    cd = OBJ_NEW(ext2x_opcaddy_t);

    /* bozo check */
    if (NULL == info || 0 == (ninfo = opal_list_get_size(info))) {
        rc = OPAL_ERR_BAD_PARAM;
        goto CLEANUP;
    }

    /* setup the operation */
    cd->opcbfunc = cbfunc;
    cd->cbdata = cbdata;
    cd->ninfo = ninfo;

    /* colwert the list to an array of info objects */
    PMIX_INFO_CREATE(cd->info, cd->ninfo);
    n=0;
    OPAL_LIST_FOREACH(ival, info, opal_value_t) {
        (void)strncpy(cd->info[n].key, ival->key, PMIX_MAX_KEYLEN);
        ext2x_value_load(&cd->info[n].value, ival);
        ++n;
    }

    /* pass it down */
    if (PMIX_SUCCESS != (prc = PMIx_Log_nb(cd->info, cd->ninfo, NULL, 0,
                                           opcbfunc, cd))) {
        /* do not hang! */
        rc = ext2x_colwert_rc(prc);
        goto CLEANUP;
    }

    return;

  CLEANUP:
    if (NULL != cbfunc) {
        cbfunc(rc, cbdata);
    }
    OBJ_RELEASE(cd);
}

opal_pmix_alloc_directive_t ext2x_colwert_allocdir(pmix_alloc_directive_t dir)
{
    switch (dir) {
        case PMIX_ALLOC_NEW:
            return OPAL_PMIX_ALLOC_NEW;
        case PMIX_ALLOC_EXTEND:
            return OPAL_PMIX_ALLOC_EXTEND;
        case PMIX_ALLOC_RELEASE:
            return OPAL_PMIX_ALLOC_RELEASE;
        case PMIX_ALLOC_REAQUIRE:
            return OPAL_PMIX_ALLOC_REAQLWIRE;
        default:
            return OPAL_PMIX_ALLOC_UNDEF;
    }
}

/****  INSTANTIATE INTERNAL CLASSES  ****/
OBJ_CLASS_INSTANCE(opal_ext2x_jobid_trkr_t,
                   opal_list_item_t,
                   NULL, NULL);

static void evcon(opal_ext2x_event_t *p)
{
    OPAL_PMIX_CONSTRUCT_LOCK(&p->lock);
    p->handler = NULL;
    p->cbdata = NULL;
}
static void evdes(opal_ext2x_event_t *p)
{
    OPAL_PMIX_DESTRUCT_LOCK(&p->lock);
}
OBJ_CLASS_INSTANCE(opal_ext2x_event_t,
                   opal_list_item_t,
                   evcon, evdes);

static void opcon(ext2x_opcaddy_t *p)
{
    memset(&p->p, 0, sizeof(pmix_proc_t));
    p->nspace = NULL;
    p->procs = NULL;
    p->nprocs = 0;
    p->pdata = NULL;
    p->npdata = 0;
    p->error_procs = NULL;
    p->nerror_procs = 0;
    p->info = NULL;
    p->ninfo = 0;
    p->apps = NULL;
    p->sz = 0;
    OPAL_PMIX_CONSTRUCT_LOCK(&p->lock);
    p->codes = NULL;
    p->pcodes = NULL;
    p->ncodes = 0;
    p->queries = NULL;
    p->nqueries = 0;
    p->event = NULL;
    p->opcbfunc = NULL;
    p->mdxcbfunc = NULL;
    p->valcbfunc = NULL;
    p->lkcbfunc = NULL;
    p->spcbfunc = NULL;
    p->evregcbfunc = NULL;
    p->qcbfunc = NULL;
    p->cbdata = NULL;
}
static void opdes(ext2x_opcaddy_t *p)
{
    OPAL_PMIX_DESTRUCT_LOCK(&p->lock);
    if (NULL != p->nspace) {
        free(p->nspace);
    }
    if (NULL != p->procs) {
        PMIX_PROC_FREE(p->procs, p->nprocs);
    }
    if (NULL != p->pdata) {
        PMIX_PDATA_FREE(p->pdata, p->npdata);
    }
    if (NULL != p->error_procs) {
        PMIX_PROC_FREE(p->error_procs, p->nerror_procs);
    }
    if (NULL != p->info) {
        PMIX_INFO_FREE(p->info, p->ninfo);
    }
    if (NULL != p->apps) {
        PMIX_APP_FREE(p->apps, p->sz);
    }
    if (NULL != p->pcodes) {
        free(p->pcodes);
    }
    if (NULL != p->queries) {
        PMIX_QUERY_FREE(p->queries, p->nqueries);
    }
}
OBJ_CLASS_INSTANCE(ext2x_opcaddy_t,
                   opal_object_t,
                   opcon, opdes);

static void ocadcon(ext2x_opalcaddy_t *p)
{
    OBJ_CONSTRUCT(&p->procs, opal_list_t);
    OBJ_CONSTRUCT(&p->info, opal_list_t);
    OBJ_CONSTRUCT(&p->apps, opal_list_t);
    p->opcbfunc = NULL;
    p->dmdxfunc = NULL;
    p->mdxcbfunc = NULL;
    p->lkupcbfunc = NULL;
    p->spwncbfunc = NULL;
    p->cbdata = NULL;
    p->odmdxfunc = NULL;
    p->infocbfunc = NULL;
    p->toolcbfunc = NULL;
    p->ocbdata = NULL;
}
static void ocaddes(ext2x_opalcaddy_t *p)
{
    OPAL_LIST_DESTRUCT(&p->procs);
    OPAL_LIST_DESTRUCT(&p->info);
    OPAL_LIST_DESTRUCT(&p->apps);
}
OBJ_CLASS_INSTANCE(ext2x_opalcaddy_t,
                   opal_object_t,
                   ocadcon, ocaddes);

static void tscon(ext2x_threadshift_t *p)
{
    OPAL_PMIX_CONSTRUCT_LOCK(&p->lock);
    p->msg = NULL;
    p->strings = NULL;
    p->source = NULL;
    p->event_codes = NULL;
    p->info = NULL;
    OBJ_CONSTRUCT(&p->results, opal_list_t);
    p->evhandler = NULL;
    p->nondefault = false;
    p->cbfunc = NULL;
    p->opcbfunc = NULL;
    p->cbdata = NULL;
}
static void tsdes(ext2x_threadshift_t *p)
{
    OPAL_PMIX_DESTRUCT_LOCK(&p->lock);
    if (NULL != p->strings) {
        free(p->strings);
    }
    OPAL_LIST_DESTRUCT(&p->results);
}
OBJ_CLASS_INSTANCE(ext2x_threadshift_t,
                   opal_object_t,
                   tscon, tsdes);

static void dmcon(opal_ext2x_dmx_trkr_t *p)
{
    p->nspace = NULL;
    p->cbfunc = NULL;
    p->cbdata = NULL;
}
static void dmdes(opal_ext2x_dmx_trkr_t *p)
{
    if (NULL != p->nspace) {
        free(p->nspace);
    }
}
OBJ_CLASS_INSTANCE(opal_ext2x_dmx_trkr_t,
                   opal_list_item_t,
                   dmcon, dmdes);
