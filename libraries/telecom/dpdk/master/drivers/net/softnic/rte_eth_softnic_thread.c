/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2018 Intel Corporation
 */

#include <stdlib.h>

#include <rte_common.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_service_component.h>
#include <rte_ring.h>

#include <rte_table_acl.h>
#include <rte_table_array.h>
#include <rte_table_hash.h>
#include <rte_table_lpm.h>
#include <rte_table_lpm_ipv6.h>
#include "rte_eth_softnic_internals.h"

/**
 * Main thread: data plane thread init
 */
void
softnic_thread_free(struct pmd_internals *softnic)
{
	uint32_t i;

	RTE_LCORE_FOREACH_WORKER(i) {
		struct softnic_thread *t = &softnic->thread[i];

		/* MSGQs */
		if (t->msgq_req)
			rte_ring_free(t->msgq_req);

		if (t->msgq_rsp)
			rte_ring_free(t->msgq_rsp);
	}
}

int
softnic_thread_init(struct pmd_internals *softnic)
{
	uint32_t i;

	for (i = 0; i < RTE_MAX_LCORE; i++) {
		char ring_name[NAME_MAX];
		struct rte_ring *msgq_req, *msgq_rsp;
		struct softnic_thread *t = &softnic->thread[i];
		struct softnic_thread_data *t_data = &softnic->thread_data[i];
		uint32_t cpu_id = rte_lcore_to_socket_id(i);

		/* MSGQs */
		snprintf(ring_name, sizeof(ring_name), "%s-TH%u-REQ",
			softnic->params.name,
			i);

		msgq_req = rte_ring_create(ring_name,
			THREAD_MSGQ_SIZE,
			cpu_id,
			RING_F_SP_ENQ | RING_F_SC_DEQ);

		if (msgq_req == NULL) {
			softnic_thread_free(softnic);
			return -1;
		}

		snprintf(ring_name, sizeof(ring_name), "%s-TH%u-RSP",
			softnic->params.name,
			i);

		msgq_rsp = rte_ring_create(ring_name,
			THREAD_MSGQ_SIZE,
			cpu_id,
			RING_F_SP_ENQ | RING_F_SC_DEQ);

		if (msgq_rsp == NULL) {
			softnic_thread_free(softnic);
			return -1;
		}

		/* Main thread records */
		t->msgq_req = msgq_req;
		t->msgq_rsp = msgq_rsp;
		t->service_id = UINT32_MAX;

		/* Data plane thread records */
		t_data->n_pipelines = 0;
		t_data->msgq_req = msgq_req;
		t_data->msgq_rsp = msgq_rsp;
		t_data->timer_period =
			(rte_get_tsc_hz() * THREAD_TIMER_PERIOD_MS) / 1000;
		t_data->time_next = rte_get_tsc_cycles() + t_data->timer_period;
		t_data->time_next_min = t_data->time_next;
	}

	return 0;
}

static inline int
thread_is_valid(struct pmd_internals *softnic, uint32_t thread_id)
{
	if (thread_id == rte_get_main_lcore())
		return 0; /* FALSE */

	if (softnic->params.sc && rte_lcore_has_role(thread_id, ROLE_SERVICE))
		return 1; /* TRUE */
	if (!softnic->params.sc && rte_lcore_has_role(thread_id, ROLE_RTE))
		return 1; /* TRUE */

	return 0; /* FALSE */
}

static inline int
thread_is_running(uint32_t thread_id)
{
	enum rte_lcore_state_t thread_state;

	thread_state = rte_eal_get_lcore_state(thread_id);
	return (thread_state == RUNNING)? 1 : 0;
}

static int32_t
rte_pmd_softnic_run_internal(void *arg);

static inline int
thread_sc_service_up(struct pmd_internals *softnic, uint32_t thread_id)
{
	struct rte_service_spec service_params;
	struct softnic_thread *t = &softnic->thread[thread_id];
	struct rte_eth_dev *dev;
	int status;
	uint16_t port_id;

	/* service params */
	status = rte_eth_dev_get_port_by_name(softnic->params.name, &port_id);
	if (status)
		return status;

	dev = &rte_eth_devices[port_id];
	snprintf(service_params.name, sizeof(service_params.name), "%s_%u",
		softnic->params.name,
		thread_id);
	service_params.callback = rte_pmd_softnic_run_internal;
	service_params.callback_userdata = dev;
	service_params.capabilities = 0;
	service_params.socket_id = (int)softnic->params.cpu_id;

	/* service register */
	status = rte_service_component_register(&service_params, &t->service_id);
	if (status)
		return status;

	status = rte_service_component_runstate_set(t->service_id, 1);
	if (status) {
		rte_service_component_unregister(t->service_id);
		t->service_id = UINT32_MAX;
		return status;
	}

	status = rte_service_runstate_set(t->service_id, 1);
	if (status) {
		rte_service_component_runstate_set(t->service_id, 0);
		rte_service_component_unregister(t->service_id);
		t->service_id = UINT32_MAX;
		return status;
	}

	/* service map to thread */
	status = rte_service_map_lcore_set(t->service_id, thread_id, 1);
	if (status) {
		rte_service_runstate_set(t->service_id, 0);
		rte_service_component_runstate_set(t->service_id, 0);
		rte_service_component_unregister(t->service_id);
		t->service_id = UINT32_MAX;
		return status;
	}

	return 0;
}

static inline void
thread_sc_service_down(struct pmd_internals *softnic, uint32_t thread_id)
{
	struct softnic_thread *t = &softnic->thread[thread_id];

	/* service unmap from thread */
	rte_service_map_lcore_set(t->service_id, thread_id, 0);

	/* service unregister */
	rte_service_runstate_set(t->service_id, 0);
	rte_service_component_runstate_set(t->service_id, 0);
	rte_service_component_unregister(t->service_id);

	t->service_id = UINT32_MAX;
}

/**
 * Pipeline is running when:
 *    (A) Pipeline is mapped to a data plane thread AND
 *    (B) Its data plane thread is in RUNNING state.
 */
static inline int
pipeline_is_running(struct pipeline *p)
{
	if (p->enabled == 0)
		return 0;

	return thread_is_running(p->thread_id);
}

/**
 * Main thread & data plane threads: message passing
 */
enum thread_req_type {
	THREAD_REQ_PIPELINE_ENABLE = 0,
	THREAD_REQ_PIPELINE_DISABLE,
	THREAD_REQ_MAX
};

struct thread_msg_req {
	enum thread_req_type type;

	union {
		struct {
			struct rte_pipeline *p;
			struct {
				struct rte_table_action *a;
			} table[RTE_PIPELINE_TABLE_MAX];
			struct rte_ring *msgq_req;
			struct rte_ring *msgq_rsp;
			uint32_t timer_period_ms;
			uint32_t n_tables;
		} pipeline_enable;

		struct {
			struct rte_pipeline *p;
		} pipeline_disable;
	};
};

struct thread_msg_rsp {
	int status;
};

/**
 * Main thread
 */
static struct thread_msg_req *
thread_msg_alloc(void)
{
	size_t size = RTE_MAX(sizeof(struct thread_msg_req),
		sizeof(struct thread_msg_rsp));

	return calloc(1, size);
}

static void
thread_msg_free(struct thread_msg_rsp *rsp)
{
	free(rsp);
}

static struct thread_msg_rsp *
thread_msg_send_recv(struct pmd_internals *softnic,
	uint32_t thread_id,
	struct thread_msg_req *req)
{
	struct softnic_thread *t = &softnic->thread[thread_id];
	struct rte_ring *msgq_req = t->msgq_req;
	struct rte_ring *msgq_rsp = t->msgq_rsp;
	struct thread_msg_rsp *rsp;
	int status;

	/* send */
	do {
		status = rte_ring_sp_enqueue(msgq_req, req);
	} while (status == -ENOBUFS);

	/* recv */
	do {
		status = rte_ring_sc_dequeue(msgq_rsp, (void **)&rsp);
	} while (status != 0);

	return rsp;
}

int
softnic_thread_pipeline_enable(struct pmd_internals *softnic,
	uint32_t thread_id,
	const char *pipeline_name)
{
	struct pipeline *p = softnic_pipeline_find(softnic, pipeline_name);
	struct thread_msg_req *req;
	struct thread_msg_rsp *rsp;
	uint32_t n_pipelines, i;
	int status;

	/* Check input params */
	if (!thread_is_valid(softnic, thread_id) ||
		(p == NULL) ||
		(p->n_ports_in == 0) ||
		(p->n_ports_out == 0) ||
		(p->n_tables == 0) ||
		p->enabled)
		return -1;

	n_pipelines = softnic_pipeline_thread_count(softnic, thread_id);
	if (n_pipelines >= THREAD_PIPELINES_MAX)
		return -1;

	if (softnic->params.sc && (n_pipelines == 0)) {
		status = thread_sc_service_up(softnic, thread_id);
		if (status)
			return status;
	}

	if (!thread_is_running(thread_id)) {
		struct softnic_thread_data *td = &softnic->thread_data[thread_id];
		struct pipeline_data *tdp = &td->pipeline_data[td->n_pipelines];

		/* Data plane thread */
		td->p[td->n_pipelines] = p->p;

		tdp->p = p->p;
		for (i = 0; i < p->n_tables; i++)
			tdp->table_data[i].a =
				p->table[i].a;
		tdp->n_tables = p->n_tables;

		tdp->msgq_req = p->msgq_req;
		tdp->msgq_rsp = p->msgq_rsp;
		tdp->timer_period = (rte_get_tsc_hz() * p->timer_period_ms) / 1000;
		tdp->time_next = rte_get_tsc_cycles() + tdp->timer_period;

		td->n_pipelines++;

		/* Pipeline */
		p->thread_id = thread_id;
		p->enabled = 1;

		return 0;
	}

	/* Allocate request */
	req = thread_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = THREAD_REQ_PIPELINE_ENABLE;
	req->pipeline_enable.p = p->p;
	for (i = 0; i < p->n_tables; i++)
		req->pipeline_enable.table[i].a =
			p->table[i].a;
	req->pipeline_enable.msgq_req = p->msgq_req;
	req->pipeline_enable.msgq_rsp = p->msgq_rsp;
	req->pipeline_enable.timer_period_ms = p->timer_period_ms;
	req->pipeline_enable.n_tables = p->n_tables;

	/* Send request and wait for response */
	rsp = thread_msg_send_recv(softnic, thread_id, req);

	/* Read response */
	status = rsp->status;

	/* Free response */
	thread_msg_free(rsp);

	/* Request completion */
	if (status)
		return status;

	p->thread_id = thread_id;
	p->enabled = 1;

	return 0;
}

int
softnic_thread_pipeline_disable(struct pmd_internals *softnic,
	uint32_t thread_id,
	const char *pipeline_name)
{
	struct pipeline *p = softnic_pipeline_find(softnic, pipeline_name);
	struct thread_msg_req *req;
	struct thread_msg_rsp *rsp;
	uint32_t n_pipelines;
	int status;

	/* Check input params */
	if (!thread_is_valid(softnic, thread_id) ||
		(p == NULL) ||
		(p->enabled && (p->thread_id != thread_id)))
		return -1;

	if (p->enabled == 0)
		return 0;

	if (!thread_is_running(thread_id)) {
		struct softnic_thread_data *td = &softnic->thread_data[thread_id];
		uint32_t i;

		for (i = 0; i < td->n_pipelines; i++) {
			struct pipeline_data *tdp = &td->pipeline_data[i];

			if (tdp->p != p->p)
				continue;

			/* Data plane thread */
			if (i < td->n_pipelines - 1) {
				struct rte_pipeline *pipeline_last =
					td->p[td->n_pipelines - 1];
				struct pipeline_data *tdp_last =
					&td->pipeline_data[td->n_pipelines - 1];

				td->p[i] = pipeline_last;
				memcpy(tdp, tdp_last, sizeof(*tdp));
			}

			td->n_pipelines--;

			/* Pipeline */
			p->enabled = 0;

			break;
		}

		if (softnic->params.sc && (td->n_pipelines == 0))
			thread_sc_service_down(softnic, thread_id);

		return 0;
	}

	/* Allocate request */
	req = thread_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = THREAD_REQ_PIPELINE_DISABLE;
	req->pipeline_disable.p = p->p;

	/* Send request and wait for response */
	rsp = thread_msg_send_recv(softnic, thread_id, req);

	/* Read response */
	status = rsp->status;

	/* Free response */
	thread_msg_free(rsp);

	/* Request completion */
	if (status)
		return status;

	p->enabled = 0;

	n_pipelines = softnic_pipeline_thread_count(softnic, thread_id);
	if (softnic->params.sc && (n_pipelines == 0))
		thread_sc_service_down(softnic, thread_id);

	return 0;
}

/**
 * Data plane threads: message handling
 */
static inline struct thread_msg_req *
thread_msg_recv(struct rte_ring *msgq_req)
{
	struct thread_msg_req *req;

	int status = rte_ring_sc_dequeue(msgq_req, (void **)&req);

	if (status != 0)
		return NULL;

	return req;
}

static inline void
thread_msg_send(struct rte_ring *msgq_rsp,
	struct thread_msg_rsp *rsp)
{
	int status;

	do {
		status = rte_ring_sp_enqueue(msgq_rsp, rsp);
	} while (status == -ENOBUFS);
}

static struct thread_msg_rsp *
thread_msg_handle_pipeline_enable(struct softnic_thread_data *t,
	struct thread_msg_req *req)
{
	struct thread_msg_rsp *rsp = (struct thread_msg_rsp *)req;
	struct pipeline_data *p = &t->pipeline_data[t->n_pipelines];
	uint32_t i;

	/* Request */
	t->p[t->n_pipelines] = req->pipeline_enable.p;

	p->p = req->pipeline_enable.p;
	for (i = 0; i < req->pipeline_enable.n_tables; i++)
		p->table_data[i].a =
			req->pipeline_enable.table[i].a;

	p->n_tables = req->pipeline_enable.n_tables;

	p->msgq_req = req->pipeline_enable.msgq_req;
	p->msgq_rsp = req->pipeline_enable.msgq_rsp;
	p->timer_period =
		(rte_get_tsc_hz() * req->pipeline_enable.timer_period_ms) / 1000;
	p->time_next = rte_get_tsc_cycles() + p->timer_period;

	t->n_pipelines++;

	/* Response */
	rsp->status = 0;
	return rsp;
}

static struct thread_msg_rsp *
thread_msg_handle_pipeline_disable(struct softnic_thread_data *t,
	struct thread_msg_req *req)
{
	struct thread_msg_rsp *rsp = (struct thread_msg_rsp *)req;
	uint32_t n_pipelines = t->n_pipelines;
	struct rte_pipeline *pipeline = req->pipeline_disable.p;
	uint32_t i;

	/* find pipeline */
	for (i = 0; i < n_pipelines; i++) {
		struct pipeline_data *p = &t->pipeline_data[i];

		if (p->p != pipeline)
			continue;

		if (i < n_pipelines - 1) {
			struct rte_pipeline *pipeline_last =
				t->p[n_pipelines - 1];
			struct pipeline_data *p_last =
				&t->pipeline_data[n_pipelines - 1];

			t->p[i] = pipeline_last;
			memcpy(p, p_last, sizeof(*p));
		}

		t->n_pipelines--;

		rsp->status = 0;
		return rsp;
	}

	/* should not get here */
	rsp->status = 0;
	return rsp;
}

static void
thread_msg_handle(struct softnic_thread_data *t)
{
	for ( ; ; ) {
		struct thread_msg_req *req;
		struct thread_msg_rsp *rsp;

		req = thread_msg_recv(t->msgq_req);
		if (req == NULL)
			break;

		switch (req->type) {
		case THREAD_REQ_PIPELINE_ENABLE:
			rsp = thread_msg_handle_pipeline_enable(t, req);
			break;

		case THREAD_REQ_PIPELINE_DISABLE:
			rsp = thread_msg_handle_pipeline_disable(t, req);
			break;

		default:
			rsp = (struct thread_msg_rsp *)req;
			rsp->status = -1;
		}

		thread_msg_send(t->msgq_rsp, rsp);
	}
}

/**
 * Main thread & data plane threads: message passing
 */
enum pipeline_req_type {
	/* Port IN */
	PIPELINE_REQ_PORT_IN_STATS_READ,
	PIPELINE_REQ_PORT_IN_ENABLE,
	PIPELINE_REQ_PORT_IN_DISABLE,

	/* Port OUT */
	PIPELINE_REQ_PORT_OUT_STATS_READ,

	/* Table */
	PIPELINE_REQ_TABLE_STATS_READ,
	PIPELINE_REQ_TABLE_RULE_ADD,
	PIPELINE_REQ_TABLE_RULE_ADD_DEFAULT,
	PIPELINE_REQ_TABLE_RULE_ADD_BULK,
	PIPELINE_REQ_TABLE_RULE_DELETE,
	PIPELINE_REQ_TABLE_RULE_DELETE_DEFAULT,
	PIPELINE_REQ_TABLE_RULE_STATS_READ,
	PIPELINE_REQ_TABLE_MTR_PROFILE_ADD,
	PIPELINE_REQ_TABLE_MTR_PROFILE_DELETE,
	PIPELINE_REQ_TABLE_RULE_MTR_READ,
	PIPELINE_REQ_TABLE_DSCP_TABLE_UPDATE,
	PIPELINE_REQ_TABLE_RULE_TTL_READ,
	PIPELINE_REQ_MAX
};

struct pipeline_msg_req_port_in_stats_read {
	int clear;
};

struct pipeline_msg_req_port_out_stats_read {
	int clear;
};

struct pipeline_msg_req_table_stats_read {
	int clear;
};

struct pipeline_msg_req_table_rule_add {
	struct softnic_table_rule_match match;
	struct softnic_table_rule_action action;
};

struct pipeline_msg_req_table_rule_add_default {
	struct softnic_table_rule_action action;
};

struct pipeline_msg_req_table_rule_add_bulk {
	struct softnic_table_rule_match *match;
	struct softnic_table_rule_action *action;
	void **data;
	uint32_t n_rules;
	int bulk;
};

struct pipeline_msg_req_table_rule_delete {
	struct softnic_table_rule_match match;
};

struct pipeline_msg_req_table_rule_stats_read {
	void *data;
	int clear;
};

struct pipeline_msg_req_table_mtr_profile_add {
	uint32_t meter_profile_id;
	struct rte_table_action_meter_profile profile;
};

struct pipeline_msg_req_table_mtr_profile_delete {
	uint32_t meter_profile_id;
};

struct pipeline_msg_req_table_rule_mtr_read {
	void *data;
	uint32_t tc_mask;
	int clear;
};

struct pipeline_msg_req_table_dscp_table_update {
	uint64_t dscp_mask;
	struct rte_table_action_dscp_table dscp_table;
};

struct pipeline_msg_req_table_rule_ttl_read {
	void *data;
	int clear;
};

struct pipeline_msg_req {
	enum pipeline_req_type type;
	uint32_t id; /* Port IN, port OUT or table ID */

	RTE_STD_C11
	union {
		struct pipeline_msg_req_port_in_stats_read port_in_stats_read;
		struct pipeline_msg_req_port_out_stats_read port_out_stats_read;
		struct pipeline_msg_req_table_stats_read table_stats_read;
		struct pipeline_msg_req_table_rule_add table_rule_add;
		struct pipeline_msg_req_table_rule_add_default table_rule_add_default;
		struct pipeline_msg_req_table_rule_add_bulk table_rule_add_bulk;
		struct pipeline_msg_req_table_rule_delete table_rule_delete;
		struct pipeline_msg_req_table_rule_stats_read table_rule_stats_read;
		struct pipeline_msg_req_table_mtr_profile_add table_mtr_profile_add;
		struct pipeline_msg_req_table_mtr_profile_delete table_mtr_profile_delete;
		struct pipeline_msg_req_table_rule_mtr_read table_rule_mtr_read;
		struct pipeline_msg_req_table_dscp_table_update table_dscp_table_update;
		struct pipeline_msg_req_table_rule_ttl_read table_rule_ttl_read;
	};
};

struct pipeline_msg_rsp_port_in_stats_read {
	struct rte_pipeline_port_in_stats stats;
};

struct pipeline_msg_rsp_port_out_stats_read {
	struct rte_pipeline_port_out_stats stats;
};

struct pipeline_msg_rsp_table_stats_read {
	struct rte_pipeline_table_stats stats;
};

struct pipeline_msg_rsp_table_rule_add {
	void *data;
};

struct pipeline_msg_rsp_table_rule_add_default {
	void *data;
};

struct pipeline_msg_rsp_table_rule_add_bulk {
	uint32_t n_rules;
};

struct pipeline_msg_rsp_table_rule_stats_read {
	struct rte_table_action_stats_counters stats;
};

struct pipeline_msg_rsp_table_rule_mtr_read {
	struct rte_table_action_mtr_counters stats;
};

struct pipeline_msg_rsp_table_rule_ttl_read {
	struct rte_table_action_ttl_counters stats;
};

struct pipeline_msg_rsp {
	int status;

	RTE_STD_C11
	union {
		struct pipeline_msg_rsp_port_in_stats_read port_in_stats_read;
		struct pipeline_msg_rsp_port_out_stats_read port_out_stats_read;
		struct pipeline_msg_rsp_table_stats_read table_stats_read;
		struct pipeline_msg_rsp_table_rule_add table_rule_add;
		struct pipeline_msg_rsp_table_rule_add_default table_rule_add_default;
		struct pipeline_msg_rsp_table_rule_add_bulk table_rule_add_bulk;
		struct pipeline_msg_rsp_table_rule_stats_read table_rule_stats_read;
		struct pipeline_msg_rsp_table_rule_mtr_read table_rule_mtr_read;
		struct pipeline_msg_rsp_table_rule_ttl_read table_rule_ttl_read;
	};
};

/**
 * Main thread
 */
static struct pipeline_msg_req *
pipeline_msg_alloc(void)
{
	size_t size = RTE_MAX(sizeof(struct pipeline_msg_req),
		sizeof(struct pipeline_msg_rsp));

	return calloc(1, size);
}

static void
pipeline_msg_free(struct pipeline_msg_rsp *rsp)
{
	free(rsp);
}

static struct pipeline_msg_rsp *
pipeline_msg_send_recv(struct pipeline *p,
	struct pipeline_msg_req *req)
{
	struct rte_ring *msgq_req = p->msgq_req;
	struct rte_ring *msgq_rsp = p->msgq_rsp;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* send */
	do {
		status = rte_ring_sp_enqueue(msgq_req, req);
	} while (status == -ENOBUFS);

	/* recv */
	do {
		status = rte_ring_sc_dequeue(msgq_rsp, (void **)&rsp);
	} while (status != 0);

	return rsp;
}

int
softnic_pipeline_port_in_stats_read(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t port_id,
	struct rte_pipeline_port_in_stats *stats,
	int clear)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL ||
		stats == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		port_id >= p->n_ports_in)
		return -1;

	if (!pipeline_is_running(p)) {
		status = rte_pipeline_port_in_stats_read(p->p,
			port_id,
			stats,
			clear);

		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_PORT_IN_STATS_READ;
	req->id = port_id;
	req->port_in_stats_read.clear = clear;

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;
	if (status)
		memcpy(stats, &rsp->port_in_stats_read.stats, sizeof(*stats));

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_port_in_enable(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t port_id)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		port_id >= p->n_ports_in)
		return -1;

	if (!pipeline_is_running(p)) {
		status = rte_pipeline_port_in_enable(p->p, port_id);
		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_PORT_IN_ENABLE;
	req->id = port_id;

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_port_in_disable(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t port_id)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		port_id >= p->n_ports_in)
		return -1;

	if (!pipeline_is_running(p)) {
		status = rte_pipeline_port_in_disable(p->p, port_id);
		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_PORT_IN_DISABLE;
	req->id = port_id;

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_port_out_stats_read(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t port_id,
	struct rte_pipeline_port_out_stats *stats,
	int clear)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL ||
		stats == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		port_id >= p->n_ports_out)
		return -1;

	if (!pipeline_is_running(p)) {
		status = rte_pipeline_port_out_stats_read(p->p,
			port_id,
			stats,
			clear);

		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_PORT_OUT_STATS_READ;
	req->id = port_id;
	req->port_out_stats_read.clear = clear;

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;
	if (status)
		memcpy(stats, &rsp->port_out_stats_read.stats, sizeof(*stats));

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_table_stats_read(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t table_id,
	struct rte_pipeline_table_stats *stats,
	int clear)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL ||
		stats == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		table_id >= p->n_tables)
		return -1;

	if (!pipeline_is_running(p)) {
		status = rte_pipeline_table_stats_read(p->p,
			table_id,
			stats,
			clear);

		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_TABLE_STATS_READ;
	req->id = table_id;
	req->table_stats_read.clear = clear;

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;
	if (status)
		memcpy(stats, &rsp->table_stats_read.stats, sizeof(*stats));

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

static int
match_check(struct softnic_table_rule_match *match,
	struct pipeline *p,
	uint32_t table_id)
{
	struct softnic_table *table;

	if (match == NULL ||
		p == NULL ||
		table_id >= p->n_tables)
		return -1;

	table = &p->table[table_id];
	if (match->match_type != table->params.match_type)
		return -1;

	switch (match->match_type) {
	case TABLE_ACL:
	{
		struct softnic_table_acl_params *t = &table->params.match.acl;
		struct softnic_table_rule_match_acl *r = &match->match.acl;

		if ((r->ip_version && (t->ip_version == 0)) ||
			((r->ip_version == 0) && t->ip_version))
			return -1;

		if (r->ip_version) {
			if (r->sa_depth > 32 ||
				r->da_depth > 32)
				return -1;
		} else {
			if (r->sa_depth > 128 ||
				r->da_depth > 128)
				return -1;
		}
		return 0;
	}

	case TABLE_ARRAY:
		return 0;

	case TABLE_HASH:
		return 0;

	case TABLE_LPM:
	{
		struct softnic_table_lpm_params *t = &table->params.match.lpm;
		struct softnic_table_rule_match_lpm *r = &match->match.lpm;

		if ((r->ip_version && (t->key_size != 4)) ||
			((r->ip_version == 0) && (t->key_size != 16)))
			return -1;

		if (r->ip_version) {
			if (r->depth > 32)
				return -1;
		} else {
			if (r->depth > 128)
				return -1;
		}
		return 0;
	}

	case TABLE_STUB:
		return -1;

	default:
		return -1;
	}
}

static int
action_check(struct softnic_table_rule_action *action,
	struct pipeline *p,
	uint32_t table_id)
{
	struct softnic_table_action_profile *ap;

	if (action == NULL ||
		p == NULL ||
		table_id >= p->n_tables)
		return -1;

	ap = p->table[table_id].ap;
	if (action->action_mask != ap->params.action_mask)
		return -1;

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_FWD)) {
		if (action->fwd.action == RTE_PIPELINE_ACTION_PORT &&
			action->fwd.id >= p->n_ports_out)
			return -1;

		if (action->fwd.action == RTE_PIPELINE_ACTION_TABLE &&
			action->fwd.id >= p->n_tables)
			return -1;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_MTR)) {
		uint32_t tc_mask0 = (1 << ap->params.mtr.n_tc) - 1;
		uint32_t tc_mask1 = action->mtr.tc_mask;

		if (tc_mask1 != tc_mask0)
			return -1;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_TM)) {
		uint32_t n_subports_per_port =
			ap->params.tm.n_subports_per_port;
		uint32_t n_pipes_per_subport =
			ap->params.tm.n_pipes_per_subport;
		uint32_t subport_id = action->tm.subport_id;
		uint32_t pipe_id = action->tm.pipe_id;

		if (subport_id >= n_subports_per_port ||
			pipe_id >= n_pipes_per_subport)
			return -1;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_ENCAP)) {
		uint64_t encap_mask = ap->params.encap.encap_mask;
		enum rte_table_action_encap_type type = action->encap.type;

		if ((encap_mask & (1LLU << type)) == 0)
			return -1;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_NAT)) {
		int ip_version0 = ap->params.common.ip_version;
		int ip_version1 = action->nat.ip_version;

		if ((ip_version1 && (ip_version0 == 0)) ||
			((ip_version1 == 0) && ip_version0))
			return -1;
	}

	return 0;
}

static int
action_default_check(struct softnic_table_rule_action *action,
	struct pipeline *p,
	uint32_t table_id)
{
	if (action == NULL ||
		action->action_mask != (1LLU << RTE_TABLE_ACTION_FWD) ||
		p == NULL ||
		table_id >= p->n_tables)
		return -1;

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_FWD)) {
		if (action->fwd.action == RTE_PIPELINE_ACTION_PORT &&
			action->fwd.id >= p->n_ports_out)
			return -1;

		if (action->fwd.action == RTE_PIPELINE_ACTION_TABLE &&
			action->fwd.id >= p->n_tables)
			return -1;
	}

	return 0;
}

union table_rule_match_low_level {
	struct rte_table_acl_rule_add_params acl_add;
	struct rte_table_acl_rule_delete_params acl_delete;
	struct rte_table_array_key array;
	uint8_t hash[TABLE_RULE_MATCH_SIZE_MAX];
	struct rte_table_lpm_key lpm_ipv4;
	struct rte_table_lpm_ipv6_key lpm_ipv6;
};

static int
match_colwert(struct softnic_table_rule_match *mh,
	union table_rule_match_low_level *ml,
	int add);

static int
action_colwert(struct rte_table_action *a,
	struct softnic_table_rule_action *action,
	struct rte_pipeline_table_entry *data);

int
softnic_pipeline_table_rule_add(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t table_id,
	struct softnic_table_rule_match *match,
	struct softnic_table_rule_action *action,
	void **data)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL ||
		match == NULL ||
		action == NULL ||
		data == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		table_id >= p->n_tables ||
		match_check(match, p, table_id) ||
		action_check(action, p, table_id))
		return -1;

	if (!pipeline_is_running(p)) {
		struct rte_table_action *a = p->table[table_id].a;
		union table_rule_match_low_level match_ll;
		struct rte_pipeline_table_entry *data_in, *data_out;
		int key_found;
		uint8_t *buffer;

		buffer = calloc(TABLE_RULE_ACTION_SIZE_MAX, sizeof(uint8_t));
		if (buffer == NULL)
			return -1;

		/* Table match-action rule colwersion */
		data_in = (struct rte_pipeline_table_entry *)buffer;

		status = match_colwert(match, &match_ll, 1);
		if (status) {
			free(buffer);
			return -1;
		}

		status = action_colwert(a, action, data_in);
		if (status) {
			free(buffer);
			return -1;
		}

		/* Add rule (match, action) to table */
		status = rte_pipeline_table_entry_add(p->p,
				table_id,
				&match_ll,
				data_in,
				&key_found,
				&data_out);
		if (status) {
			free(buffer);
			return -1;
		}

		/* Write Response */
		*data = data_out;

		free(buffer);
		return 0;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_TABLE_RULE_ADD;
	req->id = table_id;
	memcpy(&req->table_rule_add.match, match, sizeof(*match));
	memcpy(&req->table_rule_add.action, action, sizeof(*action));

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;
	if (status == 0)
		*data = rsp->table_rule_add.data;

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_table_rule_add_default(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t table_id,
	struct softnic_table_rule_action *action,
	void **data)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL ||
		action == NULL ||
		data == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		table_id >= p->n_tables ||
		action_default_check(action, p, table_id))
		return -1;

	if (!pipeline_is_running(p)) {
		struct rte_pipeline_table_entry *data_in, *data_out;
		uint8_t *buffer;

		buffer = calloc(TABLE_RULE_ACTION_SIZE_MAX, sizeof(uint8_t));
		if (buffer == NULL)
			return -1;

		/* Apply actions */
		data_in = (struct rte_pipeline_table_entry *)buffer;

		data_in->action = action->fwd.action;
		if (action->fwd.action == RTE_PIPELINE_ACTION_PORT)
			data_in->port_id = action->fwd.id;
		if (action->fwd.action == RTE_PIPELINE_ACTION_TABLE)
			data_in->table_id = action->fwd.id;

		/* Add default rule to table */
		status = rte_pipeline_table_default_entry_add(p->p,
				table_id,
				data_in,
				&data_out);
		if (status) {
			free(buffer);
			return -1;
		}

		/* Write Response */
		*data = data_out;

		free(buffer);
		return 0;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_TABLE_RULE_ADD_DEFAULT;
	req->id = table_id;
	memcpy(&req->table_rule_add_default.action, action, sizeof(*action));

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;
	if (status == 0)
		*data = rsp->table_rule_add_default.data;

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_table_rule_add_bulk(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t table_id,
	struct softnic_table_rule_match *match,
	struct softnic_table_rule_action *action,
	void **data,
	uint32_t *n_rules)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	uint32_t i;
	int status;

	/* Check input params */
	if (pipeline_name == NULL ||
		match == NULL ||
		action == NULL ||
		data == NULL ||
		n_rules == NULL ||
		(*n_rules == 0))
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		table_id >= p->n_tables)
		return -1;

	for (i = 0; i < *n_rules; i++)
		if (match_check(match, p, table_id) ||
			action_check(action, p, table_id))
			return -1;

	if (!pipeline_is_running(p)) {
		struct rte_table_action *a = p->table[table_id].a;
		union table_rule_match_low_level *match_ll;
		uint8_t *action_ll;
		void **match_ll_ptr;
		struct rte_pipeline_table_entry **action_ll_ptr;
		struct rte_pipeline_table_entry **entries_ptr =
			(struct rte_pipeline_table_entry **)data;
		uint32_t bulk =
			(p->table[table_id].params.match_type == TABLE_ACL) ? 1 : 0;
		int *found;

		/* Memory allocation */
		match_ll = calloc(*n_rules, sizeof(union table_rule_match_low_level));
		action_ll = calloc(*n_rules, TABLE_RULE_ACTION_SIZE_MAX);
		match_ll_ptr = calloc(*n_rules, sizeof(void *));
		action_ll_ptr =
			calloc(*n_rules, sizeof(struct rte_pipeline_table_entry *));
		found = calloc(*n_rules, sizeof(int));

		if (match_ll == NULL ||
			action_ll == NULL ||
			match_ll_ptr == NULL ||
			action_ll_ptr == NULL ||
			found == NULL)
			goto fail;

		for (i = 0; i < *n_rules; i++) {
			match_ll_ptr[i] = (void *)&match_ll[i];
			action_ll_ptr[i] =
				(struct rte_pipeline_table_entry *)&action_ll[i * TABLE_RULE_ACTION_SIZE_MAX];
		}

		/* Rule match colwersion */
		for (i = 0; i < *n_rules; i++) {
			status = match_colwert(&match[i], match_ll_ptr[i], 1);
			if (status)
				goto fail;
		}

		/* Rule action colwersion */
		for (i = 0; i < *n_rules; i++) {
			status = action_colwert(a, &action[i], action_ll_ptr[i]);
			if (status)
				goto fail;
		}

		/* Add rule (match, action) to table */
		if (bulk) {
			status = rte_pipeline_table_entry_add_bulk(p->p,
				table_id,
				match_ll_ptr,
				action_ll_ptr,
				*n_rules,
				found,
				entries_ptr);
			if (status)
				*n_rules = 0;
		} else {
			for (i = 0; i < *n_rules; i++) {
				status = rte_pipeline_table_entry_add(p->p,
					table_id,
					match_ll_ptr[i],
					action_ll_ptr[i],
					&found[i],
					&entries_ptr[i]);
				if (status) {
					*n_rules = i;
					break;
				}
			}
		}

		/* Free */
		free(found);
		free(action_ll_ptr);
		free(match_ll_ptr);
		free(action_ll);
		free(match_ll);

		return status;

fail:
		free(found);
		free(action_ll_ptr);
		free(match_ll_ptr);
		free(action_ll);
		free(match_ll);

		*n_rules = 0;
		return -1;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_TABLE_RULE_ADD_BULK;
	req->id = table_id;
	req->table_rule_add_bulk.match = match;
	req->table_rule_add_bulk.action = action;
	req->table_rule_add_bulk.data = data;
	req->table_rule_add_bulk.n_rules = *n_rules;
	req->table_rule_add_bulk.bulk =
		(p->table[table_id].params.match_type == TABLE_ACL) ? 1 : 0;

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;
	if (status == 0)
		*n_rules = rsp->table_rule_add_bulk.n_rules;

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_table_rule_delete(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t table_id,
	struct softnic_table_rule_match *match)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL ||
		match == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		table_id >= p->n_tables ||
		match_check(match, p, table_id))
		return -1;

	if (!pipeline_is_running(p)) {
		union table_rule_match_low_level match_ll;
		int key_found;

		status = match_colwert(match, &match_ll, 0);
		if (status)
			return -1;

		status = rte_pipeline_table_entry_delete(p->p,
				table_id,
				&match_ll,
				&key_found,
				NULL);

		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_TABLE_RULE_DELETE;
	req->id = table_id;
	memcpy(&req->table_rule_delete.match, match, sizeof(*match));

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_table_rule_delete_default(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t table_id)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		table_id >= p->n_tables)
		return -1;

	if (!pipeline_is_running(p)) {
		status = rte_pipeline_table_default_entry_delete(p->p,
			table_id,
			NULL);

		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_TABLE_RULE_DELETE_DEFAULT;
	req->id = table_id;

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_table_rule_stats_read(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t table_id,
	void *data,
	struct rte_table_action_stats_counters *stats,
	int clear)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL ||
		data == NULL ||
		stats == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		table_id >= p->n_tables)
		return -1;

	if (!pipeline_is_running(p)) {
		struct rte_table_action *a = p->table[table_id].a;

		status = rte_table_action_stats_read(a,
			data,
			stats,
			clear);

		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_TABLE_RULE_STATS_READ;
	req->id = table_id;
	req->table_rule_stats_read.data = data;
	req->table_rule_stats_read.clear = clear;

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;
	if (status)
		memcpy(stats, &rsp->table_rule_stats_read.stats, sizeof(*stats));

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_table_mtr_profile_add(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t table_id,
	uint32_t meter_profile_id,
	struct rte_table_action_meter_profile *profile)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	struct softnic_table *table;
	struct softnic_table_meter_profile *mp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL ||
		profile == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		table_id >= p->n_tables)
		return -1;

	table = &p->table[table_id];
	mp = softnic_pipeline_table_meter_profile_find(table, meter_profile_id);
	if (mp)
		return -1;

	/* Resource Allocation */
	mp = calloc(1, sizeof(struct softnic_table_meter_profile));
	if (mp == NULL)
		return -1;

	mp->meter_profile_id = meter_profile_id;
	memcpy(&mp->profile, profile, sizeof(mp->profile));

	if (!pipeline_is_running(p)) {
		status = rte_table_action_meter_profile_add(table->a,
			meter_profile_id,
			profile);
		if (status) {
			free(mp);
			return status;
		}

		/* Add profile to the table. */
		TAILQ_INSERT_TAIL(&table->meter_profiles, mp, node);

		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL) {
		free(mp);
		return -1;
	}

	/* Write request */
	req->type = PIPELINE_REQ_TABLE_MTR_PROFILE_ADD;
	req->id = table_id;
	req->table_mtr_profile_add.meter_profile_id = meter_profile_id;
	memcpy(&req->table_mtr_profile_add.profile, profile, sizeof(*profile));

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;
	if (status == 0)
		TAILQ_INSERT_TAIL(&table->meter_profiles, mp, node);
	else
		free(mp);

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_table_mtr_profile_delete(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t table_id,
	uint32_t meter_profile_id)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		table_id >= p->n_tables)
		return -1;

	if (!pipeline_is_running(p)) {
		struct rte_table_action *a = p->table[table_id].a;

		status = rte_table_action_meter_profile_delete(a,
				meter_profile_id);

		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_TABLE_MTR_PROFILE_DELETE;
	req->id = table_id;
	req->table_mtr_profile_delete.meter_profile_id = meter_profile_id;

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_table_rule_mtr_read(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t table_id,
	void *data,
	uint32_t tc_mask,
	struct rte_table_action_mtr_counters *stats,
	int clear)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL ||
		data == NULL ||
		stats == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		table_id >= p->n_tables)
		return -1;

	if (!pipeline_is_running(p)) {
		struct rte_table_action *a = p->table[table_id].a;

		status = rte_table_action_meter_read(a,
				data,
				tc_mask,
				stats,
				clear);

		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_TABLE_RULE_MTR_READ;
	req->id = table_id;
	req->table_rule_mtr_read.data = data;
	req->table_rule_mtr_read.tc_mask = tc_mask;
	req->table_rule_mtr_read.clear = clear;

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;
	if (status)
		memcpy(stats, &rsp->table_rule_mtr_read.stats, sizeof(*stats));

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_table_dscp_table_update(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t table_id,
	uint64_t dscp_mask,
	struct rte_table_action_dscp_table *dscp_table)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL ||
		dscp_table == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		table_id >= p->n_tables)
		return -1;

	if (!pipeline_is_running(p)) {
		struct rte_table_action *a = p->table[table_id].a;

		status = rte_table_action_dscp_table_update(a,
				dscp_mask,
				dscp_table);

		/* Update table dscp table */
		if (!status)
			memcpy(&p->table[table_id].dscp_table, dscp_table,
				sizeof(p->table[table_id].dscp_table));

		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_TABLE_DSCP_TABLE_UPDATE;
	req->id = table_id;
	req->table_dscp_table_update.dscp_mask = dscp_mask;
	memcpy(&req->table_dscp_table_update.dscp_table,
		dscp_table, sizeof(*dscp_table));

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;

	/* Update table dscp table */
	if (!status)
		memcpy(&p->table[table_id].dscp_table, dscp_table,
			sizeof(p->table[table_id].dscp_table));

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

int
softnic_pipeline_table_rule_ttl_read(struct pmd_internals *softnic,
	const char *pipeline_name,
	uint32_t table_id,
	void *data,
	struct rte_table_action_ttl_counters *stats,
	int clear)
{
	struct pipeline *p;
	struct pipeline_msg_req *req;
	struct pipeline_msg_rsp *rsp;
	int status;

	/* Check input params */
	if (pipeline_name == NULL ||
		data == NULL ||
		stats == NULL)
		return -1;

	p = softnic_pipeline_find(softnic, pipeline_name);
	if (p == NULL ||
		table_id >= p->n_tables)
		return -1;

	if (!pipeline_is_running(p)) {
		struct rte_table_action *a = p->table[table_id].a;

		status = rte_table_action_ttl_read(a,
				data,
				stats,
				clear);

		return status;
	}

	/* Allocate request */
	req = pipeline_msg_alloc();
	if (req == NULL)
		return -1;

	/* Write request */
	req->type = PIPELINE_REQ_TABLE_RULE_TTL_READ;
	req->id = table_id;
	req->table_rule_ttl_read.data = data;
	req->table_rule_ttl_read.clear = clear;

	/* Send request and wait for response */
	rsp = pipeline_msg_send_recv(p, req);

	/* Read response */
	status = rsp->status;
	if (status)
		memcpy(stats, &rsp->table_rule_ttl_read.stats, sizeof(*stats));

	/* Free response */
	pipeline_msg_free(rsp);

	return status;
}

/**
 * Data plane threads: message handling
 */
static inline struct pipeline_msg_req *
pipeline_msg_recv(struct rte_ring *msgq_req)
{
	struct pipeline_msg_req *req;

	int status = rte_ring_sc_dequeue(msgq_req, (void **)&req);

	if (status != 0)
		return NULL;

	return req;
}

static inline void
pipeline_msg_send(struct rte_ring *msgq_rsp,
	struct pipeline_msg_rsp *rsp)
{
	int status;

	do {
		status = rte_ring_sp_enqueue(msgq_rsp, rsp);
	} while (status == -ENOBUFS);
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_port_in_stats_read(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	uint32_t port_id = req->id;
	int clear = req->port_in_stats_read.clear;

	rsp->status = rte_pipeline_port_in_stats_read(p->p,
		port_id,
		&rsp->port_in_stats_read.stats,
		clear);

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_port_in_enable(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	uint32_t port_id = req->id;

	rsp->status = rte_pipeline_port_in_enable(p->p,
		port_id);

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_port_in_disable(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	uint32_t port_id = req->id;

	rsp->status = rte_pipeline_port_in_disable(p->p,
		port_id);

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_port_out_stats_read(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	uint32_t port_id = req->id;
	int clear = req->port_out_stats_read.clear;

	rsp->status = rte_pipeline_port_out_stats_read(p->p,
		port_id,
		&rsp->port_out_stats_read.stats,
		clear);

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_table_stats_read(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	uint32_t port_id = req->id;
	int clear = req->table_stats_read.clear;

	rsp->status = rte_pipeline_table_stats_read(p->p,
		port_id,
		&rsp->table_stats_read.stats,
		clear);

	return rsp;
}

static int
match_colwert_ipv6_depth(uint32_t depth, uint32_t *depth32)
{
	if (depth > 128)
		return -1;

	switch (depth / 32) {
	case 0:
		depth32[0] = depth;
		depth32[1] = 0;
		depth32[2] = 0;
		depth32[3] = 0;
		return 0;

	case 1:
		depth32[0] = 32;
		depth32[1] = depth - 32;
		depth32[2] = 0;
		depth32[3] = 0;
		return 0;

	case 2:
		depth32[0] = 32;
		depth32[1] = 32;
		depth32[2] = depth - 64;
		depth32[3] = 0;
		return 0;

	case 3:
		depth32[0] = 32;
		depth32[1] = 32;
		depth32[2] = 32;
		depth32[3] = depth - 96;
		return 0;

	case 4:
		depth32[0] = 32;
		depth32[1] = 32;
		depth32[2] = 32;
		depth32[3] = 32;
		return 0;

	default:
		return -1;
	}
}

static int
match_colwert(struct softnic_table_rule_match *mh,
	union table_rule_match_low_level *ml,
	int add)
{
	memset(ml, 0, sizeof(*ml));

	switch (mh->match_type) {
	case TABLE_ACL:
		if (mh->match.acl.ip_version)
			if (add) {
				ml->acl_add.field_value[0].value.u8 =
					mh->match.acl.proto;
				ml->acl_add.field_value[0].mask_range.u8 =
					mh->match.acl.proto_mask;

				ml->acl_add.field_value[1].value.u32 =
					mh->match.acl.ipv4.sa;
				ml->acl_add.field_value[1].mask_range.u32 =
					mh->match.acl.sa_depth;

				ml->acl_add.field_value[2].value.u32 =
					mh->match.acl.ipv4.da;
				ml->acl_add.field_value[2].mask_range.u32 =
					mh->match.acl.da_depth;

				ml->acl_add.field_value[3].value.u16 =
					mh->match.acl.sp0;
				ml->acl_add.field_value[3].mask_range.u16 =
					mh->match.acl.sp1;

				ml->acl_add.field_value[4].value.u16 =
					mh->match.acl.dp0;
				ml->acl_add.field_value[4].mask_range.u16 =
					mh->match.acl.dp1;

				ml->acl_add.priority =
					(int32_t)mh->match.acl.priority;
			} else {
				ml->acl_delete.field_value[0].value.u8 =
					mh->match.acl.proto;
				ml->acl_delete.field_value[0].mask_range.u8 =
					mh->match.acl.proto_mask;

				ml->acl_delete.field_value[1].value.u32 =
					mh->match.acl.ipv4.sa;
				ml->acl_delete.field_value[1].mask_range.u32 =
					mh->match.acl.sa_depth;

				ml->acl_delete.field_value[2].value.u32 =
					mh->match.acl.ipv4.da;
				ml->acl_delete.field_value[2].mask_range.u32 =
					mh->match.acl.da_depth;

				ml->acl_delete.field_value[3].value.u16 =
					mh->match.acl.sp0;
				ml->acl_delete.field_value[3].mask_range.u16 =
					mh->match.acl.sp1;

				ml->acl_delete.field_value[4].value.u16 =
					mh->match.acl.dp0;
				ml->acl_delete.field_value[4].mask_range.u16 =
					mh->match.acl.dp1;
			}
		else
			if (add) {
				uint32_t *sa32 =
					(uint32_t *)mh->match.acl.ipv6.sa;
				uint32_t *da32 =
					(uint32_t *)mh->match.acl.ipv6.da;
				uint32_t sa32_depth[4], da32_depth[4];
				int status;

				status = match_colwert_ipv6_depth(mh->match.acl.sa_depth,
					sa32_depth);
				if (status)
					return status;

				status = match_colwert_ipv6_depth(
					mh->match.acl.da_depth,
					da32_depth);
				if (status)
					return status;

				ml->acl_add.field_value[0].value.u8 =
					mh->match.acl.proto;
				ml->acl_add.field_value[0].mask_range.u8 =
					mh->match.acl.proto_mask;

				ml->acl_add.field_value[1].value.u32 =
					rte_be_to_cpu_32(sa32[0]);
				ml->acl_add.field_value[1].mask_range.u32 =
					sa32_depth[0];
				ml->acl_add.field_value[2].value.u32 =
					rte_be_to_cpu_32(sa32[1]);
				ml->acl_add.field_value[2].mask_range.u32 =
					sa32_depth[1];
				ml->acl_add.field_value[3].value.u32 =
					rte_be_to_cpu_32(sa32[2]);
				ml->acl_add.field_value[3].mask_range.u32 =
					sa32_depth[2];
				ml->acl_add.field_value[4].value.u32 =
					rte_be_to_cpu_32(sa32[3]);
				ml->acl_add.field_value[4].mask_range.u32 =
					sa32_depth[3];

				ml->acl_add.field_value[5].value.u32 =
					rte_be_to_cpu_32(da32[0]);
				ml->acl_add.field_value[5].mask_range.u32 =
					da32_depth[0];
				ml->acl_add.field_value[6].value.u32 =
					rte_be_to_cpu_32(da32[1]);
				ml->acl_add.field_value[6].mask_range.u32 =
					da32_depth[1];
				ml->acl_add.field_value[7].value.u32 =
					rte_be_to_cpu_32(da32[2]);
				ml->acl_add.field_value[7].mask_range.u32 =
					da32_depth[2];
				ml->acl_add.field_value[8].value.u32 =
					rte_be_to_cpu_32(da32[3]);
				ml->acl_add.field_value[8].mask_range.u32 =
					da32_depth[3];

				ml->acl_add.field_value[9].value.u16 =
					mh->match.acl.sp0;
				ml->acl_add.field_value[9].mask_range.u16 =
					mh->match.acl.sp1;

				ml->acl_add.field_value[10].value.u16 =
					mh->match.acl.dp0;
				ml->acl_add.field_value[10].mask_range.u16 =
					mh->match.acl.dp1;

				ml->acl_add.priority =
					(int32_t)mh->match.acl.priority;
			} else {
				uint32_t *sa32 =
					(uint32_t *)mh->match.acl.ipv6.sa;
				uint32_t *da32 =
					(uint32_t *)mh->match.acl.ipv6.da;
				uint32_t sa32_depth[4], da32_depth[4];
				int status;

				status = match_colwert_ipv6_depth(mh->match.acl.sa_depth,
					sa32_depth);
				if (status)
					return status;

				status = match_colwert_ipv6_depth(mh->match.acl.da_depth,
					da32_depth);
				if (status)
					return status;

				ml->acl_delete.field_value[0].value.u8 =
					mh->match.acl.proto;
				ml->acl_delete.field_value[0].mask_range.u8 =
					mh->match.acl.proto_mask;

				ml->acl_delete.field_value[1].value.u32 =
					rte_be_to_cpu_32(sa32[0]);
				ml->acl_delete.field_value[1].mask_range.u32 =
					sa32_depth[0];
				ml->acl_delete.field_value[2].value.u32 =
					rte_be_to_cpu_32(sa32[1]);
				ml->acl_delete.field_value[2].mask_range.u32 =
					sa32_depth[1];
				ml->acl_delete.field_value[3].value.u32 =
					rte_be_to_cpu_32(sa32[2]);
				ml->acl_delete.field_value[3].mask_range.u32 =
					sa32_depth[2];
				ml->acl_delete.field_value[4].value.u32 =
					rte_be_to_cpu_32(sa32[3]);
				ml->acl_delete.field_value[4].mask_range.u32 =
					sa32_depth[3];

				ml->acl_delete.field_value[5].value.u32 =
					rte_be_to_cpu_32(da32[0]);
				ml->acl_delete.field_value[5].mask_range.u32 =
					da32_depth[0];
				ml->acl_delete.field_value[6].value.u32 =
					rte_be_to_cpu_32(da32[1]);
				ml->acl_delete.field_value[6].mask_range.u32 =
					da32_depth[1];
				ml->acl_delete.field_value[7].value.u32 =
					rte_be_to_cpu_32(da32[2]);
				ml->acl_delete.field_value[7].mask_range.u32 =
					da32_depth[2];
				ml->acl_delete.field_value[8].value.u32 =
					rte_be_to_cpu_32(da32[3]);
				ml->acl_delete.field_value[8].mask_range.u32 =
					da32_depth[3];

				ml->acl_delete.field_value[9].value.u16 =
					mh->match.acl.sp0;
				ml->acl_delete.field_value[9].mask_range.u16 =
					mh->match.acl.sp1;

				ml->acl_delete.field_value[10].value.u16 =
					mh->match.acl.dp0;
				ml->acl_delete.field_value[10].mask_range.u16 =
					mh->match.acl.dp1;
			}
		return 0;

	case TABLE_ARRAY:
		ml->array.pos = mh->match.array.pos;
		return 0;

	case TABLE_HASH:
		memcpy(ml->hash, mh->match.hash.key, sizeof(ml->hash));
		return 0;

	case TABLE_LPM:
		if (mh->match.lpm.ip_version) {
			ml->lpm_ipv4.ip = mh->match.lpm.ipv4;
			ml->lpm_ipv4.depth = mh->match.lpm.depth;
		} else {
			memcpy(ml->lpm_ipv6.ip,
				mh->match.lpm.ipv6, sizeof(ml->lpm_ipv6.ip));
			ml->lpm_ipv6.depth = mh->match.lpm.depth;
		}

		return 0;

	default:
		return -1;
	}
}

static int
action_colwert(struct rte_table_action *a,
	struct softnic_table_rule_action *action,
	struct rte_pipeline_table_entry *data)
{
	int status;

	/* Apply actions */
	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_FWD)) {
		status = rte_table_action_apply(a,
			data,
			RTE_TABLE_ACTION_FWD,
			&action->fwd);

		if (status)
			return status;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_LB)) {
		status = rte_table_action_apply(a,
			data,
			RTE_TABLE_ACTION_LB,
			&action->lb);

		if (status)
			return status;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_MTR)) {
		status = rte_table_action_apply(a,
			data,
			RTE_TABLE_ACTION_MTR,
			&action->mtr);

		if (status)
			return status;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_TM)) {
		status = rte_table_action_apply(a,
			data,
			RTE_TABLE_ACTION_TM,
			&action->tm);

		if (status)
			return status;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_ENCAP)) {
		status = rte_table_action_apply(a,
			data,
			RTE_TABLE_ACTION_ENCAP,
			&action->encap);

		if (status)
			return status;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_NAT)) {
		status = rte_table_action_apply(a,
			data,
			RTE_TABLE_ACTION_NAT,
			&action->nat);

		if (status)
			return status;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_TTL)) {
		status = rte_table_action_apply(a,
			data,
			RTE_TABLE_ACTION_TTL,
			&action->ttl);

		if (status)
			return status;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_STATS)) {
		status = rte_table_action_apply(a,
			data,
			RTE_TABLE_ACTION_STATS,
			&action->stats);

		if (status)
			return status;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_TIME)) {
		status = rte_table_action_apply(a,
			data,
			RTE_TABLE_ACTION_TIME,
			&action->time);

		if (status)
			return status;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_TAG)) {
		status = rte_table_action_apply(a,
			data,
			RTE_TABLE_ACTION_TAG,
			&action->tag);

		if (status)
			return status;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_DECAP)) {
		status = rte_table_action_apply(a,
			data,
			RTE_TABLE_ACTION_DECAP,
			&action->decap);

		if (status)
			return status;
	}

	if (action->action_mask & (1LLU << RTE_TABLE_ACTION_SYM_CRYPTO)) {
		status = rte_table_action_apply(a,
			data,
			RTE_TABLE_ACTION_SYM_CRYPTO,
			&action->sym_crypto);

		if (status)
			return status;
	}

	return 0;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_table_rule_add(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	union table_rule_match_low_level match_ll;
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	struct softnic_table_rule_match *match = &req->table_rule_add.match;
	struct softnic_table_rule_action *action = &req->table_rule_add.action;
	struct rte_pipeline_table_entry *data_in, *data_out;
	uint32_t table_id = req->id;
	int key_found, status;
	struct rte_table_action *a = p->table_data[table_id].a;

	/* Apply actions */
	memset(p->buffer, 0, sizeof(p->buffer));
	data_in = (struct rte_pipeline_table_entry *)p->buffer;

	status = match_colwert(match, &match_ll, 1);
	if (status) {
		rsp->status = -1;
		return rsp;
	}

	status = action_colwert(a, action, data_in);
	if (status) {
		rsp->status = -1;
		return rsp;
	}

	status = rte_pipeline_table_entry_add(p->p,
		table_id,
		&match_ll,
		data_in,
		&key_found,
		&data_out);
	if (status) {
		rsp->status = -1;
		return rsp;
	}

	/* Write response */
	rsp->status = 0;
	rsp->table_rule_add.data = data_out;

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_table_rule_add_default(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	struct softnic_table_rule_action *action = &req->table_rule_add_default.action;
	struct rte_pipeline_table_entry *data_in, *data_out;
	uint32_t table_id = req->id;
	int status;

	/* Apply actions */
	memset(p->buffer, 0, sizeof(p->buffer));
	data_in = (struct rte_pipeline_table_entry *)p->buffer;

	data_in->action = action->fwd.action;
	if (action->fwd.action == RTE_PIPELINE_ACTION_PORT)
		data_in->port_id = action->fwd.id;
	if (action->fwd.action == RTE_PIPELINE_ACTION_TABLE)
		data_in->table_id = action->fwd.id;

	/* Add default rule to table */
	status = rte_pipeline_table_default_entry_add(p->p,
		table_id,
		data_in,
		&data_out);
	if (status) {
		rsp->status = -1;
		return rsp;
	}

	/* Write response */
	rsp->status = 0;
	rsp->table_rule_add_default.data = data_out;

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_table_rule_add_bulk(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;

	uint32_t table_id = req->id;
	struct softnic_table_rule_match *match = req->table_rule_add_bulk.match;
	struct softnic_table_rule_action *action = req->table_rule_add_bulk.action;
	struct rte_pipeline_table_entry **data =
		(struct rte_pipeline_table_entry **)req->table_rule_add_bulk.data;
	uint32_t n_rules = req->table_rule_add_bulk.n_rules;
	uint32_t bulk = req->table_rule_add_bulk.bulk;

	struct rte_table_action *a = p->table_data[table_id].a;
	union table_rule_match_low_level *match_ll;
	uint8_t *action_ll;
	void **match_ll_ptr;
	struct rte_pipeline_table_entry **action_ll_ptr;
	int *found, status;
	uint32_t i;

	/* Memory allocation */
	match_ll = calloc(n_rules, sizeof(union table_rule_match_low_level));
	action_ll = calloc(n_rules, TABLE_RULE_ACTION_SIZE_MAX);
	match_ll_ptr = calloc(n_rules, sizeof(void *));
	action_ll_ptr =
		calloc(n_rules, sizeof(struct rte_pipeline_table_entry *));
	found = calloc(n_rules, sizeof(int));

	if (match_ll == NULL ||
		action_ll == NULL ||
		match_ll_ptr == NULL ||
		action_ll_ptr == NULL ||
		found == NULL)
		goto fail;

	for (i = 0; i < n_rules; i++) {
		match_ll_ptr[i] = (void *)&match_ll[i];
		action_ll_ptr[i] =
			(struct rte_pipeline_table_entry *)&action_ll[i * TABLE_RULE_ACTION_SIZE_MAX];
	}

	/* Rule match colwersion */
	for (i = 0; i < n_rules; i++) {
		status = match_colwert(&match[i], match_ll_ptr[i], 1);
		if (status)
			goto fail;
	}

	/* Rule action colwersion */
	for (i = 0; i < n_rules; i++) {
		status = action_colwert(a, &action[i], action_ll_ptr[i]);
		if (status)
			goto fail;
	}

	/* Add rule (match, action) to table */
	if (bulk) {
		status = rte_pipeline_table_entry_add_bulk(p->p,
			table_id,
			match_ll_ptr,
			action_ll_ptr,
			n_rules,
			found,
			data);
		if (status)
			n_rules = 0;
	} else {
		for (i = 0; i < n_rules; i++) {
			status = rte_pipeline_table_entry_add(p->p,
				table_id,
				match_ll_ptr[i],
				action_ll_ptr[i],
				&found[i],
				&data[i]);
			if (status) {
				n_rules = i;
				break;
			}
		}
	}

	/* Write response */
	rsp->status = 0;
	rsp->table_rule_add_bulk.n_rules = n_rules;

	/* Free */
	free(found);
	free(action_ll_ptr);
	free(match_ll_ptr);
	free(action_ll);
	free(match_ll);

	return rsp;

fail:
	free(found);
	free(action_ll_ptr);
	free(match_ll_ptr);
	free(action_ll);
	free(match_ll);

	rsp->status = -1;
	rsp->table_rule_add_bulk.n_rules = 0;
	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_table_rule_delete(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	union table_rule_match_low_level match_ll;
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	struct softnic_table_rule_match *match = &req->table_rule_delete.match;
	uint32_t table_id = req->id;
	int key_found, status;

	status = match_colwert(match, &match_ll, 0);
	if (status) {
		rsp->status = -1;
		return rsp;
	}

	rsp->status = rte_pipeline_table_entry_delete(p->p,
		table_id,
		&match_ll,
		&key_found,
		NULL);

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_table_rule_delete_default(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	uint32_t table_id = req->id;

	rsp->status = rte_pipeline_table_default_entry_delete(p->p,
		table_id,
		NULL);

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_table_rule_stats_read(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	uint32_t table_id = req->id;
	void *data = req->table_rule_stats_read.data;
	int clear = req->table_rule_stats_read.clear;
	struct rte_table_action *a = p->table_data[table_id].a;

	rsp->status = rte_table_action_stats_read(a,
		data,
		&rsp->table_rule_stats_read.stats,
		clear);

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_table_mtr_profile_add(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	uint32_t table_id = req->id;
	uint32_t meter_profile_id = req->table_mtr_profile_add.meter_profile_id;
	struct rte_table_action_meter_profile *profile =
		&req->table_mtr_profile_add.profile;
	struct rte_table_action *a = p->table_data[table_id].a;

	rsp->status = rte_table_action_meter_profile_add(a,
		meter_profile_id,
		profile);

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_table_mtr_profile_delete(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	uint32_t table_id = req->id;
	uint32_t meter_profile_id =
		req->table_mtr_profile_delete.meter_profile_id;
	struct rte_table_action *a = p->table_data[table_id].a;

	rsp->status = rte_table_action_meter_profile_delete(a,
		meter_profile_id);

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_table_rule_mtr_read(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	uint32_t table_id = req->id;
	void *data = req->table_rule_mtr_read.data;
	uint32_t tc_mask = req->table_rule_mtr_read.tc_mask;
	int clear = req->table_rule_mtr_read.clear;
	struct rte_table_action *a = p->table_data[table_id].a;

	rsp->status = rte_table_action_meter_read(a,
		data,
		tc_mask,
		&rsp->table_rule_mtr_read.stats,
		clear);

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_table_dscp_table_update(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	uint32_t table_id = req->id;
	uint64_t dscp_mask = req->table_dscp_table_update.dscp_mask;
	struct rte_table_action_dscp_table *dscp_table =
		&req->table_dscp_table_update.dscp_table;
	struct rte_table_action *a = p->table_data[table_id].a;

	rsp->status = rte_table_action_dscp_table_update(a,
		dscp_mask,
		dscp_table);

	return rsp;
}

static struct pipeline_msg_rsp *
pipeline_msg_handle_table_rule_ttl_read(struct pipeline_data *p,
	struct pipeline_msg_req *req)
{
	struct pipeline_msg_rsp *rsp = (struct pipeline_msg_rsp *)req;
	uint32_t table_id = req->id;
	void *data = req->table_rule_ttl_read.data;
	int clear = req->table_rule_ttl_read.clear;
	struct rte_table_action *a = p->table_data[table_id].a;

	rsp->status = rte_table_action_ttl_read(a,
		data,
		&rsp->table_rule_ttl_read.stats,
		clear);

	return rsp;
}

static void
pipeline_msg_handle(struct pipeline_data *p)
{
	for ( ; ; ) {
		struct pipeline_msg_req *req;
		struct pipeline_msg_rsp *rsp;

		req = pipeline_msg_recv(p->msgq_req);
		if (req == NULL)
			break;

		switch (req->type) {
		case PIPELINE_REQ_PORT_IN_STATS_READ:
			rsp = pipeline_msg_handle_port_in_stats_read(p, req);
			break;

		case PIPELINE_REQ_PORT_IN_ENABLE:
			rsp = pipeline_msg_handle_port_in_enable(p, req);
			break;

		case PIPELINE_REQ_PORT_IN_DISABLE:
			rsp = pipeline_msg_handle_port_in_disable(p, req);
			break;

		case PIPELINE_REQ_PORT_OUT_STATS_READ:
			rsp = pipeline_msg_handle_port_out_stats_read(p, req);
			break;

		case PIPELINE_REQ_TABLE_STATS_READ:
			rsp = pipeline_msg_handle_table_stats_read(p, req);
			break;

		case PIPELINE_REQ_TABLE_RULE_ADD:
			rsp = pipeline_msg_handle_table_rule_add(p, req);
			break;

		case PIPELINE_REQ_TABLE_RULE_ADD_DEFAULT:
			rsp = pipeline_msg_handle_table_rule_add_default(p,	req);
			break;

		case PIPELINE_REQ_TABLE_RULE_ADD_BULK:
			rsp = pipeline_msg_handle_table_rule_add_bulk(p, req);
			break;

		case PIPELINE_REQ_TABLE_RULE_DELETE:
			rsp = pipeline_msg_handle_table_rule_delete(p, req);
			break;

		case PIPELINE_REQ_TABLE_RULE_DELETE_DEFAULT:
			rsp = pipeline_msg_handle_table_rule_delete_default(p, req);
			break;

		case PIPELINE_REQ_TABLE_RULE_STATS_READ:
			rsp = pipeline_msg_handle_table_rule_stats_read(p, req);
			break;

		case PIPELINE_REQ_TABLE_MTR_PROFILE_ADD:
			rsp = pipeline_msg_handle_table_mtr_profile_add(p, req);
			break;

		case PIPELINE_REQ_TABLE_MTR_PROFILE_DELETE:
			rsp = pipeline_msg_handle_table_mtr_profile_delete(p, req);
			break;

		case PIPELINE_REQ_TABLE_RULE_MTR_READ:
			rsp = pipeline_msg_handle_table_rule_mtr_read(p, req);
			break;

		case PIPELINE_REQ_TABLE_DSCP_TABLE_UPDATE:
			rsp = pipeline_msg_handle_table_dscp_table_update(p, req);
			break;

		case PIPELINE_REQ_TABLE_RULE_TTL_READ:
			rsp = pipeline_msg_handle_table_rule_ttl_read(p, req);
			break;

		default:
			rsp = (struct pipeline_msg_rsp *)req;
			rsp->status = -1;
		}

		pipeline_msg_send(p->msgq_rsp, rsp);
	}
}

/**
 * Data plane threads: main
 */
static int32_t
rte_pmd_softnic_run_internal(void *arg)
{
	struct rte_eth_dev *dev = arg;
	struct pmd_internals *softnic;
	struct softnic_thread_data *t;
	uint32_t thread_id, j;

	softnic = dev->data->dev_private;
	thread_id = rte_lcore_id();
	t = &softnic->thread_data[thread_id];
	t->iter++;

	/* Data Plane */
	for (j = 0; j < t->n_pipelines; j++)
		rte_pipeline_run(t->p[j]);

	/* Control Plane */
	if ((t->iter & 0xFLLU) == 0) {
		uint64_t time = rte_get_tsc_cycles();
		uint64_t time_next_min = UINT64_MAX;

		if (time < t->time_next_min)
			return 0;

		/* Pipeline message queues */
		for (j = 0; j < t->n_pipelines; j++) {
			struct pipeline_data *p =
				&t->pipeline_data[j];
			uint64_t time_next = p->time_next;

			if (time_next <= time) {
				pipeline_msg_handle(p);
				rte_pipeline_flush(p->p);
				time_next = time + p->timer_period;
				p->time_next = time_next;
			}

			if (time_next < time_next_min)
				time_next_min = time_next;
		}

		/* Thread message queues */
		{
			uint64_t time_next = t->time_next;

			if (time_next <= time) {
				thread_msg_handle(t);
				time_next = time + t->timer_period;
				t->time_next = time_next;
			}

			if (time_next < time_next_min)
				time_next_min = time_next;
		}

		t->time_next_min = time_next_min;
	}

	return 0;
}

int
rte_pmd_softnic_run(uint16_t port_id)
{
	struct rte_eth_dev *dev = &rte_eth_devices[port_id];

#ifdef RTE_LIBRTE_ETHDEV_DEBUG
	RTE_ETH_VALID_PORTID_OR_ERR_RET(port_id, 0);
#endif

	return (int)rte_pmd_softnic_run_internal(dev);
}
