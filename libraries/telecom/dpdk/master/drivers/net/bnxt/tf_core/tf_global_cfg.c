/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2019-2020 Broadcom
 * All rights reserved.
 */

#include <rte_common.h>

#include "tf_global_cfg.h"
#include "tf_common.h"
#include "tf_util.h"
#include "tf_msg.h"
#include "tfp.h"

struct tf;
/**
 * Global Cfg DBs.
 */
static void *global_cfg_db[TF_DIR_MAX];

/**
 * Init flag, set on bind and cleared on unbind
 */
static uint8_t init;

/**
 * Get HCAPI type parameters for a single element
 */
struct tf_global_cfg_get_hcapi_parms {
	/**
	 * [in] Global Cfg DB Handle
	 */
	void *global_cfg_db;
	/**
	 * [in] DB Index, indicates which DB entry to perform the
	 * action on.
	 */
	uint16_t db_index;
	/**
	 * [out] Pointer to the hcapi type for the specified db_index
	 */
	uint16_t *hcapi_type;
};

/**
 * Check global_cfg_type and return hwrm type.
 *
 * [in] global_cfg_type
 *   Global Cfg type
 *
 * [out] hwrm_type
 *   HWRM device data type
 *
 * Returns:
 *    0          - Success
 *   -EOPNOTSUPP - Type not supported
 */
static int
tf_global_cfg_get_hcapi_type(struct tf_global_cfg_get_hcapi_parms *parms)
{
	struct tf_global_cfg_cfg *global_cfg;
	enum tf_global_cfg_cfg_type cfg_type;

	global_cfg = (struct tf_global_cfg_cfg *)parms->global_cfg_db;
	cfg_type = global_cfg[parms->db_index].cfg_type;

	if (cfg_type != TF_GLOBAL_CFG_CFG_HCAPI)
		return -ENOTSUP;

	*parms->hcapi_type = global_cfg[parms->db_index].hcapi_type;

	return 0;
}

int
tf_global_cfg_bind(struct tf *tfp __rte_unused,
		   struct tf_global_cfg_cfg_parms *parms)
{
	TF_CHECK_PARMS2(tfp, parms);

	if (init) {
		TFP_DRV_LOG(ERR,
			    "Global Cfg DB already initialized\n");
		return -EILWAL;
	}

	global_cfg_db[TF_DIR_RX] = parms->cfg;
	global_cfg_db[TF_DIR_TX] = parms->cfg;

	init = 1;

	TFP_DRV_LOG(INFO,
		    "Global Cfg - initialized\n");

	return 0;
}

int
tf_global_cfg_unbind(struct tf *tfp __rte_unused)
{
	/* Bail if nothing has been initialized */
	if (!init) {
		TFP_DRV_LOG(INFO,
			    "No Global Cfg DBs created\n");
		return 0;
	}

	global_cfg_db[TF_DIR_RX] = NULL;
	global_cfg_db[TF_DIR_TX] = NULL;
	init = 0;

	return 0;
}

int
tf_global_cfg_set(struct tf *tfp,
		  struct tf_global_cfg_parms *parms)
{
	int rc;
	struct tf_global_cfg_get_hcapi_parms hparms;
	uint16_t hcapi_type;

	TF_CHECK_PARMS3(tfp, parms, parms->config);

	if (!init) {
		TFP_DRV_LOG(ERR,
			    "%s: No Global Cfg DBs created\n",
			    tf_dir_2_str(parms->dir));
		return -EILWAL;
	}

	/* Colwert TF type to HCAPI type */
	hparms.global_cfg_db = global_cfg_db[parms->dir];
	hparms.db_index = parms->type;
	hparms.hcapi_type = &hcapi_type;
	rc = tf_global_cfg_get_hcapi_type(&hparms);
	if (rc) {
		TFP_DRV_LOG(ERR,
			    "%s, Failed type lookup, type:%d, rc:%s\n",
			    tf_dir_2_str(parms->dir),
			    parms->type,
			    strerror(-rc));
		return rc;
	}

	rc = tf_msg_set_global_cfg(tfp, parms);
	if (rc) {
		TFP_DRV_LOG(ERR,
			    "%s, Set failed, type:%d, rc:%s\n",
			    tf_dir_2_str(parms->dir),
			    parms->type,
			    strerror(-rc));
	}

	return 0;
}

int
tf_global_cfg_get(struct tf *tfp,
		  struct tf_global_cfg_parms *parms)

{
	int rc;
	struct tf_global_cfg_get_hcapi_parms hparms;
	uint16_t hcapi_type;

	TF_CHECK_PARMS3(tfp, parms, parms->config);

	if (!init) {
		TFP_DRV_LOG(ERR,
			    "%s: No Global Cfg DBs created\n",
			    tf_dir_2_str(parms->dir));
		return -EILWAL;
	}

	hparms.global_cfg_db = global_cfg_db[parms->dir];
	hparms.db_index = parms->type;
	hparms.hcapi_type = &hcapi_type;
	rc = tf_global_cfg_get_hcapi_type(&hparms);
	if (rc) {
		TFP_DRV_LOG(ERR,
			    "%s, Failed type lookup, type:%d, rc:%s\n",
			    tf_dir_2_str(parms->dir),
			    parms->type,
			    strerror(-rc));
		return rc;
	}

	/* Get the entry */
	rc = tf_msg_get_global_cfg(tfp, parms);
	if (rc) {
		TFP_DRV_LOG(ERR,
			    "%s, Get failed, type:%d, rc:%s\n",
			    tf_dir_2_str(parms->dir),
			    parms->type,
			    strerror(-rc));
	}

	return 0;
}
