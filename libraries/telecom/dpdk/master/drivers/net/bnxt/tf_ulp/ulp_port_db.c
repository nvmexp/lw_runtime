/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2014-2020 Broadcom
 * All rights reserved.
 */

#include <rte_malloc.h>
#include "bnxt.h"
#include "bnxt_vnic.h"
#include "bnxt_tf_common.h"
#include "ulp_port_db.h"

static uint32_t
ulp_port_db_allocate_ifindex(struct bnxt_ulp_port_db *port_db)
{
	uint32_t idx = 1;

	while (idx < port_db->ulp_intf_list_size &&
	       port_db->ulp_intf_list[idx].type != BNXT_ULP_INTF_TYPE_ILWALID)
		idx++;

	if (idx >= port_db->ulp_intf_list_size) {
		BNXT_TF_DBG(ERR, "Port DB interface list is full\n");
		return 0;
	}
	return idx;
}

/*
 * Initialize the port database. Memory is allocated in this
 * call and assigned to the port database.
 *
 * ulp_ctxt [in] Ptr to ulp context
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t	ulp_port_db_init(struct bnxt_ulp_context *ulp_ctxt, uint8_t port_cnt)
{
	struct bnxt_ulp_port_db *port_db;

	port_db = rte_zmalloc("bnxt_ulp_port_db",
			      sizeof(struct bnxt_ulp_port_db), 0);
	if (!port_db) {
		BNXT_TF_DBG(ERR,
			    "Failed to allocate memory for port db\n");
		return -ENOMEM;
	}

	/* Attach the port database to the ulp context. */
	bnxt_ulp_cntxt_ptr2_port_db_set(ulp_ctxt, port_db);

	/* index 0 is not being used hence add 1 to size */
	port_db->ulp_intf_list_size = BNXT_PORT_DB_MAX_INTF_LIST + 1;
	/* Allocate the port tables */
	port_db->ulp_intf_list = rte_zmalloc("bnxt_ulp_port_db_intf_list",
					     port_db->ulp_intf_list_size *
					     sizeof(struct ulp_interface_info),
					     0);
	if (!port_db->ulp_intf_list) {
		BNXT_TF_DBG(ERR,
			    "Failed to allocate mem for port interface list\n");
		goto error_free;
	}

	/* Allocate the phy port list */
	port_db->phy_port_list = rte_zmalloc("bnxt_ulp_phy_port_list",
					     port_cnt *
					     sizeof(struct ulp_phy_port_info),
					     0);
	if (!port_db->phy_port_list) {
		BNXT_TF_DBG(ERR,
			    "Failed to allocate mem for phy port list\n");
		goto error_free;
	}
	port_db->phy_port_cnt = port_cnt;
	return 0;

error_free:
	ulp_port_db_deinit(ulp_ctxt);
	return -ENOMEM;
}

/*
 * Deinitialize the port database. Memory is deallocated in
 * this call.
 *
 * ulp_ctxt [in] Ptr to ulp context
 *
 * Returns 0 on success.
 */
int32_t	ulp_port_db_deinit(struct bnxt_ulp_context *ulp_ctxt)
{
	struct bnxt_ulp_port_db *port_db;

	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}

	/* Detach the flow database from the ulp context. */
	bnxt_ulp_cntxt_ptr2_port_db_set(ulp_ctxt, NULL);

	/* Free up all the memory. */
	rte_free(port_db->phy_port_list);
	rte_free(port_db->ulp_intf_list);
	rte_free(port_db);
	return 0;
}

/*
 * Update the port database.This api is called when the port
 * details are available during the startup.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * bp [in]. ptr to the device function.
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t	ulp_port_db_dev_port_intf_update(struct bnxt_ulp_context *ulp_ctxt,
					 struct rte_eth_dev *eth_dev)
{
	uint32_t port_id = eth_dev->data->port_id;
	struct ulp_phy_port_info *port_data;
	struct bnxt_ulp_port_db *port_db;
	struct ulp_interface_info *intf;
	struct ulp_func_if_info *func;
	uint32_t ifindex;
	int32_t rc;

	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}

	rc = ulp_port_db_dev_port_to_ulp_index(ulp_ctxt, port_id, &ifindex);
	if (rc == -ENOENT) {
		/* port not found, allocate one */
		ifindex = ulp_port_db_allocate_ifindex(port_db);
		if (!ifindex)
			return -ENOMEM;
		port_db->dev_port_list[port_id] = ifindex;
	} else if (rc == -EILWAL) {
		return -EILWAL;
	}

	/* update the interface details */
	intf = &port_db->ulp_intf_list[ifindex];

	intf->type = bnxt_get_interface_type(port_id);
	intf->drv_func_id = bnxt_get_fw_func_id(port_id,
						BNXT_ULP_INTF_TYPE_ILWALID);

	func = &port_db->ulp_func_id_tbl[intf->drv_func_id];
	if (!func->func_valid) {
		func->func_svif = bnxt_get_svif(port_id, true,
						BNXT_ULP_INTF_TYPE_ILWALID);
		func->func_spif = bnxt_get_phy_port_id(port_id);
		func->func_parif =
			bnxt_get_parif(port_id, BNXT_ULP_INTF_TYPE_ILWALID);
		func->func_vnic =
			bnxt_get_vnic_id(port_id, BNXT_ULP_INTF_TYPE_ILWALID);
		func->phy_port_id = bnxt_get_phy_port_id(port_id);
		func->func_valid = true;
		func->ifindex = ifindex;
	}

	if (intf->type == BNXT_ULP_INTF_TYPE_VF_REP) {
		intf->vf_func_id =
			bnxt_get_fw_func_id(port_id, BNXT_ULP_INTF_TYPE_VF_REP);

		func = &port_db->ulp_func_id_tbl[intf->vf_func_id];
		func->func_svif =
			bnxt_get_svif(port_id, true, BNXT_ULP_INTF_TYPE_VF_REP);
		func->func_spif =
			bnxt_get_phy_port_id(port_id);
		func->func_parif =
			bnxt_get_parif(port_id, BNXT_ULP_INTF_TYPE_ILWALID);
		func->func_vnic =
			bnxt_get_vnic_id(port_id, BNXT_ULP_INTF_TYPE_VF_REP);
		func->phy_port_id = bnxt_get_phy_port_id(port_id);
		func->ifindex = ifindex;
	}

	port_data = &port_db->phy_port_list[func->phy_port_id];
	if (!port_data->port_valid) {
		port_data->port_svif =
			bnxt_get_svif(port_id, false,
				      BNXT_ULP_INTF_TYPE_ILWALID);
		port_data->port_spif = bnxt_get_phy_port_id(port_id);
		port_data->port_parif =
			bnxt_get_parif(port_id, BNXT_ULP_INTF_TYPE_ILWALID);
		port_data->port_vport = bnxt_get_vport(port_id);
		port_data->port_valid = true;
	}

	return 0;
}

/*
 * Api to get the ulp ifindex for a given device port.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * port_id [in].device port id
 * ifindex [out] ulp ifindex
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t
ulp_port_db_dev_port_to_ulp_index(struct bnxt_ulp_context *ulp_ctxt,
				  uint32_t port_id,
				  uint32_t *ifindex)
{
	struct bnxt_ulp_port_db *port_db;

	*ifindex = 0;
	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db || port_id >= RTE_MAX_ETHPORTS) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}
	if (!port_db->dev_port_list[port_id])
		return -ENOENT;

	*ifindex = port_db->dev_port_list[port_id];
	return 0;
}

/*
 * Api to get the function id for a given ulp ifindex.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * ifindex [in] ulp ifindex
 * func_id [out] the function id of the given ifindex.
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t
ulp_port_db_function_id_get(struct bnxt_ulp_context *ulp_ctxt,
			    uint32_t ifindex,
			    uint32_t fid_type,
			    uint16_t *func_id)
{
	struct bnxt_ulp_port_db *port_db;

	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db || ifindex >= port_db->ulp_intf_list_size || !ifindex) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}

	if (fid_type == BNXT_ULP_DRV_FUNC_FID)
		*func_id =  port_db->ulp_intf_list[ifindex].drv_func_id;
	else
		*func_id =  port_db->ulp_intf_list[ifindex].vf_func_id;

	return 0;
}

/*
 * Api to get the svif for a given ulp ifindex.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * ifindex [in] ulp ifindex
 * svif_type [in] the svif type of the given ifindex.
 * svif [out] the svif of the given ifindex.
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t
ulp_port_db_svif_get(struct bnxt_ulp_context *ulp_ctxt,
		     uint32_t ifindex,
		     uint32_t svif_type,
		     uint16_t *svif)
{
	struct bnxt_ulp_port_db *port_db;
	uint16_t phy_port_id, func_id;

	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db || ifindex >= port_db->ulp_intf_list_size || !ifindex) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}

	if (svif_type == BNXT_ULP_DRV_FUNC_SVIF) {
		func_id = port_db->ulp_intf_list[ifindex].drv_func_id;
		*svif = port_db->ulp_func_id_tbl[func_id].func_svif;
	} else if (svif_type == BNXT_ULP_VF_FUNC_SVIF) {
		func_id = port_db->ulp_intf_list[ifindex].vf_func_id;
		*svif = port_db->ulp_func_id_tbl[func_id].func_svif;
	} else {
		func_id = port_db->ulp_intf_list[ifindex].drv_func_id;
		phy_port_id = port_db->ulp_func_id_tbl[func_id].phy_port_id;
		*svif = port_db->phy_port_list[phy_port_id].port_svif;
	}

	return 0;
}

/*
 * Api to get the spif for a given ulp ifindex.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * ifindex [in] ulp ifindex
 * spif_type [in] the spif type of the given ifindex.
 * spif [out] the spif of the given ifindex.
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t
ulp_port_db_spif_get(struct bnxt_ulp_context *ulp_ctxt,
		     uint32_t ifindex,
		     uint32_t spif_type,
		     uint16_t *spif)
{
	struct bnxt_ulp_port_db *port_db;
	uint16_t phy_port_id, func_id;

	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db || ifindex >= port_db->ulp_intf_list_size || !ifindex) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}

	if (spif_type == BNXT_ULP_DRV_FUNC_SPIF) {
		func_id = port_db->ulp_intf_list[ifindex].drv_func_id;
		*spif = port_db->ulp_func_id_tbl[func_id].func_spif;
	} else if (spif_type == BNXT_ULP_VF_FUNC_SPIF) {
		func_id = port_db->ulp_intf_list[ifindex].vf_func_id;
		*spif = port_db->ulp_func_id_tbl[func_id].func_spif;
	} else {
		func_id = port_db->ulp_intf_list[ifindex].drv_func_id;
		phy_port_id = port_db->ulp_func_id_tbl[func_id].phy_port_id;
		*spif = port_db->phy_port_list[phy_port_id].port_spif;
	}

	return 0;
}

/*
 * Api to get the parif for a given ulp ifindex.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * ifindex [in] ulp ifindex
 * parif_type [in] the parif type of the given ifindex.
 * parif [out] the parif of the given ifindex.
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t
ulp_port_db_parif_get(struct bnxt_ulp_context *ulp_ctxt,
		     uint32_t ifindex,
		     uint32_t parif_type,
		     uint16_t *parif)
{
	struct bnxt_ulp_port_db *port_db;
	uint16_t phy_port_id, func_id;

	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db || ifindex >= port_db->ulp_intf_list_size || !ifindex) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}
	if (parif_type == BNXT_ULP_DRV_FUNC_PARIF) {
		func_id = port_db->ulp_intf_list[ifindex].drv_func_id;
		*parif = port_db->ulp_func_id_tbl[func_id].func_parif;
	} else if (parif_type == BNXT_ULP_VF_FUNC_PARIF) {
		func_id = port_db->ulp_intf_list[ifindex].vf_func_id;
		*parif = port_db->ulp_func_id_tbl[func_id].func_parif;
	} else {
		func_id = port_db->ulp_intf_list[ifindex].drv_func_id;
		phy_port_id = port_db->ulp_func_id_tbl[func_id].phy_port_id;
		*parif = port_db->phy_port_list[phy_port_id].port_parif;
	}
	/* Parif needs to be reset to a free partition */
	*parif += BNXT_ULP_FREE_PARIF_BASE;

	return 0;
}

/*
 * Api to get the vnic id for a given ulp ifindex.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * ifindex [in] ulp ifindex
 * vnic [out] the vnic of the given ifindex.
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t
ulp_port_db_default_vnic_get(struct bnxt_ulp_context *ulp_ctxt,
			     uint32_t ifindex,
			     uint32_t vnic_type,
			     uint16_t *vnic)
{
	struct bnxt_ulp_port_db *port_db;
	uint16_t func_id;

	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db || ifindex >= port_db->ulp_intf_list_size || !ifindex) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}

	if (vnic_type == BNXT_ULP_DRV_FUNC_VNIC) {
		func_id = port_db->ulp_intf_list[ifindex].drv_func_id;
		*vnic = port_db->ulp_func_id_tbl[func_id].func_vnic;
	} else {
		func_id = port_db->ulp_intf_list[ifindex].vf_func_id;
		*vnic = port_db->ulp_func_id_tbl[func_id].func_vnic;
	}

	return 0;
}

/*
 * Api to get the vport id for a given ulp ifindex.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * ifindex [in] ulp ifindex
 * vport [out] the port of the given ifindex.
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t
ulp_port_db_vport_get(struct bnxt_ulp_context *ulp_ctxt,
		      uint32_t ifindex, uint16_t *vport)
{
	struct bnxt_ulp_port_db *port_db;
	uint16_t phy_port_id, func_id;

	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db || ifindex >= port_db->ulp_intf_list_size || !ifindex) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}

	func_id = port_db->ulp_intf_list[ifindex].drv_func_id;
	phy_port_id = port_db->ulp_func_id_tbl[func_id].phy_port_id;
	*vport = port_db->phy_port_list[phy_port_id].port_vport;
	return 0;
}

/*
 * Api to get the vport for a given physical port.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * phy_port [in] physical port index
 * out_port [out] the port of the given physical index
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t
ulp_port_db_phy_port_vport_get(struct bnxt_ulp_context *ulp_ctxt,
			       uint32_t phy_port,
			       uint16_t *out_port)
{
	struct bnxt_ulp_port_db *port_db;

	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db || phy_port >= port_db->phy_port_cnt) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}
	*out_port = port_db->phy_port_list[phy_port].port_vport;
	return 0;
}

/*
 * Api to get the svif for a given physical port.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * phy_port [in] physical port index
 * svif [out] the svif of the given physical index
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t
ulp_port_db_phy_port_svif_get(struct bnxt_ulp_context *ulp_ctxt,
			      uint32_t phy_port,
			      uint16_t *svif)
{
	struct bnxt_ulp_port_db *port_db;

	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db || phy_port >= port_db->phy_port_cnt) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}
	*svif = port_db->phy_port_list[phy_port].port_svif;
	return 0;
}

/*
 * Api to get the port type for a given ulp ifindex.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * ifindex [in] ulp ifindex
 *
 * Returns port type.
 */
enum bnxt_ulp_intf_type
ulp_port_db_port_type_get(struct bnxt_ulp_context *ulp_ctxt,
			  uint32_t ifindex)
{
	struct bnxt_ulp_port_db *port_db;

	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db || ifindex >= port_db->ulp_intf_list_size || !ifindex) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return BNXT_ULP_INTF_TYPE_ILWALID;
	}
	return port_db->ulp_intf_list[ifindex].type;
}

/*
 * Api to get the ulp ifindex for a given function id.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * func_id [in].device func id
 * ifindex [out] ulp ifindex
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t
ulp_port_db_dev_func_id_to_ulp_index(struct bnxt_ulp_context *ulp_ctxt,
				     uint32_t func_id, uint32_t *ifindex)
{
	struct bnxt_ulp_port_db *port_db;

	*ifindex = 0;
	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db || func_id >= BNXT_PORT_DB_MAX_FUNC) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}
	if (!port_db->ulp_func_id_tbl[func_id].func_valid)
		return -ENOENT;

	*ifindex = port_db->ulp_func_id_tbl[func_id].ifindex;
	return 0;
}

/*
 * Api to get the function id for a given port id.
 *
 * ulp_ctxt [in] Ptr to ulp context
 * port_id [in] dpdk port id
 * func_id [out] the function id of the given ifindex.
 *
 * Returns 0 on success or negative number on failure.
 */
int32_t
ulp_port_db_port_func_id_get(struct bnxt_ulp_context *ulp_ctxt,
			     uint16_t port_id, uint16_t *func_id)
{
	struct bnxt_ulp_port_db *port_db;
	uint32_t ifindex;

	port_db = bnxt_ulp_cntxt_ptr2_port_db_get(ulp_ctxt);
	if (!port_db || port_id >= RTE_MAX_ETHPORTS) {
		BNXT_TF_DBG(ERR, "Invalid Arguments\n");
		return -EILWAL;
	}
	ifindex = port_db->dev_port_list[port_id];
	if (!ifindex)
		return -ENOENT;

	switch (port_db->ulp_intf_list[ifindex].type) {
	case BNXT_ULP_INTF_TYPE_TRUSTED_VF:
	case BNXT_ULP_INTF_TYPE_PF:
		*func_id =  port_db->ulp_intf_list[ifindex].drv_func_id;
		break;
	case BNXT_ULP_INTF_TYPE_VF:
	case BNXT_ULP_INTF_TYPE_VF_REP:
		*func_id =  port_db->ulp_intf_list[ifindex].vf_func_id;
		break;
	default:
		*func_id = 0;
		break;
	}
	return 0;
}
