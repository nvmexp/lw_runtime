/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2017 NXP
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/queue.h>

#include <rte_string_fns.h>
#include <rte_byteorder.h>
#include <rte_log.h>
#include <rte_debug.h>
#include <rte_dev.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_memzone.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_atomic.h>
#include <rte_branch_prediction.h>
#include <rte_common.h>
#include <rte_malloc.h>
#include <rte_errno.h>
#include <rte_telemetry.h>

#include "rte_rawdev.h"
#include "rte_rawdev_pmd.h"

static struct rte_rawdev rte_rawdevices[RTE_RAWDEV_MAX_DEVS];

struct rte_rawdev *rte_rawdevs = rte_rawdevices;

static struct rte_rawdev_global rawdev_globals = {
	.nb_devs		= 0
};

/* Raw device, northbound API implementation */
uint8_t
rte_rawdev_count(void)
{
	return rawdev_globals.nb_devs;
}

uint16_t
rte_rawdev_get_dev_id(const char *name)
{
	uint16_t i;

	if (!name)
		return -EILWAL;

	for (i = 0; i < rawdev_globals.nb_devs; i++)
		if ((strcmp(rte_rawdevices[i].name, name)
				== 0) &&
				(rte_rawdevices[i].attached ==
						RTE_RAWDEV_ATTACHED))
			return i;
	return -ENODEV;
}

int
rte_rawdev_socket_id(uint16_t dev_id)
{
	struct rte_rawdev *dev;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];

	return dev->socket_id;
}

int
rte_rawdev_info_get(uint16_t dev_id, struct rte_rawdev_info *dev_info,
		size_t dev_private_size)
{
	struct rte_rawdev *rawdev;
	int ret = 0;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	RTE_FUNC_PTR_OR_ERR_RET(dev_info, -EILWAL);

	rawdev = &rte_rawdevs[dev_id];

	if (dev_info->dev_private != NULL) {
		RTE_FUNC_PTR_OR_ERR_RET(*rawdev->dev_ops->dev_info_get, -ENOTSUP);
		ret = (*rawdev->dev_ops->dev_info_get)(rawdev,
				dev_info->dev_private,
				dev_private_size);
	}

	dev_info->driver_name = rawdev->driver_name;
	dev_info->device = rawdev->device;
	dev_info->socket_id = rawdev->socket_id;

	return ret;
}

int
rte_rawdev_configure(uint16_t dev_id, struct rte_rawdev_info *dev_conf,
		size_t dev_private_size)
{
	struct rte_rawdev *dev;
	int diag;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	RTE_FUNC_PTR_OR_ERR_RET(dev_conf, -EILWAL);

	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->dev_configure, -ENOTSUP);

	if (dev->started) {
		RTE_RDEV_ERR(
		   "device %d must be stopped to allow configuration", dev_id);
		return -EBUSY;
	}

	/* Configure the device */
	diag = (*dev->dev_ops->dev_configure)(dev, dev_conf->dev_private,
			dev_private_size);
	if (diag != 0)
		RTE_RDEV_ERR("dev%d dev_configure = %d", dev_id, diag);
	else
		dev->attached = 1;

	return diag;
}

int
rte_rawdev_queue_conf_get(uint16_t dev_id,
			  uint16_t queue_id,
			  rte_rawdev_obj_t queue_conf,
			  size_t queue_conf_size)
{
	struct rte_rawdev *dev;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->queue_def_conf, -ENOTSUP);
	return (*dev->dev_ops->queue_def_conf)(dev, queue_id, queue_conf,
			queue_conf_size);
}

int
rte_rawdev_queue_setup(uint16_t dev_id,
		       uint16_t queue_id,
		       rte_rawdev_obj_t queue_conf,
		       size_t queue_conf_size)
{
	struct rte_rawdev *dev;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->queue_setup, -ENOTSUP);
	return (*dev->dev_ops->queue_setup)(dev, queue_id, queue_conf,
			queue_conf_size);
}

int
rte_rawdev_queue_release(uint16_t dev_id, uint16_t queue_id)
{
	struct rte_rawdev *dev;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->queue_release, -ENOTSUP);
	return (*dev->dev_ops->queue_release)(dev, queue_id);
}

uint16_t
rte_rawdev_queue_count(uint16_t dev_id)
{
	struct rte_rawdev *dev;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->queue_count, -ENOTSUP);
	return (*dev->dev_ops->queue_count)(dev);
}

int
rte_rawdev_get_attr(uint16_t dev_id,
		    const char *attr_name,
		    uint64_t *attr_value)
{
	struct rte_rawdev *dev;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->attr_get, -ENOTSUP);
	return (*dev->dev_ops->attr_get)(dev, attr_name, attr_value);
}

int
rte_rawdev_set_attr(uint16_t dev_id,
		    const char *attr_name,
		    const uint64_t attr_value)
{
	struct rte_rawdev *dev;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->attr_set, -ENOTSUP);
	return (*dev->dev_ops->attr_set)(dev, attr_name, attr_value);
}

int
rte_rawdev_enqueue_buffers(uint16_t dev_id,
			   struct rte_rawdev_buf **buffers,
			   unsigned int count,
			   rte_rawdev_obj_t context)
{
	struct rte_rawdev *dev;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->enqueue_bufs, -ENOTSUP);
	return (*dev->dev_ops->enqueue_bufs)(dev, buffers, count, context);
}

int
rte_rawdev_dequeue_buffers(uint16_t dev_id,
			   struct rte_rawdev_buf **buffers,
			   unsigned int count,
			   rte_rawdev_obj_t context)
{
	struct rte_rawdev *dev;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->dequeue_bufs, -ENOTSUP);
	return (*dev->dev_ops->dequeue_bufs)(dev, buffers, count, context);
}

int
rte_rawdev_dump(uint16_t dev_id, FILE *f)
{
	struct rte_rawdev *dev;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->dump, -ENOTSUP);
	return (*dev->dev_ops->dump)(dev, f);
}

static int
xstats_get_count(uint16_t dev_id)
{
	struct rte_rawdev *dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->xstats_get_names, -ENOTSUP);
	return (*dev->dev_ops->xstats_get_names)(dev, NULL, 0);
}

int
rte_rawdev_xstats_names_get(uint16_t dev_id,
		struct rte_rawdev_xstats_name *xstats_names,
		unsigned int size)
{
	const struct rte_rawdev *dev;
	int cnt_expected_entries;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -ENODEV);

	cnt_expected_entries = xstats_get_count(dev_id);

	if (xstats_names == NULL || cnt_expected_entries < 0 ||
	    (int)size < cnt_expected_entries || size <= 0)
		return cnt_expected_entries;

	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->xstats_get_names, -ENOTSUP);
	return (*dev->dev_ops->xstats_get_names)(dev, xstats_names, size);
}

/* retrieve rawdev extended statistics */
int
rte_rawdev_xstats_get(uint16_t dev_id,
		      const unsigned int ids[],
		      uint64_t values[],
		      unsigned int n)
{
	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -ENODEV);
	const struct rte_rawdev *dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->xstats_get, -ENOTSUP);
	return (*dev->dev_ops->xstats_get)(dev, ids, values, n);
}

uint64_t
rte_rawdev_xstats_by_name_get(uint16_t dev_id,
			      const char *name,
			      unsigned int *id)
{
	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, 0);
	const struct rte_rawdev *dev = &rte_rawdevs[dev_id];
	unsigned int temp = -1;

	if (id != NULL)
		*id = (unsigned int)-1;
	else
		id = &temp; /* driver never gets a NULL value */

	/* implemented by driver */
	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->xstats_get_by_name, -ENOTSUP);
	return (*dev->dev_ops->xstats_get_by_name)(dev, name, id);
}

int
rte_rawdev_xstats_reset(uint16_t dev_id,
			const uint32_t ids[], uint32_t nb_ids)
{
	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	struct rte_rawdev *dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->xstats_reset, -ENOTSUP);
	return (*dev->dev_ops->xstats_reset)(dev, ids, nb_ids);
}

int
rte_rawdev_firmware_status_get(uint16_t dev_id, rte_rawdev_obj_t status_info)
{
	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	struct rte_rawdev *dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->firmware_status_get, -ENOTSUP);
	return (*dev->dev_ops->firmware_status_get)(dev, status_info);
}

int
rte_rawdev_firmware_version_get(uint16_t dev_id, rte_rawdev_obj_t version_info)
{
	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	struct rte_rawdev *dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->firmware_version_get, -ENOTSUP);
	return (*dev->dev_ops->firmware_version_get)(dev, version_info);
}

int
rte_rawdev_firmware_load(uint16_t dev_id, rte_rawdev_obj_t firmware_image)
{
	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	struct rte_rawdev *dev = &rte_rawdevs[dev_id];

	if (!firmware_image)
		return -EILWAL;

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->firmware_load, -ENOTSUP);
	return (*dev->dev_ops->firmware_load)(dev, firmware_image);
}

int
rte_rawdev_firmware_unload(uint16_t dev_id)
{
	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	struct rte_rawdev *dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->firmware_load, -ENOTSUP);
	return (*dev->dev_ops->firmware_unload)(dev);
}

int
rte_rawdev_selftest(uint16_t dev_id)
{
	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	struct rte_rawdev *dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->dev_selftest, -ENOTSUP);
	return (*dev->dev_ops->dev_selftest)(dev_id);
}

int
rte_rawdev_start(uint16_t dev_id)
{
	struct rte_rawdev *dev;
	int diag;

	RTE_RDEV_DEBUG("Start dev_id=%" PRIu8, dev_id);

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];
	if (dev->started != 0) {
		RTE_RDEV_ERR("Device with dev_id=%" PRIu8 "already started",
			     dev_id);
		return 0;
	}

	if (dev->dev_ops->dev_start == NULL)
		goto mark_started;

	diag = (*dev->dev_ops->dev_start)(dev);
	if (diag != 0)
		return diag;

mark_started:
	dev->started = 1;
	return 0;
}

void
rte_rawdev_stop(uint16_t dev_id)
{
	struct rte_rawdev *dev;

	RTE_RDEV_DEBUG("Stop dev_id=%" PRIu8, dev_id);

	RTE_RAWDEV_VALID_DEVID_OR_RET(dev_id);
	dev = &rte_rawdevs[dev_id];

	if (dev->started == 0) {
		RTE_RDEV_ERR("Device with dev_id=%" PRIu8 "already stopped",
			dev_id);
		return;
	}

	if (dev->dev_ops->dev_stop == NULL)
		goto mark_stopped;

	(*dev->dev_ops->dev_stop)(dev);

mark_stopped:
	dev->started = 0;
}

int
rte_rawdev_close(uint16_t dev_id)
{
	struct rte_rawdev *dev;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->dev_close, -ENOTSUP);
	/* Device must be stopped before it can be closed */
	if (dev->started == 1) {
		RTE_RDEV_ERR("Device %u must be stopped before closing",
			     dev_id);
		return -EBUSY;
	}

	return (*dev->dev_ops->dev_close)(dev);
}

int
rte_rawdev_reset(uint16_t dev_id)
{
	struct rte_rawdev *dev;

	RTE_RAWDEV_VALID_DEVID_OR_ERR_RET(dev_id, -EILWAL);
	dev = &rte_rawdevs[dev_id];

	RTE_FUNC_PTR_OR_ERR_RET(*dev->dev_ops->dev_reset, -ENOTSUP);
	/* Reset is not dependent on state of the device */
	return (*dev->dev_ops->dev_reset)(dev);
}

static inline uint8_t
rte_rawdev_find_free_device_index(void)
{
	uint16_t dev_id;

	for (dev_id = 0; dev_id < RTE_RAWDEV_MAX_DEVS; dev_id++) {
		if (rte_rawdevs[dev_id].attached ==
				RTE_RAWDEV_DETACHED)
			return dev_id;
	}

	return RTE_RAWDEV_MAX_DEVS;
}

struct rte_rawdev *
rte_rawdev_pmd_allocate(const char *name, size_t dev_priv_size, int socket_id)
{
	struct rte_rawdev *rawdev;
	uint16_t dev_id;

	if (rte_rawdev_pmd_get_named_dev(name) != NULL) {
		RTE_RDEV_ERR("Event device with name %s already allocated!",
			     name);
		return NULL;
	}

	dev_id = rte_rawdev_find_free_device_index();
	if (dev_id == RTE_RAWDEV_MAX_DEVS) {
		RTE_RDEV_ERR("Reached maximum number of raw devices");
		return NULL;
	}

	rawdev = &rte_rawdevs[dev_id];

	if (dev_priv_size > 0) {
		rawdev->dev_private = rte_zmalloc_socket("rawdev private",
				     dev_priv_size,
				     RTE_CACHE_LINE_SIZE,
				     socket_id);
		if (!rawdev->dev_private) {
			RTE_RDEV_ERR("Unable to allocate memory for rawdev");
			return NULL;
		}
	}

	rawdev->dev_id = dev_id;
	rawdev->socket_id = socket_id;
	rawdev->started = 0;
	strlcpy(rawdev->name, name, RTE_RAWDEV_NAME_MAX_LEN);

	rawdev->attached = RTE_RAWDEV_ATTACHED;
	rawdev_globals.nb_devs++;

	return rawdev;
}

int
rte_rawdev_pmd_release(struct rte_rawdev *rawdev)
{
	int ret;

	if (rawdev == NULL)
		return -EILWAL;

	ret = rte_rawdev_close(rawdev->dev_id);
	if (ret < 0)
		return ret;

	rawdev->attached = RTE_RAWDEV_DETACHED;
	rawdev_globals.nb_devs--;

	rawdev->dev_id = 0;
	rawdev->socket_id = 0;
	rawdev->dev_ops = NULL;
	if (rawdev->dev_private) {
		rte_free(rawdev->dev_private);
		rawdev->dev_private = NULL;
	}

	return 0;
}

static int
handle_dev_list(const char *cmd __rte_unused,
		const char *params __rte_unused,
		struct rte_tel_data *d)
{
	int i;

	rte_tel_data_start_array(d, RTE_TEL_INT_VAL);
	for (i = 0; i < rawdev_globals.nb_devs; i++)
		if (rte_rawdevices[i].attached == RTE_RAWDEV_ATTACHED)
			rte_tel_data_add_array_int(d, i);
	return 0;
}

static int
handle_dev_xstats(const char *cmd __rte_unused,
		const char *params,
		struct rte_tel_data *d)
{
	uint64_t *rawdev_xstats;
	struct rte_rawdev_xstats_name *xstat_names;
	int dev_id, num_xstats, i, ret;
	unsigned int *ids;
	char *end_param;

	if (params == NULL || strlen(params) == 0 || !isdigit(*params))
		return -1;

	dev_id = strtoul(params, &end_param, 0);
	if (*end_param != '\0')
		RTE_RDEV_LOG(NOTICE,
			"Extra parameters passed to rawdev telemetry command, ignoring");
	if (!rte_rawdev_pmd_is_valid_dev(dev_id))
		return -1;

	num_xstats = xstats_get_count(dev_id);
	if (num_xstats < 0)
		return -1;

	/* use one malloc for names, stats and ids */
	rawdev_xstats = malloc((sizeof(uint64_t) +
			sizeof(struct rte_rawdev_xstats_name) +
			sizeof(unsigned int)) * num_xstats);
	if (rawdev_xstats == NULL)
		return -1;
	xstat_names = (void *)&rawdev_xstats[num_xstats];
	ids = (void *)&xstat_names[num_xstats];

	ret = rte_rawdev_xstats_names_get(dev_id, xstat_names, num_xstats);
	if (ret < 0 || ret > num_xstats) {
		free(rawdev_xstats);
		return -1;
	}

	for (i = 0; i < num_xstats; i++)
		ids[i] = i;

	ret = rte_rawdev_xstats_get(dev_id, ids, rawdev_xstats, num_xstats);
	if (ret < 0 || ret > num_xstats) {
		free(rawdev_xstats);
		return -1;
	}

	rte_tel_data_start_dict(d);
	for (i = 0; i < num_xstats; i++)
		rte_tel_data_add_dict_u64(d, xstat_names[i].name,
				rawdev_xstats[i]);

	free(rawdev_xstats);
	return 0;
}

RTE_LOG_REGISTER(librawdev_logtype, lib.rawdev, INFO);

RTE_INIT(librawdev_init_telemetry)
{
	rte_telemetry_register_cmd("/rawdev/list", handle_dev_list,
			"Returns list of available rawdev ports. Takes no parameters");
	rte_telemetry_register_cmd("/rawdev/xstats", handle_dev_xstats,
			"Returns the xstats for a rawdev port. Parameters: int port_id");
}
