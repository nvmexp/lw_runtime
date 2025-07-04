/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2019 Mellanox Technologies, Ltd
 */
#include <unistd.h>
#include <net/if.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <netinet/in.h>

#include <rte_malloc.h>
#include <rte_log.h>
#include <rte_errno.h>
#include <rte_pci.h>
#include <rte_string_fns.h>

#include <mlx5_glue.h>
#include <mlx5_common.h>
#include <mlx5_common_pci.h>
#include <mlx5_devx_cmds.h>
#include <mlx5_prm.h>
#include <mlx5_nl.h>

#include "mlx5_vdpa_utils.h"
#include "mlx5_vdpa.h"


#define MLX5_VDPA_DEFAULT_FEATURES ((1ULL << VHOST_USER_F_PROTOCOL_FEATURES) | \
			    (1ULL << VIRTIO_F_ANY_LAYOUT) | \
			    (1ULL << VIRTIO_NET_F_MQ) | \
			    (1ULL << VIRTIO_NET_F_GUEST_ANNOUNCE) | \
			    (1ULL << VIRTIO_F_ORDER_PLATFORM) | \
			    (1ULL << VHOST_F_LOG_ALL) | \
			    (1ULL << VIRTIO_NET_F_MTU))

#define MLX5_VDPA_PROTOCOL_FEATURES \
			    ((1ULL << VHOST_USER_PROTOCOL_F_SLAVE_REQ) | \
			     (1ULL << VHOST_USER_PROTOCOL_F_SLAVE_SEND_FD) | \
			     (1ULL << VHOST_USER_PROTOCOL_F_HOST_NOTIFIER) | \
			     (1ULL << VHOST_USER_PROTOCOL_F_LOG_SHMFD) | \
			     (1ULL << VHOST_USER_PROTOCOL_F_MQ) | \
			     (1ULL << VHOST_USER_PROTOCOL_F_NET_MTU) | \
			     (1ULL << VHOST_USER_PROTOCOL_F_STATUS))

#define MLX5_VDPA_MAX_RETRIES 20
#define MLX5_VDPA_USEC 1000
#define MLX5_VDPA_DEFAULT_NO_TRAFFIC_TIME_S 2LLU

TAILQ_HEAD(mlx5_vdpa_privs, mlx5_vdpa_priv) priv_list =
					      TAILQ_HEAD_INITIALIZER(priv_list);
static pthread_mutex_t priv_list_lock = PTHREAD_MUTEX_INITIALIZER;

static struct mlx5_vdpa_priv *
mlx5_vdpa_find_priv_resource_by_vdev(struct rte_vdpa_device *vdev)
{
	struct mlx5_vdpa_priv *priv;
	int found = 0;

	pthread_mutex_lock(&priv_list_lock);
	TAILQ_FOREACH(priv, &priv_list, next) {
		if (vdev == priv->vdev) {
			found = 1;
			break;
		}
	}
	pthread_mutex_unlock(&priv_list_lock);
	if (!found) {
		DRV_LOG(ERR, "Invalid vDPA device: %s.", vdev->device->name);
		rte_errno = EILWAL;
		return NULL;
	}
	return priv;
}

static int
mlx5_vdpa_get_queue_num(struct rte_vdpa_device *vdev, uint32_t *queue_num)
{
	struct mlx5_vdpa_priv *priv =
		mlx5_vdpa_find_priv_resource_by_vdev(vdev);

	if (priv == NULL) {
		DRV_LOG(ERR, "Invalid vDPA device: %s.", vdev->device->name);
		return -1;
	}
	*queue_num = priv->caps.max_num_virtio_queues;
	return 0;
}

static int
mlx5_vdpa_get_vdpa_features(struct rte_vdpa_device *vdev, uint64_t *features)
{
	struct mlx5_vdpa_priv *priv =
		mlx5_vdpa_find_priv_resource_by_vdev(vdev);

	if (priv == NULL) {
		DRV_LOG(ERR, "Invalid vDPA device: %s.", vdev->device->name);
		return -1;
	}
	*features = MLX5_VDPA_DEFAULT_FEATURES;
	if (priv->caps.virtio_queue_type & (1 << MLX5_VIRTQ_TYPE_PACKED))
		*features |= (1ULL << VIRTIO_F_RING_PACKED);
	if (priv->caps.tso_ipv4)
		*features |= (1ULL << VIRTIO_NET_F_HOST_TSO4);
	if (priv->caps.tso_ipv6)
		*features |= (1ULL << VIRTIO_NET_F_HOST_TSO6);
	if (priv->caps.tx_csum)
		*features |= (1ULL << VIRTIO_NET_F_CSUM);
	if (priv->caps.rx_csum)
		*features |= (1ULL << VIRTIO_NET_F_GUEST_CSUM);
	if (priv->caps.virtio_version_1_0)
		*features |= (1ULL << VIRTIO_F_VERSION_1);
	return 0;
}

static int
mlx5_vdpa_get_protocol_features(struct rte_vdpa_device *vdev,
		uint64_t *features)
{
	struct mlx5_vdpa_priv *priv =
		mlx5_vdpa_find_priv_resource_by_vdev(vdev);

	if (priv == NULL) {
		DRV_LOG(ERR, "Invalid vDPA device: %s.", vdev->device->name);
		return -1;
	}
	*features = MLX5_VDPA_PROTOCOL_FEATURES;
	return 0;
}

static int
mlx5_vdpa_set_vring_state(int vid, int vring, int state)
{
	struct rte_vdpa_device *vdev = rte_vhost_get_vdpa_device(vid);
	struct mlx5_vdpa_priv *priv =
		mlx5_vdpa_find_priv_resource_by_vdev(vdev);
	int ret;

	if (priv == NULL) {
		DRV_LOG(ERR, "Invalid vDPA device: %s.", vdev->device->name);
		return -EILWAL;
	}
	if (vring >= (int)priv->caps.max_num_virtio_queues * 2) {
		DRV_LOG(ERR, "Too big vring id: %d.", vring);
		return -E2BIG;
	}
	pthread_mutex_lock(&priv->vq_config_lock);
	ret = mlx5_vdpa_virtq_enable(priv, vring, state);
	pthread_mutex_unlock(&priv->vq_config_lock);
	return ret;
}

static int
mlx5_vdpa_features_set(int vid)
{
	struct rte_vdpa_device *vdev = rte_vhost_get_vdpa_device(vid);
	struct mlx5_vdpa_priv *priv =
		mlx5_vdpa_find_priv_resource_by_vdev(vdev);
	uint64_t log_base, log_size;
	uint64_t features;
	int ret;

	if (priv == NULL) {
		DRV_LOG(ERR, "Invalid vDPA device: %s.", vdev->device->name);
		return -EILWAL;
	}
	ret = rte_vhost_get_negotiated_features(vid, &features);
	if (ret) {
		DRV_LOG(ERR, "Failed to get negotiated features.");
		return ret;
	}
	if (RTE_VHOST_NEED_LOG(features)) {
		ret = rte_vhost_get_log_base(vid, &log_base, &log_size);
		if (ret) {
			DRV_LOG(ERR, "Failed to get log base.");
			return ret;
		}
		ret = mlx5_vdpa_dirty_bitmap_set(priv, log_base, log_size);
		if (ret) {
			DRV_LOG(ERR, "Failed to set dirty bitmap.");
			return ret;
		}
		DRV_LOG(INFO, "mlx5 vdpa: enabling dirty logging...");
		ret = mlx5_vdpa_logging_enable(priv, 1);
		if (ret) {
			DRV_LOG(ERR, "Failed t enable dirty logging.");
			return ret;
		}
	}
	return 0;
}

static int
mlx5_vdpa_pd_create(struct mlx5_vdpa_priv *priv)
{
#ifdef HAVE_IBV_FLOW_DV_SUPPORT
	priv->pd = mlx5_glue->alloc_pd(priv->ctx);
	if (priv->pd == NULL) {
		DRV_LOG(ERR, "Failed to allocate PD.");
		return errno ? -errno : -ENOMEM;
	}
	struct mlx5dv_obj obj;
	struct mlx5dv_pd pd_info;
	int ret = 0;

	obj.pd.in = priv->pd;
	obj.pd.out = &pd_info;
	ret = mlx5_glue->dv_init_obj(&obj, MLX5DV_OBJ_PD);
	if (ret) {
		DRV_LOG(ERR, "Fail to get PD object info.");
		mlx5_glue->dealloc_pd(priv->pd);
		priv->pd = NULL;
		return -errno;
	}
	priv->pdn = pd_info.pdn;
	return 0;
#else
	(void)priv;
	DRV_LOG(ERR, "Cannot get pdn - no DV support.");
	return -ENOTSUP;
#endif /* HAVE_IBV_FLOW_DV_SUPPORT */
}

static int
mlx5_vdpa_mtu_set(struct mlx5_vdpa_priv *priv)
{
	struct ifreq request;
	uint16_t vhost_mtu = 0;
	uint16_t kern_mtu = 0;
	int ret = rte_vhost_get_mtu(priv->vid, &vhost_mtu);
	int sock;
	int retries = MLX5_VDPA_MAX_RETRIES;

	if (ret) {
		DRV_LOG(DEBUG, "Cannot get vhost MTU - %d.", ret);
		return ret;
	}
	if (!vhost_mtu) {
		DRV_LOG(DEBUG, "Vhost MTU is 0.");
		return ret;
	}
	ret = mlx5_get_ifname_sysfs(priv->ctx->device->ibdev_path,
				    request.ifr_name);
	if (ret) {
		DRV_LOG(DEBUG, "Cannot get kernel IF name - %d.", ret);
		return ret;
	}
	sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_IP);
	if (sock == -1) {
		DRV_LOG(DEBUG, "Cannot open IF socket.");
		return sock;
	}
	while (retries--) {
		ret = ioctl(sock, SIOCGIFMTU, &request);
		if (ret == -1)
			break;
		kern_mtu = request.ifr_mtu;
		DRV_LOG(DEBUG, "MTU: current %d requested %d.", (int)kern_mtu,
			(int)vhost_mtu);
		if (kern_mtu == vhost_mtu)
			break;
		request.ifr_mtu = vhost_mtu;
		ret = ioctl(sock, SIOCSIFMTU, &request);
		if (ret == -1)
			break;
		request.ifr_mtu = 0;
		usleep(MLX5_VDPA_USEC);
	}
	close(sock);
	return kern_mtu == vhost_mtu ? 0 : -1;
}

static int
mlx5_vdpa_dev_close(int vid)
{
	struct rte_vdpa_device *vdev = rte_vhost_get_vdpa_device(vid);
	struct mlx5_vdpa_priv *priv =
		mlx5_vdpa_find_priv_resource_by_vdev(vdev);
	int ret = 0;

	if (priv == NULL) {
		DRV_LOG(ERR, "Invalid vDPA device: %s.", vdev->device->name);
		return -1;
	}
	if (priv->configured)
		ret |= mlx5_vdpa_lm_log(priv);
	mlx5_vdpa_err_event_unset(priv);
	mlx5_vdpa_cqe_event_unset(priv);
	mlx5_vdpa_steer_unset(priv);
	mlx5_vdpa_virtqs_release(priv);
	mlx5_vdpa_event_qp_global_release(priv);
	mlx5_vdpa_mem_dereg(priv);
	if (priv->pd) {
		claim_zero(mlx5_glue->dealloc_pd(priv->pd));
		priv->pd = NULL;
	}
	priv->configured = 0;
	priv->vid = 0;
	DRV_LOG(INFO, "vDPA device %d was closed.", vid);
	return ret;
}

static int
mlx5_vdpa_dev_config(int vid)
{
	struct rte_vdpa_device *vdev = rte_vhost_get_vdpa_device(vid);
	struct mlx5_vdpa_priv *priv =
		mlx5_vdpa_find_priv_resource_by_vdev(vdev);

	if (priv == NULL) {
		DRV_LOG(ERR, "Invalid vDPA device: %s.", vdev->device->name);
		return -EILWAL;
	}
	if (priv->configured && mlx5_vdpa_dev_close(vid)) {
		DRV_LOG(ERR, "Failed to reconfigure vid %d.", vid);
		return -1;
	}
	priv->vid = vid;
	if (mlx5_vdpa_mtu_set(priv))
		DRV_LOG(WARNING, "MTU cannot be set on device %s.",
				vdev->device->name);
	if (mlx5_vdpa_pd_create(priv) || mlx5_vdpa_mem_register(priv) ||
	    mlx5_vdpa_err_event_setup(priv) ||
	    mlx5_vdpa_virtqs_prepare(priv) || mlx5_vdpa_steer_setup(priv) ||
	    mlx5_vdpa_cqe_event_setup(priv)) {
		mlx5_vdpa_dev_close(vid);
		return -1;
	}
	priv->configured = 1;
	DRV_LOG(INFO, "vDPA device %d was configured.", vid);
	return 0;
}

static int
mlx5_vdpa_get_device_fd(int vid)
{
	struct rte_vdpa_device *vdev = rte_vhost_get_vdpa_device(vid);
	struct mlx5_vdpa_priv *priv =
		mlx5_vdpa_find_priv_resource_by_vdev(vdev);

	if (priv == NULL) {
		DRV_LOG(ERR, "Invalid vDPA device: %s.", vdev->device->name);
		return -EILWAL;
	}
	return priv->ctx->cmd_fd;
}

static int
mlx5_vdpa_get_notify_area(int vid, int qid, uint64_t *offset, uint64_t *size)
{
	struct rte_vdpa_device *vdev = rte_vhost_get_vdpa_device(vid);
	struct mlx5_vdpa_priv *priv =
		mlx5_vdpa_find_priv_resource_by_vdev(vdev);

	RTE_SET_USED(qid);
	if (priv == NULL) {
		DRV_LOG(ERR, "Invalid vDPA device: %s.", vdev->device->name);
		return -EILWAL;
	}
	if (!priv->var) {
		DRV_LOG(ERR, "VAR was not created for device %s, is the device"
			" configured?.", vdev->device->name);
		return -EILWAL;
	}
	*offset = priv->var->mmap_off;
	*size = priv->var->length;
	return 0;
}

static int
mlx5_vdpa_get_stats_names(struct rte_vdpa_device *vdev,
		struct rte_vdpa_stat_name *stats_names,
		unsigned int size)
{
	static const char *mlx5_vdpa_stats_names[MLX5_VDPA_STATS_MAX] = {
		"received_descriptors",
		"completed_descriptors",
		"bad descriptor errors",
		"exceed max chain",
		"invalid buffer",
		"completion errors",
	};
	struct mlx5_vdpa_priv *priv =
		mlx5_vdpa_find_priv_resource_by_vdev(vdev);
	unsigned int i;

	if (priv == NULL) {
		DRV_LOG(ERR, "Invalid device: %s.", vdev->device->name);
		return -ENODEV;
	}
	if (!stats_names)
		return MLX5_VDPA_STATS_MAX;
	size = RTE_MIN(size, (unsigned int)MLX5_VDPA_STATS_MAX);
	for (i = 0; i < size; ++i)
		strlcpy(stats_names[i].name, mlx5_vdpa_stats_names[i],
			RTE_VDPA_STATS_NAME_SIZE);
	return size;
}

static int
mlx5_vdpa_get_stats(struct rte_vdpa_device *vdev, int qid,
		struct rte_vdpa_stat *stats, unsigned int n)
{
	struct mlx5_vdpa_priv *priv =
		mlx5_vdpa_find_priv_resource_by_vdev(vdev);

	if (priv == NULL) {
		DRV_LOG(ERR, "Invalid device: %s.", vdev->device->name);
		return -ENODEV;
	}
	if (!priv->configured) {
		DRV_LOG(ERR, "Device %s was not configured.",
				vdev->device->name);
		return -ENODATA;
	}
	if (qid >= (int)priv->nr_virtqs) {
		DRV_LOG(ERR, "Too big vring id: %d for device %s.", qid,
				vdev->device->name);
		return -E2BIG;
	}
	if (!priv->caps.queue_counters_valid) {
		DRV_LOG(ERR, "Virtq statistics is not supported for device %s.",
			vdev->device->name);
		return -ENOTSUP;
	}
	return mlx5_vdpa_virtq_stats_get(priv, qid, stats, n);
}

static int
mlx5_vdpa_reset_stats(struct rte_vdpa_device *vdev, int qid)
{
	struct mlx5_vdpa_priv *priv =
		mlx5_vdpa_find_priv_resource_by_vdev(vdev);

	if (priv == NULL) {
		DRV_LOG(ERR, "Invalid device: %s.", vdev->device->name);
		return -ENODEV;
	}
	if (!priv->configured) {
		DRV_LOG(ERR, "Device %s was not configured.",
				vdev->device->name);
		return -ENODATA;
	}
	if (qid >= (int)priv->nr_virtqs) {
		DRV_LOG(ERR, "Too big vring id: %d for device %s.", qid,
				vdev->device->name);
		return -E2BIG;
	}
	if (!priv->caps.queue_counters_valid) {
		DRV_LOG(ERR, "Virtq statistics is not supported for device %s.",
			vdev->device->name);
		return -ENOTSUP;
	}
	return mlx5_vdpa_virtq_stats_reset(priv, qid);
}

static struct rte_vdpa_dev_ops mlx5_vdpa_ops = {
	.get_queue_num = mlx5_vdpa_get_queue_num,
	.get_features = mlx5_vdpa_get_vdpa_features,
	.get_protocol_features = mlx5_vdpa_get_protocol_features,
	.dev_conf = mlx5_vdpa_dev_config,
	.dev_close = mlx5_vdpa_dev_close,
	.set_vring_state = mlx5_vdpa_set_vring_state,
	.set_features = mlx5_vdpa_features_set,
	.migration_done = NULL,
	.get_vfio_group_fd = NULL,
	.get_vfio_device_fd = mlx5_vdpa_get_device_fd,
	.get_notify_area = mlx5_vdpa_get_notify_area,
	.get_stats_names = mlx5_vdpa_get_stats_names,
	.get_stats = mlx5_vdpa_get_stats,
	.reset_stats = mlx5_vdpa_reset_stats,
};

static struct ibv_device *
mlx5_vdpa_get_ib_device_match(struct rte_pci_addr *addr)
{
	int n;
	struct ibv_device **ibv_list = mlx5_glue->get_device_list(&n);
	struct ibv_device *ibv_match = NULL;

	if (!ibv_list) {
		rte_errno = ENOSYS;
		return NULL;
	}
	while (n-- > 0) {
		struct rte_pci_addr pci_addr;

		DRV_LOG(DEBUG, "Checking device \"%s\"..", ibv_list[n]->name);
		if (mlx5_dev_to_pci_addr(ibv_list[n]->ibdev_path, &pci_addr))
			continue;
		if (rte_pci_addr_cmp(addr, &pci_addr))
			continue;
		ibv_match = ibv_list[n];
		break;
	}
	if (!ibv_match)
		rte_errno = ENOENT;
	mlx5_glue->free_device_list(ibv_list);
	return ibv_match;
}

/* Try to disable ROCE by Netlink\Devlink. */
static int
mlx5_vdpa_nl_roce_disable(const char *addr)
{
	int nlsk_fd = mlx5_nl_init(NETLINK_GENERIC);
	int devlink_id;
	int enable;
	int ret;

	if (nlsk_fd < 0)
		return nlsk_fd;
	devlink_id = mlx5_nl_devlink_family_id_get(nlsk_fd);
	if (devlink_id < 0) {
		ret = devlink_id;
		DRV_LOG(DEBUG, "Failed to get devlink id for ROCE operations by"
			" Netlink.");
		goto close;
	}
	ret = mlx5_nl_enable_roce_get(nlsk_fd, devlink_id, addr, &enable);
	if (ret) {
		DRV_LOG(DEBUG, "Failed to get ROCE enable by Netlink: %d.",
			ret);
		goto close;
	} else if (!enable) {
		DRV_LOG(INFO, "ROCE has already disabled(Netlink).");
		goto close;
	}
	ret = mlx5_nl_enable_roce_set(nlsk_fd, devlink_id, addr, 0);
	if (ret)
		DRV_LOG(DEBUG, "Failed to disable ROCE by Netlink: %d.", ret);
	else
		DRV_LOG(INFO, "ROCE is disabled by Netlink successfully.");
close:
	close(nlsk_fd);
	return ret;
}

/* Try to disable ROCE by sysfs. */
static int
mlx5_vdpa_sys_roce_disable(const char *addr)
{
	FILE *file_o;
	int enable;
	int ret;

	MKSTR(file_p, "/sys/bus/pci/devices/%s/roce_enable", addr);
	file_o = fopen(file_p, "rb");
	if (!file_o) {
		rte_errno = ENOTSUP;
		return -ENOTSUP;
	}
	ret = fscanf(file_o, "%d", &enable);
	if (ret != 1) {
		rte_errno = EILWAL;
		ret = EILWAL;
		goto close;
	} else if (!enable) {
		ret = 0;
		DRV_LOG(INFO, "ROCE has already disabled(sysfs).");
		goto close;
	}
	fclose(file_o);
	file_o = fopen(file_p, "wb");
	if (!file_o) {
		rte_errno = ENOTSUP;
		return -ENOTSUP;
	}
	fprintf(file_o, "0\n");
	ret = 0;
close:
	if (ret)
		DRV_LOG(DEBUG, "Failed to disable ROCE by sysfs: %d.", ret);
	else
		DRV_LOG(INFO, "ROCE is disabled by sysfs successfully.");
	fclose(file_o);
	return ret;
}

static int
mlx5_vdpa_roce_disable(struct rte_pci_addr *addr, struct ibv_device **ibv)
{
	char addr_name[64] = {0};

	rte_pci_device_name(addr, addr_name, sizeof(addr_name));
	/* Firstly try to disable ROCE by Netlink and fallback to sysfs. */
	if (mlx5_vdpa_nl_roce_disable(addr_name) == 0 ||
	    mlx5_vdpa_sys_roce_disable(addr_name) == 0) {
		/*
		 * Succeed to disable ROCE, wait for the IB device to appear
		 * again after reload.
		 */
		int r;
		struct ibv_device *ibv_new;

		for (r = MLX5_VDPA_MAX_RETRIES; r; r--) {
			ibv_new = mlx5_vdpa_get_ib_device_match(addr);
			if (ibv_new) {
				*ibv = ibv_new;
				return 0;
			}
			usleep(MLX5_VDPA_USEC);
		}
		DRV_LOG(ERR, "Cannot much device %s after ROCE disable, "
			"retries exceed %d", addr_name, MLX5_VDPA_MAX_RETRIES);
		rte_errno = EAGAIN;
	}
	return -rte_errno;
}

static int
mlx5_vdpa_args_check_handler(const char *key, const char *val, void *opaque)
{
	struct mlx5_vdpa_priv *priv = opaque;
	unsigned long tmp;

	if (strcmp(key, "class") == 0)
		return 0;
	errno = 0;
	tmp = strtoul(val, NULL, 0);
	if (errno) {
		DRV_LOG(WARNING, "%s: \"%s\" is an invalid integer.", key, val);
		return -errno;
	}
	if (strcmp(key, "event_mode") == 0) {
		if (tmp <= MLX5_VDPA_EVENT_MODE_ONLY_INTERRUPT)
			priv->event_mode = (int)tmp;
		else
			DRV_LOG(WARNING, "Invalid event_mode %s.", val);
	} else if (strcmp(key, "event_us") == 0) {
		priv->event_us = (uint32_t)tmp;
	} else if (strcmp(key, "no_traffic_time") == 0) {
		priv->no_traffic_time_s = (uint32_t)tmp;
	} else {
		DRV_LOG(WARNING, "Invalid key %s.", key);
	}
	return 0;
}

static void
mlx5_vdpa_config_get(struct rte_devargs *devargs, struct mlx5_vdpa_priv *priv)
{
	struct rte_kvargs *kvlist;

	priv->event_mode = MLX5_VDPA_EVENT_MODE_DYNAMIC_TIMER;
	priv->event_us = 0;
	priv->no_traffic_time_s = MLX5_VDPA_DEFAULT_NO_TRAFFIC_TIME_S;
	if (devargs == NULL)
		return;
	kvlist = rte_kvargs_parse(devargs->args, NULL);
	if (kvlist == NULL)
		return;
	rte_kvargs_process(kvlist, NULL, mlx5_vdpa_args_check_handler, priv);
	rte_kvargs_free(kvlist);
	if (!priv->event_us) {
		if (priv->event_mode == MLX5_VDPA_EVENT_MODE_DYNAMIC_TIMER)
			priv->event_us = MLX5_VDPA_DEFAULT_TIMER_STEP_US;
		else if (priv->event_mode == MLX5_VDPA_EVENT_MODE_FIXED_TIMER)
			priv->event_us = MLX5_VDPA_DEFAULT_TIMER_DELAY_US;
	}
	DRV_LOG(DEBUG, "event mode is %d.", priv->event_mode);
	DRV_LOG(DEBUG, "event_us is %u us.", priv->event_us);
	DRV_LOG(DEBUG, "no traffic time is %u s.", priv->no_traffic_time_s);
}

/**
 * DPDK callback to register a mlx5 PCI device.
 *
 * This function spawns vdpa device out of a given PCI device.
 *
 * @param[in] pci_drv
 *   PCI driver structure (mlx5_vpda_driver).
 * @param[in] pci_dev
 *   PCI device information.
 *
 * @return
 *   0 on success, 1 to skip this driver, a negative errno value otherwise
 *   and rte_errno is set.
 */
static int
mlx5_vdpa_pci_probe(struct rte_pci_driver *pci_drv __rte_unused,
		    struct rte_pci_device *pci_dev __rte_unused)
{
	struct ibv_device *ibv;
	struct mlx5_vdpa_priv *priv = NULL;
	struct ibv_context *ctx = NULL;
	struct mlx5_hca_attr attr;
	int ret;

	ibv = mlx5_vdpa_get_ib_device_match(&pci_dev->addr);
	if (!ibv) {
		DRV_LOG(ERR, "No matching IB device for PCI slot "
			PCI_PRI_FMT ".", pci_dev->addr.domain,
			pci_dev->addr.bus, pci_dev->addr.devid,
			pci_dev->addr.function);
		return -rte_errno;
	} else {
		DRV_LOG(INFO, "PCI information matches for device \"%s\".",
			ibv->name);
	}
	if (mlx5_vdpa_roce_disable(&pci_dev->addr, &ibv) != 0) {
		DRV_LOG(WARNING, "Failed to disable ROCE for \"%s\".",
			ibv->name);
		return -rte_errno;
	}
	ctx = mlx5_glue->dv_open_device(ibv);
	if (!ctx) {
		DRV_LOG(ERR, "Failed to open IB device \"%s\".", ibv->name);
		rte_errno = ENODEV;
		return -rte_errno;
	}
	ret = mlx5_devx_cmd_query_hca_attr(ctx, &attr);
	if (ret) {
		DRV_LOG(ERR, "Unable to read HCA capabilities.");
		rte_errno = ENOTSUP;
		goto error;
	} else if (!attr.vdpa.valid || !attr.vdpa.max_num_virtio_queues) {
		DRV_LOG(ERR, "Not enough capabilities to support vdpa, maybe "
			"old FW/OFED version?");
		rte_errno = ENOTSUP;
		goto error;
	}
	if (!attr.vdpa.queue_counters_valid)
		DRV_LOG(DEBUG, "No capability to support virtq statistics.");
	priv = rte_zmalloc("mlx5 vDPA device private", sizeof(*priv) +
			   sizeof(struct mlx5_vdpa_virtq) *
			   attr.vdpa.max_num_virtio_queues * 2,
			   RTE_CACHE_LINE_SIZE);
	if (!priv) {
		DRV_LOG(ERR, "Failed to allocate private memory.");
		rte_errno = ENOMEM;
		goto error;
	}
	priv->caps = attr.vdpa;
	priv->log_max_rqt_size = attr.log_max_rqt_size;
	priv->num_lag_ports = attr.num_lag_ports;
	if (attr.num_lag_ports == 0)
		priv->num_lag_ports = 1;
	priv->ctx = ctx;
	priv->pci_dev = pci_dev;
	priv->var = mlx5_glue->dv_alloc_var(ctx, 0);
	if (!priv->var) {
		DRV_LOG(ERR, "Failed to allocate VAR %u.\n", errno);
		goto error;
	}
	priv->vdev = rte_vdpa_register_device(&pci_dev->device,
			&mlx5_vdpa_ops);
	if (priv->vdev == NULL) {
		DRV_LOG(ERR, "Failed to register vDPA device.");
		rte_errno = rte_errno ? rte_errno : EILWAL;
		goto error;
	}
	mlx5_vdpa_config_get(pci_dev->device.devargs, priv);
	SLIST_INIT(&priv->mr_list);
	pthread_mutex_init(&priv->vq_config_lock, NULL);
	pthread_mutex_lock(&priv_list_lock);
	TAILQ_INSERT_TAIL(&priv_list, priv, next);
	pthread_mutex_unlock(&priv_list_lock);
	return 0;

error:
	if (priv) {
		if (priv->var)
			mlx5_glue->dv_free_var(priv->var);
		rte_free(priv);
	}
	if (ctx)
		mlx5_glue->close_device(ctx);
	return -rte_errno;
}

/**
 * DPDK callback to remove a PCI device.
 *
 * This function removes all vDPA devices belong to a given PCI device.
 *
 * @param[in] pci_dev
 *   Pointer to the PCI device.
 *
 * @return
 *   0 on success, the function cannot fail.
 */
static int
mlx5_vdpa_pci_remove(struct rte_pci_device *pci_dev)
{
	struct mlx5_vdpa_priv *priv = NULL;
	int found = 0;

	pthread_mutex_lock(&priv_list_lock);
	TAILQ_FOREACH(priv, &priv_list, next) {
		if (!rte_pci_addr_cmp(&priv->pci_dev->addr, &pci_dev->addr)) {
			found = 1;
			break;
		}
	}
	if (found)
		TAILQ_REMOVE(&priv_list, priv, next);
	pthread_mutex_unlock(&priv_list_lock);
	if (found) {
		if (priv->configured)
			mlx5_vdpa_dev_close(priv->vid);
		if (priv->var) {
			mlx5_glue->dv_free_var(priv->var);
			priv->var = NULL;
		}
		mlx5_glue->close_device(priv->ctx);
		pthread_mutex_destroy(&priv->vq_config_lock);
		rte_free(priv);
	}
	return 0;
}

static const struct rte_pci_id mlx5_vdpa_pci_id_map[] = {
	{
		RTE_PCI_DEVICE(PCI_VENDOR_ID_MELLANOX,
				PCI_DEVICE_ID_MELLANOX_CONNECTX6)
	},
	{
		RTE_PCI_DEVICE(PCI_VENDOR_ID_MELLANOX,
				PCI_DEVICE_ID_MELLANOX_CONNECTX6VF)
	},
	{
		RTE_PCI_DEVICE(PCI_VENDOR_ID_MELLANOX,
				PCI_DEVICE_ID_MELLANOX_CONNECTX6DX)
	},
	{
		RTE_PCI_DEVICE(PCI_VENDOR_ID_MELLANOX,
				PCI_DEVICE_ID_MELLANOX_CONNECTXVF)
	},
	{
		RTE_PCI_DEVICE(PCI_VENDOR_ID_MELLANOX,
				PCI_DEVICE_ID_MELLANOX_CONNECTX6DXBF)
	},
	{
		RTE_PCI_DEVICE(PCI_VENDOR_ID_MELLANOX,
				PCI_DEVICE_ID_MELLANOX_CONNECTX7)
	},
	{
		RTE_PCI_DEVICE(PCI_VENDOR_ID_MELLANOX,
				PCI_DEVICE_ID_MELLANOX_CONNECTX7BF)
	},
	{
		.vendor_id = 0
	}
};

static struct mlx5_pci_driver mlx5_vdpa_driver = {
	.driver_class = MLX5_CLASS_VDPA,
	.pci_driver = {
		.driver = {
			.name = "mlx5_vdpa",
		},
		.id_table = mlx5_vdpa_pci_id_map,
		.probe = mlx5_vdpa_pci_probe,
		.remove = mlx5_vdpa_pci_remove,
		.drv_flags = 0,
	},
};

RTE_LOG_REGISTER(mlx5_vdpa_logtype, pmd.vdpa.mlx5, NOTICE)

/**
 * Driver initialization routine.
 */
RTE_INIT(rte_mlx5_vdpa_init)
{
	mlx5_common_init();
	if (mlx5_glue)
		mlx5_pci_driver_register(&mlx5_vdpa_driver);
}

RTE_PMD_EXPORT_NAME(net_mlx5_vdpa, __COUNTER__);
RTE_PMD_REGISTER_PCI_TABLE(net_mlx5_vdpa, mlx5_vdpa_pci_id_map);
RTE_PMD_REGISTER_KMOD_DEP(net_mlx5_vdpa, "* ib_uverbs & mlx5_core & mlx5_ib");
