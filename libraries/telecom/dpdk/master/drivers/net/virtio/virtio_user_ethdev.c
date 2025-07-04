/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2016 Intel Corporation
 */

#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/major.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/socket.h>

#include <rte_malloc.h>
#include <rte_kvargs.h>
#include <rte_ethdev_vdev.h>
#include <rte_bus_vdev.h>
#include <rte_alarm.h>
#include <rte_cycles.h>

#include "virtio_ethdev.h"
#include "virtio_logs.h"
#include "virtio_pci.h"
#include "virtqueue.h"
#include "virtio_rxtx.h"
#include "virtio_user/virtio_user_dev.h"
#include "virtio_user/vhost.h"

#define virtio_user_get_dev(hw) \
	((struct virtio_user_dev *)(hw)->virtio_user_dev)

static void
virtio_user_reset_queues_packed(struct rte_eth_dev *dev)
{
	struct virtio_hw *hw = dev->data->dev_private;
	struct virtnet_rx *rxvq;
	struct virtnet_tx *txvq;
	uint16_t i;

	/* Add lock to avoid queue contention. */
	rte_spinlock_lock(&hw->state_lock);
	hw->started = 0;

	/*
	 * Waitting for datapath to complete before resetting queues.
	 * 1 ms should be enough for the ongoing Tx/Rx function to finish.
	 */
	rte_delay_ms(1);

	/* Vring reset for each Tx queue and Rx queue. */
	for (i = 0; i < dev->data->nb_rx_queues; i++) {
		rxvq = dev->data->rx_queues[i];
		virtqueue_rxvq_reset_packed(rxvq->vq);
		virtio_dev_rx_queue_setup_finish(dev, i);
	}

	for (i = 0; i < dev->data->nb_tx_queues; i++) {
		txvq = dev->data->tx_queues[i];
		virtqueue_txvq_reset_packed(txvq->vq);
	}

	hw->started = 1;
	rte_spinlock_unlock(&hw->state_lock);
}


static int
virtio_user_server_reconnect(struct virtio_user_dev *dev)
{
	int ret, connectfd, old_status;
	struct rte_eth_dev *eth_dev = &rte_eth_devices[dev->port_id];
	struct virtio_hw *hw = eth_dev->data->dev_private;
	uint64_t protocol_features;

	connectfd = accept(dev->listenfd, NULL, NULL);
	if (connectfd < 0)
		return -1;

	dev->vhostfd = connectfd;
	old_status = vtpci_get_status(hw);

	vtpci_reset(hw);

	vtpci_set_status(hw, VIRTIO_CONFIG_STATUS_ACK);

	vtpci_set_status(hw, VIRTIO_CONFIG_STATUS_DRIVER);

	if (dev->ops->send_request(dev, VHOST_USER_GET_FEATURES,
				   &dev->device_features) < 0) {
		PMD_INIT_LOG(ERR, "get_features failed: %s",
			     strerror(errno));
		return -1;
	}

	if (dev->device_features &
			(1ULL << VHOST_USER_F_PROTOCOL_FEATURES)) {
		if (dev->ops->send_request(dev,
					VHOST_USER_GET_PROTOCOL_FEATURES,
					&protocol_features))
			return -1;

		/* Offer VHOST_USER_PROTOCOL_F_STATUS */
		dev->protocol_features |=
			(1ULL << VHOST_USER_PROTOCOL_F_STATUS);
		dev->protocol_features &= protocol_features;

		if (dev->ops->send_request(dev,
					VHOST_USER_SET_PROTOCOL_FEATURES,
					&dev->protocol_features))
			return -1;

		if (!(dev->protocol_features &
				(1ULL << VHOST_USER_PROTOCOL_F_MQ)))
			dev->unsupported_features |= (1ull << VIRTIO_NET_F_MQ);
	}

	dev->device_features |= dev->frontend_features;

	/* umask vhost-user unsupported features */
	dev->device_features &= ~(dev->unsupported_features);

	dev->features &= dev->device_features;

	/* For packed ring, resetting queues is required in reconnection. */
	if (vtpci_packed_queue(hw) &&
	   (old_status & VIRTIO_CONFIG_STATUS_DRIVER_OK)) {
		PMD_INIT_LOG(NOTICE, "Packets on the fly will be dropped"
				" when packed ring reconnecting.");
		virtio_user_reset_queues_packed(eth_dev);
	}

	vtpci_set_status(hw, VIRTIO_CONFIG_STATUS_FEATURES_OK);

	/* Start the device */
	vtpci_set_status(hw, VIRTIO_CONFIG_STATUS_DRIVER_OK);
	if (!dev->started)
		return -1;

	if (dev->queue_pairs > 1) {
		ret = virtio_user_handle_mq(dev, dev->queue_pairs);
		if (ret != 0) {
			PMD_INIT_LOG(ERR, "Fails to enable multi-queue pairs!");
			return -1;
		}
	}
	if (eth_dev->data->dev_flags & RTE_ETH_DEV_INTR_LSC) {
		if (rte_intr_disable(eth_dev->intr_handle) < 0) {
			PMD_DRV_LOG(ERR, "interrupt disable failed");
			return -1;
		}
		rte_intr_callback_unregister(eth_dev->intr_handle,
					     virtio_interrupt_handler,
					     eth_dev);
		eth_dev->intr_handle->fd = connectfd;
		rte_intr_callback_register(eth_dev->intr_handle,
					   virtio_interrupt_handler, eth_dev);

		if (rte_intr_enable(eth_dev->intr_handle) < 0) {
			PMD_DRV_LOG(ERR, "interrupt enable failed");
			return -1;
		}
	}
	PMD_INIT_LOG(NOTICE, "server mode virtio-user reconnection succeeds!");
	return 0;
}

static void
virtio_user_delayed_handler(void *param)
{
	struct virtio_hw *hw = (struct virtio_hw *)param;
	struct rte_eth_dev *eth_dev = &rte_eth_devices[hw->port_id];
	struct virtio_user_dev *dev = virtio_user_get_dev(hw);

	if (rte_intr_disable(eth_dev->intr_handle) < 0) {
		PMD_DRV_LOG(ERR, "interrupt disable failed");
		return;
	}
	rte_intr_callback_unregister(eth_dev->intr_handle,
				     virtio_interrupt_handler, eth_dev);
	if (dev->is_server) {
		if (dev->vhostfd >= 0) {
			close(dev->vhostfd);
			dev->vhostfd = -1;
			/* Until the featuers are negotiated again, don't assume
			 * the backend supports VHOST_USER_PROTOCOL_F_STATUS
			 */
			dev->protocol_features &=
				~(1ULL << VHOST_USER_PROTOCOL_F_STATUS);
		}
		eth_dev->intr_handle->fd = dev->listenfd;
		rte_intr_callback_register(eth_dev->intr_handle,
					   virtio_interrupt_handler, eth_dev);
		if (rte_intr_enable(eth_dev->intr_handle) < 0) {
			PMD_DRV_LOG(ERR, "interrupt enable failed");
			return;
		}
	}
}

static void
virtio_user_read_dev_config(struct virtio_hw *hw, size_t offset,
		     void *dst, int length)
{
	int i;
	struct virtio_user_dev *dev = virtio_user_get_dev(hw);

	if (offset == offsetof(struct virtio_net_config, mac) &&
	    length == RTE_ETHER_ADDR_LEN) {
		for (i = 0; i < RTE_ETHER_ADDR_LEN; ++i)
			((uint8_t *)dst)[i] = dev->mac_addr[i];
		return;
	}

	if (offset == offsetof(struct virtio_net_config, status)) {
		char buf[128];

		if (dev->vhostfd >= 0) {
			int r;
			int flags;

			flags = fcntl(dev->vhostfd, F_GETFL);
			if (fcntl(dev->vhostfd, F_SETFL,
					flags | O_NONBLOCK) == -1) {
				PMD_DRV_LOG(ERR, "error setting O_NONBLOCK flag");
				return;
			}
			r = recv(dev->vhostfd, buf, 128, MSG_PEEK);
			if (r == 0 || (r < 0 && errno != EAGAIN)) {
				dev->net_status &= (~VIRTIO_NET_S_LINK_UP);
				PMD_DRV_LOG(ERR, "virtio-user port %u is down",
					    hw->port_id);

				/* This function could be called in the process
				 * of interrupt handling, callback cannot be
				 * unregistered here, set an alarm to do it.
				 */
				rte_eal_alarm_set(1,
						  virtio_user_delayed_handler,
						  (void *)hw);
			} else {
				dev->net_status |= VIRTIO_NET_S_LINK_UP;
			}
			if (fcntl(dev->vhostfd, F_SETFL,
					flags & ~O_NONBLOCK) == -1) {
				PMD_DRV_LOG(ERR, "error clearing O_NONBLOCK flag");
				return;
			}
		} else if (dev->is_server) {
			dev->net_status &= (~VIRTIO_NET_S_LINK_UP);
			if (virtio_user_server_reconnect(dev) >= 0)
				dev->net_status |= VIRTIO_NET_S_LINK_UP;
		}

		*(uint16_t *)dst = dev->net_status;
	}

	if (offset == offsetof(struct virtio_net_config, max_virtqueue_pairs))
		*(uint16_t *)dst = dev->max_queue_pairs;
}

static void
virtio_user_write_dev_config(struct virtio_hw *hw, size_t offset,
		      const void *src, int length)
{
	int i;
	struct virtio_user_dev *dev = virtio_user_get_dev(hw);

	if ((offset == offsetof(struct virtio_net_config, mac)) &&
	    (length == RTE_ETHER_ADDR_LEN))
		for (i = 0; i < RTE_ETHER_ADDR_LEN; ++i)
			dev->mac_addr[i] = ((const uint8_t *)src)[i];
	else
		PMD_DRV_LOG(ERR, "not supported offset=%zu, len=%d",
			    offset, length);
}

static void
virtio_user_reset(struct virtio_hw *hw)
{
	struct virtio_user_dev *dev = virtio_user_get_dev(hw);

	if (dev->status & VIRTIO_CONFIG_STATUS_DRIVER_OK)
		virtio_user_stop_device(dev);
}

static void
virtio_user_set_status(struct virtio_hw *hw, uint8_t status)
{
	struct virtio_user_dev *dev = virtio_user_get_dev(hw);
	uint8_t old_status = dev->status;

	if (status & VIRTIO_CONFIG_STATUS_FEATURES_OK &&
			~old_status & VIRTIO_CONFIG_STATUS_FEATURES_OK)
		virtio_user_dev_set_features(dev);
	if (status & VIRTIO_CONFIG_STATUS_DRIVER_OK)
		virtio_user_start_device(dev);
	else if (status == VIRTIO_CONFIG_STATUS_RESET)
		virtio_user_reset(hw);

	virtio_user_dev_set_status(dev, status);
}

static uint8_t
virtio_user_get_status(struct virtio_hw *hw)
{
	struct virtio_user_dev *dev = virtio_user_get_dev(hw);

	virtio_user_dev_update_status(dev);

	return dev->status;
}

static uint64_t
virtio_user_get_features(struct virtio_hw *hw)
{
	struct virtio_user_dev *dev = virtio_user_get_dev(hw);

	/* unmask feature bits defined in vhost user protocol */
	return dev->device_features & VIRTIO_PMD_SUPPORTED_GUEST_FEATURES;
}

static void
virtio_user_set_features(struct virtio_hw *hw, uint64_t features)
{
	struct virtio_user_dev *dev = virtio_user_get_dev(hw);

	dev->features = features & dev->device_features;
}

static uint8_t
virtio_user_get_isr(struct virtio_hw *hw __rte_unused)
{
	/* rxq interrupts and config interrupt are separated in virtio-user,
	 * here we only report config change.
	 */
	return VIRTIO_PCI_ISR_CONFIG;
}

static uint16_t
virtio_user_set_config_irq(struct virtio_hw *hw __rte_unused,
		    uint16_t vec __rte_unused)
{
	return 0;
}

static uint16_t
virtio_user_set_queue_irq(struct virtio_hw *hw __rte_unused,
			  struct virtqueue *vq __rte_unused,
			  uint16_t vec)
{
	/* pretend we have done that */
	return vec;
}

/* This function is to get the queue size, aka, number of descs, of a specified
 * queue. Different with the VHOST_USER_GET_QUEUE_NUM, which is used to get the
 * max supported queues.
 */
static uint16_t
virtio_user_get_queue_num(struct virtio_hw *hw, uint16_t queue_id __rte_unused)
{
	struct virtio_user_dev *dev = virtio_user_get_dev(hw);

	/* Lwrrently, each queue has same queue size */
	return dev->queue_size;
}

static void
virtio_user_setup_queue_packed(struct virtqueue *vq,
			       struct virtio_user_dev *dev)
{
	uint16_t queue_idx = vq->vq_queue_index;
	struct vring_packed *vring;
	uint64_t desc_addr;
	uint64_t avail_addr;
	uint64_t used_addr;
	uint16_t i;

	vring  = &dev->packed_vrings[queue_idx];
	desc_addr = (uintptr_t)vq->vq_ring_virt_mem;
	avail_addr = desc_addr + vq->vq_nentries *
		sizeof(struct vring_packed_desc);
	used_addr = RTE_ALIGN_CEIL(avail_addr +
			   sizeof(struct vring_packed_desc_event),
			   VIRTIO_PCI_VRING_ALIGN);
	vring->num = vq->vq_nentries;
	vring->desc = (void *)(uintptr_t)desc_addr;
	vring->driver = (void *)(uintptr_t)avail_addr;
	vring->device = (void *)(uintptr_t)used_addr;
	dev->packed_queues[queue_idx].avail_wrap_counter = true;
	dev->packed_queues[queue_idx].used_wrap_counter = true;

	for (i = 0; i < vring->num; i++)
		vring->desc[i].flags = 0;
}

static void
virtio_user_setup_queue_split(struct virtqueue *vq, struct virtio_user_dev *dev)
{
	uint16_t queue_idx = vq->vq_queue_index;
	uint64_t desc_addr, avail_addr, used_addr;

	desc_addr = (uintptr_t)vq->vq_ring_virt_mem;
	avail_addr = desc_addr + vq->vq_nentries * sizeof(struct vring_desc);
	used_addr = RTE_ALIGN_CEIL(avail_addr + offsetof(struct vring_avail,
							 ring[vq->vq_nentries]),
				   VIRTIO_PCI_VRING_ALIGN);

	dev->vrings[queue_idx].num = vq->vq_nentries;
	dev->vrings[queue_idx].desc = (void *)(uintptr_t)desc_addr;
	dev->vrings[queue_idx].avail = (void *)(uintptr_t)avail_addr;
	dev->vrings[queue_idx].used = (void *)(uintptr_t)used_addr;
}

static int
virtio_user_setup_queue(struct virtio_hw *hw, struct virtqueue *vq)
{
	struct virtio_user_dev *dev = virtio_user_get_dev(hw);

	if (vtpci_packed_queue(hw))
		virtio_user_setup_queue_packed(vq, dev);
	else
		virtio_user_setup_queue_split(vq, dev);

	return 0;
}

static void
virtio_user_del_queue(struct virtio_hw *hw, struct virtqueue *vq)
{
	/* For legacy devices, write 0 to VIRTIO_PCI_QUEUE_PFN port, QEMU
	 * correspondingly stops the ioeventfds, and reset the status of
	 * the device.
	 * For modern devices, set queue desc, avail, used in PCI bar to 0,
	 * not see any more behavior in QEMU.
	 *
	 * Here we just care about what information to deliver to vhost-user
	 * or vhost-kernel. So we just close ioeventfd for now.
	 */
	struct virtio_user_dev *dev = virtio_user_get_dev(hw);

	close(dev->callfds[vq->vq_queue_index]);
	close(dev->kickfds[vq->vq_queue_index]);
}

static void
virtio_user_notify_queue(struct virtio_hw *hw, struct virtqueue *vq)
{
	uint64_t buf = 1;
	struct virtio_user_dev *dev = virtio_user_get_dev(hw);

	if (hw->cvq && (hw->cvq->vq == vq)) {
		if (vtpci_packed_queue(vq->hw))
			virtio_user_handle_cq_packed(dev, vq->vq_queue_index);
		else
			virtio_user_handle_cq(dev, vq->vq_queue_index);
		return;
	}

	if (write(dev->kickfds[vq->vq_queue_index], &buf, sizeof(buf)) < 0)
		PMD_DRV_LOG(ERR, "failed to kick backend: %s",
			    strerror(errno));
}

const struct virtio_pci_ops virtio_user_ops = {
	.read_dev_cfg	= virtio_user_read_dev_config,
	.write_dev_cfg	= virtio_user_write_dev_config,
	.get_status	= virtio_user_get_status,
	.set_status	= virtio_user_set_status,
	.get_features	= virtio_user_get_features,
	.set_features	= virtio_user_set_features,
	.get_isr	= virtio_user_get_isr,
	.set_config_irq	= virtio_user_set_config_irq,
	.set_queue_irq	= virtio_user_set_queue_irq,
	.get_queue_num	= virtio_user_get_queue_num,
	.setup_queue	= virtio_user_setup_queue,
	.del_queue	= virtio_user_del_queue,
	.notify_queue	= virtio_user_notify_queue,
};

static const char *valid_args[] = {
#define VIRTIO_USER_ARG_QUEUES_NUM     "queues"
	VIRTIO_USER_ARG_QUEUES_NUM,
#define VIRTIO_USER_ARG_CQ_NUM         "cq"
	VIRTIO_USER_ARG_CQ_NUM,
#define VIRTIO_USER_ARG_MAC            "mac"
	VIRTIO_USER_ARG_MAC,
#define VIRTIO_USER_ARG_PATH           "path"
	VIRTIO_USER_ARG_PATH,
#define VIRTIO_USER_ARG_QUEUE_SIZE     "queue_size"
	VIRTIO_USER_ARG_QUEUE_SIZE,
#define VIRTIO_USER_ARG_INTERFACE_NAME "iface"
	VIRTIO_USER_ARG_INTERFACE_NAME,
#define VIRTIO_USER_ARG_SERVER_MODE    "server"
	VIRTIO_USER_ARG_SERVER_MODE,
#define VIRTIO_USER_ARG_MRG_RXBUF      "mrg_rxbuf"
	VIRTIO_USER_ARG_MRG_RXBUF,
#define VIRTIO_USER_ARG_IN_ORDER       "in_order"
	VIRTIO_USER_ARG_IN_ORDER,
#define VIRTIO_USER_ARG_PACKED_VQ      "packed_vq"
	VIRTIO_USER_ARG_PACKED_VQ,
#define VIRTIO_USER_ARG_SPEED          "speed"
	VIRTIO_USER_ARG_SPEED,
#define VIRTIO_USER_ARG_VECTORIZED     "vectorized"
	VIRTIO_USER_ARG_VECTORIZED,
	NULL
};

#define VIRTIO_USER_DEF_CQ_EN	0
#define VIRTIO_USER_DEF_Q_NUM	1
#define VIRTIO_USER_DEF_Q_SZ	256
#define VIRTIO_USER_DEF_SERVER_MODE	0

static int
get_string_arg(const char *key __rte_unused,
	       const char *value, void *extra_args)
{
	if (!value || !extra_args)
		return -EILWAL;

	*(char **)extra_args = strdup(value);

	if (!*(char **)extra_args)
		return -ENOMEM;

	return 0;
}

static int
get_integer_arg(const char *key __rte_unused,
		const char *value, void *extra_args)
{
	uint64_t integer = 0;
	if (!value || !extra_args)
		return -EILWAL;
	errno = 0;
	integer = strtoull(value, NULL, 0);
	/* extra_args keeps default value, it should be replaced
	 * only in case of successful parsing of the 'value' arg
	 */
	if (errno == 0)
		*(uint64_t *)extra_args = integer;
	return -errno;
}

static uint32_t
vdpa_dynamic_major_num(void)
{
	FILE *fp;
	char *line = NULL;
	size_t size;
	char name[11];
	bool found = false;
	uint32_t num;

	fp = fopen("/proc/devices", "r");
	if (fp == NULL) {
		PMD_INIT_LOG(ERR, "Cannot open /proc/devices: %s",
			     strerror(errno));
		return UNNAMED_MAJOR;
	}

	while (getline(&line, &size, fp) > 0) {
		char *stripped = line + strspn(line, " ");
		if ((sscanf(stripped, "%u %10s", &num, name) == 2) &&
		    (strncmp(name, "vhost-vdpa", 10) == 0)) {
			found = true;
			break;
		}
	}
	fclose(fp);
	return found ? num : UNNAMED_MAJOR;
}

static enum virtio_user_backend_type
virtio_user_backend_type(const char *path)
{
	struct stat sb;

	if (stat(path, &sb) == -1) {
		if (errno == ENOENT)
			return VIRTIO_USER_BACKEND_VHOST_USER;

		PMD_INIT_LOG(ERR, "Stat fails: %s (%s)\n", path,
			     strerror(errno));
		return VIRTIO_USER_BACKEND_UNKNOWN;
	}

	if (S_ISSOCK(sb.st_mode)) {
		return VIRTIO_USER_BACKEND_VHOST_USER;
	} else if (S_ISCHR(sb.st_mode)) {
		if (major(sb.st_rdev) == MISC_MAJOR)
			return VIRTIO_USER_BACKEND_VHOST_KERNEL;
		if (major(sb.st_rdev) == vdpa_dynamic_major_num())
			return VIRTIO_USER_BACKEND_VHOST_VDPA;
	}
	return VIRTIO_USER_BACKEND_UNKNOWN;
}

static struct rte_eth_dev *
virtio_user_eth_dev_alloc(struct rte_vdev_device *vdev)
{
	struct rte_eth_dev *eth_dev;
	struct rte_eth_dev_data *data;
	struct virtio_hw *hw;
	struct virtio_user_dev *dev;

	eth_dev = rte_eth_vdev_allocate(vdev, sizeof(*hw));
	if (!eth_dev) {
		PMD_INIT_LOG(ERR, "cannot alloc rte_eth_dev");
		return NULL;
	}

	data = eth_dev->data;
	hw = eth_dev->data->dev_private;

	dev = rte_zmalloc(NULL, sizeof(*dev), 0);
	if (!dev) {
		PMD_INIT_LOG(ERR, "malloc virtio_user_dev failed");
		rte_eth_dev_release_port(eth_dev);
		return NULL;
	}

	hw->port_id = data->port_id;
	dev->port_id = data->port_id;
	virtio_hw_internal[hw->port_id].vtpci_ops = &virtio_user_ops;
	/*
	 * MSIX is required to enable LSC (see virtio_init_device).
	 * Here just pretend that we support msix.
	 */
	hw->use_msix = 1;
	hw->modern   = 0;
	hw->use_vec_rx = 0;
	hw->use_vec_tx = 0;
	hw->use_inorder_rx = 0;
	hw->use_inorder_tx = 0;
	hw->virtio_user_dev = dev;
	return eth_dev;
}

static void
virtio_user_eth_dev_free(struct rte_eth_dev *eth_dev)
{
	struct rte_eth_dev_data *data = eth_dev->data;
	struct virtio_hw *hw = data->dev_private;

	rte_free(hw->virtio_user_dev);
	rte_eth_dev_release_port(eth_dev);
}

/* Dev initialization routine. Ilwoked once for each virtio vdev at
 * EAL init time, see rte_bus_probe().
 * Returns 0 on success.
 */
static int
virtio_user_pmd_probe(struct rte_vdev_device *dev)
{
	struct rte_kvargs *kvlist = NULL;
	struct rte_eth_dev *eth_dev;
	struct virtio_hw *hw;
	enum virtio_user_backend_type backend_type = VIRTIO_USER_BACKEND_UNKNOWN;
	uint64_t queues = VIRTIO_USER_DEF_Q_NUM;
	uint64_t cq = VIRTIO_USER_DEF_CQ_EN;
	uint64_t queue_size = VIRTIO_USER_DEF_Q_SZ;
	uint64_t server_mode = VIRTIO_USER_DEF_SERVER_MODE;
	uint64_t mrg_rxbuf = 1;
	uint64_t in_order = 1;
	uint64_t packed_vq = 0;
	uint64_t vectorized = 0;
	char *path = NULL;
	char *ifname = NULL;
	char *mac_addr = NULL;
	int ret = -1;

	if (rte_eal_process_type() == RTE_PROC_SECONDARY) {
		const char *name = rte_vdev_device_name(dev);
		eth_dev = rte_eth_dev_attach_secondary(name);
		if (!eth_dev) {
			PMD_INIT_LOG(ERR, "Failed to probe %s", name);
			return -1;
		}

		if (eth_virtio_dev_init(eth_dev) < 0) {
			PMD_INIT_LOG(ERR, "eth_virtio_dev_init fails");
			rte_eth_dev_release_port(eth_dev);
			return -1;
		}

		eth_dev->dev_ops = &virtio_user_secondary_eth_dev_ops;
		eth_dev->device = &dev->device;
		rte_eth_dev_probing_finish(eth_dev);
		return 0;
	}

	kvlist = rte_kvargs_parse(rte_vdev_device_args(dev), valid_args);
	if (!kvlist) {
		PMD_INIT_LOG(ERR, "error when parsing param");
		goto end;
	}

	if (rte_kvargs_count(kvlist, VIRTIO_USER_ARG_PATH) == 1) {
		if (rte_kvargs_process(kvlist, VIRTIO_USER_ARG_PATH,
				       &get_string_arg, &path) < 0) {
			PMD_INIT_LOG(ERR, "error to parse %s",
				     VIRTIO_USER_ARG_PATH);
			goto end;
		}
	} else {
		PMD_INIT_LOG(ERR, "arg %s is mandatory for virtio_user",
			     VIRTIO_USER_ARG_PATH);
		goto end;
	}

	backend_type = virtio_user_backend_type(path);
	if (backend_type == VIRTIO_USER_BACKEND_UNKNOWN) {
		PMD_INIT_LOG(ERR,
			     "unable to determine backend type for path %s",
			path);
		goto end;
	}
	PMD_INIT_LOG(INFO, "Backend type detected: %s",
		     virtio_user_backend_strings[backend_type]);

	if (rte_kvargs_count(kvlist, VIRTIO_USER_ARG_INTERFACE_NAME) == 1) {
		if (backend_type != VIRTIO_USER_BACKEND_VHOST_KERNEL) {
			PMD_INIT_LOG(ERR,
				"arg %s applies only to vhost-kernel backend",
				VIRTIO_USER_ARG_INTERFACE_NAME);
			goto end;
		}

		if (rte_kvargs_process(kvlist, VIRTIO_USER_ARG_INTERFACE_NAME,
				       &get_string_arg, &ifname) < 0) {
			PMD_INIT_LOG(ERR, "error to parse %s",
				     VIRTIO_USER_ARG_INTERFACE_NAME);
			goto end;
		}
	}

	if (rte_kvargs_count(kvlist, VIRTIO_USER_ARG_MAC) == 1) {
		if (rte_kvargs_process(kvlist, VIRTIO_USER_ARG_MAC,
				       &get_string_arg, &mac_addr) < 0) {
			PMD_INIT_LOG(ERR, "error to parse %s",
				     VIRTIO_USER_ARG_MAC);
			goto end;
		}
	}

	if (rte_kvargs_count(kvlist, VIRTIO_USER_ARG_QUEUE_SIZE) == 1) {
		if (rte_kvargs_process(kvlist, VIRTIO_USER_ARG_QUEUE_SIZE,
				       &get_integer_arg, &queue_size) < 0) {
			PMD_INIT_LOG(ERR, "error to parse %s",
				     VIRTIO_USER_ARG_QUEUE_SIZE);
			goto end;
		}
	}

	if (rte_kvargs_count(kvlist, VIRTIO_USER_ARG_QUEUES_NUM) == 1) {
		if (rte_kvargs_process(kvlist, VIRTIO_USER_ARG_QUEUES_NUM,
				       &get_integer_arg, &queues) < 0) {
			PMD_INIT_LOG(ERR, "error to parse %s",
				     VIRTIO_USER_ARG_QUEUES_NUM);
			goto end;
		}
	}

	if (rte_kvargs_count(kvlist, VIRTIO_USER_ARG_SERVER_MODE) == 1) {
		if (rte_kvargs_process(kvlist, VIRTIO_USER_ARG_SERVER_MODE,
				       &get_integer_arg, &server_mode) < 0) {
			PMD_INIT_LOG(ERR, "error to parse %s",
				     VIRTIO_USER_ARG_SERVER_MODE);
			goto end;
		}
	}

	if (rte_kvargs_count(kvlist, VIRTIO_USER_ARG_CQ_NUM) == 1) {
		if (rte_kvargs_process(kvlist, VIRTIO_USER_ARG_CQ_NUM,
				       &get_integer_arg, &cq) < 0) {
			PMD_INIT_LOG(ERR, "error to parse %s",
				     VIRTIO_USER_ARG_CQ_NUM);
			goto end;
		}
	} else if (queues > 1) {
		cq = 1;
	}

	if (rte_kvargs_count(kvlist, VIRTIO_USER_ARG_PACKED_VQ) == 1) {
		if (rte_kvargs_process(kvlist, VIRTIO_USER_ARG_PACKED_VQ,
				       &get_integer_arg, &packed_vq) < 0) {
			PMD_INIT_LOG(ERR, "error to parse %s",
				     VIRTIO_USER_ARG_PACKED_VQ);
			goto end;
		}
	}

	if (rte_kvargs_count(kvlist, VIRTIO_USER_ARG_VECTORIZED) == 1) {
		if (rte_kvargs_process(kvlist, VIRTIO_USER_ARG_VECTORIZED,
				       &get_integer_arg, &vectorized) < 0) {
			PMD_INIT_LOG(ERR, "error to parse %s",
				     VIRTIO_USER_ARG_VECTORIZED);
			goto end;
		}
	}

	if (queues > 1 && cq == 0) {
		PMD_INIT_LOG(ERR, "multi-q requires ctrl-q");
		goto end;
	}

	if (queues > VIRTIO_MAX_VIRTQUEUE_PAIRS) {
		PMD_INIT_LOG(ERR, "arg %s %" PRIu64 " exceeds the limit %u",
			VIRTIO_USER_ARG_QUEUES_NUM, queues,
			VIRTIO_MAX_VIRTQUEUE_PAIRS);
		goto end;
	}

	if (rte_kvargs_count(kvlist, VIRTIO_USER_ARG_MRG_RXBUF) == 1) {
		if (rte_kvargs_process(kvlist, VIRTIO_USER_ARG_MRG_RXBUF,
				       &get_integer_arg, &mrg_rxbuf) < 0) {
			PMD_INIT_LOG(ERR, "error to parse %s",
				     VIRTIO_USER_ARG_MRG_RXBUF);
			goto end;
		}
	}

	if (rte_kvargs_count(kvlist, VIRTIO_USER_ARG_IN_ORDER) == 1) {
		if (rte_kvargs_process(kvlist, VIRTIO_USER_ARG_IN_ORDER,
				       &get_integer_arg, &in_order) < 0) {
			PMD_INIT_LOG(ERR, "error to parse %s",
				     VIRTIO_USER_ARG_IN_ORDER);
			goto end;
		}
	}

	eth_dev = virtio_user_eth_dev_alloc(dev);
	if (!eth_dev) {
		PMD_INIT_LOG(ERR, "virtio_user fails to alloc device");
		goto end;
	}

	hw = eth_dev->data->dev_private;
	if (virtio_user_dev_init(hw->virtio_user_dev, path, queues, cq,
			 queue_size, mac_addr, &ifname, server_mode,
			 mrg_rxbuf, in_order, packed_vq, backend_type) < 0) {
		PMD_INIT_LOG(ERR, "virtio_user_dev_init fails");
		virtio_user_eth_dev_free(eth_dev);
		goto end;
	}

	/* previously called by pci probing for physical dev */
	if (eth_virtio_dev_init(eth_dev) < 0) {
		PMD_INIT_LOG(ERR, "eth_virtio_dev_init fails");
		virtio_user_eth_dev_free(eth_dev);
		goto end;
	}

	if (vectorized) {
		if (packed_vq) {
#if defined(CC_AVX512_SUPPORT)
			hw->use_vec_rx = 1;
			hw->use_vec_tx = 1;
#else
			PMD_INIT_LOG(INFO,
				"building environment do not support packed ring vectorized");
#endif
		} else {
			hw->use_vec_rx = 1;
		}
	}

	rte_eth_dev_probing_finish(eth_dev);
	ret = 0;

end:
	if (kvlist)
		rte_kvargs_free(kvlist);
	if (path)
		free(path);
	if (mac_addr)
		free(mac_addr);
	if (ifname)
		free(ifname);
	return ret;
}

static int
virtio_user_pmd_remove(struct rte_vdev_device *vdev)
{
	const char *name;
	struct rte_eth_dev *eth_dev;

	if (!vdev)
		return -EILWAL;

	name = rte_vdev_device_name(vdev);
	PMD_DRV_LOG(INFO, "Un-Initializing %s", name);
	eth_dev = rte_eth_dev_allocated(name);
	/* Port has already been released by close. */
	if (!eth_dev)
		return 0;

	if (rte_eal_process_type() != RTE_PROC_PRIMARY)
		return rte_eth_dev_release_port(eth_dev);

	/* make sure the device is stopped, queues freed */
	return rte_eth_dev_close(eth_dev->data->port_id);
}

static int virtio_user_pmd_dma_map(struct rte_vdev_device *vdev, void *addr,
		uint64_t iova, size_t len)
{
	const char *name;
	struct rte_eth_dev *eth_dev;
	struct virtio_user_dev *dev;
	struct virtio_hw *hw;

	if (!vdev)
		return -EILWAL;

	name = rte_vdev_device_name(vdev);
	eth_dev = rte_eth_dev_allocated(name);
	/* Port has already been released by close. */
	if (!eth_dev)
		return 0;

	hw = (struct virtio_hw *)eth_dev->data->dev_private;
	dev = hw->virtio_user_dev;

	if (dev->ops->dma_map)
		return dev->ops->dma_map(dev, addr, iova, len);

	return 0;
}

static int virtio_user_pmd_dma_unmap(struct rte_vdev_device *vdev, void *addr,
		uint64_t iova, size_t len)
{
	const char *name;
	struct rte_eth_dev *eth_dev;
	struct virtio_user_dev *dev;
	struct virtio_hw *hw;

	if (!vdev)
		return -EILWAL;

	name = rte_vdev_device_name(vdev);
	eth_dev = rte_eth_dev_allocated(name);
	/* Port has already been released by close. */
	if (!eth_dev)
		return 0;

	hw = (struct virtio_hw *)eth_dev->data->dev_private;
	dev = hw->virtio_user_dev;

	if (dev->ops->dma_unmap)
		return dev->ops->dma_unmap(dev, addr, iova, len);

	return 0;
}

static struct rte_vdev_driver virtio_user_driver = {
	.probe = virtio_user_pmd_probe,
	.remove = virtio_user_pmd_remove,
	.dma_map = virtio_user_pmd_dma_map,
	.dma_unmap = virtio_user_pmd_dma_unmap,
};

RTE_PMD_REGISTER_VDEV(net_virtio_user, virtio_user_driver);
RTE_PMD_REGISTER_ALIAS(net_virtio_user, virtio_user);
RTE_PMD_REGISTER_PARAM_STRING(net_virtio_user,
	"path=<path> "
	"mac=<mac addr> "
	"cq=<int> "
	"queue_size=<int> "
	"queues=<int> "
	"iface=<string> "
	"server=<0|1> "
	"mrg_rxbuf=<0|1> "
	"in_order=<0|1> "
	"packed_vq=<0|1> "
	"speed=<int> "
	"vectorized=<0|1>");
