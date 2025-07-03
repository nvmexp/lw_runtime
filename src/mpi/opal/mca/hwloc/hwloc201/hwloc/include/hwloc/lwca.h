/*
 * Copyright © 2010-2017 Inria.  All rights reserved.
 * Copyright © 2010-2011 Université Bordeaux
 * Copyright © 2011 Cisco Systems, Inc.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Macros to help interaction between hwloc and the LWCA Driver API.
 *
 * Applications that use both hwloc and the LWCA Driver API may want to
 * include this file so as to get topology information for LWCA devices.
 *
 */

#ifndef HWLOC_LWDA_H
#define HWLOC_LWDA_H

#include <hwloc.h>
#include <hwloc/autogen/config.h>
#include <hwloc/helper.h>
#ifdef HWLOC_LINUX_SYS
#include <hwloc/linux.h>
#endif

#include <lwca.h>


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_lwda Interoperability with the LWCA Driver API
 *
 * This interface offers ways to retrieve topology information about
 * LWCA devices when using the LWCA Driver API.
 *
 * @{
 */

/** \brief Return the domain, bus and device IDs of the LWCA device \p lwdevice.
 *
 * Device \p lwdevice must match the local machine.
 */
static __hwloc_inline int
hwloc_lwda_get_device_pci_ids(hwloc_topology_t topology __hwloc_attribute_unused,
			      LWdevice lwdevice, int *domain, int *bus, int *dev)
{
  LWresult cres;

#if LWDA_VERSION >= 4000
  cres = lwDeviceGetAttribute(domain, LW_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, lwdevice);
  if (cres != LWDA_SUCCESS) {
    errno = ENOSYS;
    return -1;
  }
#else
  *domain = 0;
#endif
  cres = lwDeviceGetAttribute(bus, LW_DEVICE_ATTRIBUTE_PCI_BUS_ID, lwdevice);
  if (cres != LWDA_SUCCESS) {
    errno = ENOSYS;
    return -1;
  }
  cres = lwDeviceGetAttribute(dev, LW_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, lwdevice);
  if (cres != LWDA_SUCCESS) {
    errno = ENOSYS;
    return -1;
  }

  return 0;
}

/** \brief Get the CPU set of logical processors that are physically
 * close to device \p lwdevice.
 *
 * Return the CPU set describing the locality of the LWCA device \p lwdevice.
 *
 * Topology \p topology and device \p lwdevice must match the local machine.
 * I/O devices detection and the LWCA component are not needed in the topology.
 *
 * The function only returns the locality of the device.
 * If more information about the device is needed, OS objects should
 * be used instead, see hwloc_lwda_get_device_osdev()
 * and hwloc_lwda_get_device_osdev_by_index().
 *
 * This function is lwrrently only implemented in a meaningful way for
 * Linux; other systems will simply get a full cpuset.
 */
static __hwloc_inline int
hwloc_lwda_get_device_cpuset(hwloc_topology_t topology __hwloc_attribute_unused,
			     LWdevice lwdevice, hwloc_cpuset_t set)
{
#ifdef HWLOC_LINUX_SYS
  /* If we're on Linux, use the sysfs mechanism to get the local cpus */
#define HWLOC_LWDA_DEVICE_SYSFS_PATH_MAX 128
  char path[HWLOC_LWDA_DEVICE_SYSFS_PATH_MAX];
  int domainid, busid, deviceid;

  if (hwloc_lwda_get_device_pci_ids(topology, lwdevice, &domainid, &busid, &deviceid))
    return -1;

  if (!hwloc_topology_is_thissystem(topology)) {
    errno = EILWAL;
    return -1;
  }

  sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.0/local_cpus", domainid, busid, deviceid);
  if (hwloc_linux_read_path_as_cpumask(path, set) < 0
      || hwloc_bitmap_iszero(set))
    hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#else
  /* Non-Linux systems simply get a full cpuset */
  hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#endif
  return 0;
}

/** \brief Get the hwloc PCI device object corresponding to the
 * LWCA device \p lwdevice.
 *
 * Return the PCI device object describing the LWCA device \p lwdevice.
 * Return NULL if there is none.
 *
 * Topology \p topology and device \p lwdevice must match the local machine.
 * I/O devices detection must be enabled in topology \p topology.
 * The LWCA component is not needed in the topology.
 */
static __hwloc_inline hwloc_obj_t
hwloc_lwda_get_device_pcidev(hwloc_topology_t topology, LWdevice lwdevice)
{
  int domain, bus, dev;

  if (hwloc_lwda_get_device_pci_ids(topology, lwdevice, &domain, &bus, &dev))
    return NULL;

  return hwloc_get_pcidev_by_busid(topology, domain, bus, dev, 0);
}

/** \brief Get the hwloc OS device object corresponding to LWCA device \p lwdevice.
 *
 * Return the hwloc OS device object that describes the given
 * LWCA device \p lwdevice. Return NULL if there is none.
 *
 * Topology \p topology and device \p lwdevice must match the local machine.
 * I/O devices detection and the LWCA component must be enabled in the topology.
 * If not, the locality of the object may still be found using
 * hwloc_lwda_get_device_cpuset().
 *
 * \note This function cannot work if PCI devices are filtered out.
 *
 * \note The corresponding hwloc PCI device may be found by looking
 * at the result parent pointer (unless PCI devices are filtered out).
 */
static __hwloc_inline hwloc_obj_t
hwloc_lwda_get_device_osdev(hwloc_topology_t topology, LWdevice lwdevice)
{
	hwloc_obj_t osdev = NULL;
	int domain, bus, dev;

	if (hwloc_lwda_get_device_pci_ids(topology, lwdevice, &domain, &bus, &dev))
		return NULL;

	osdev = NULL;
	while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
		hwloc_obj_t pcidev = osdev->parent;
		if (strncmp(osdev->name, "lwca", 4))
			continue;
		if (pcidev
		    && pcidev->type == HWLOC_OBJ_PCI_DEVICE
		    && (int) pcidev->attr->pcidev.domain == domain
		    && (int) pcidev->attr->pcidev.bus == bus
		    && (int) pcidev->attr->pcidev.dev == dev
		    && pcidev->attr->pcidev.func == 0)
			return osdev;
		/* if PCI are filtered out, we need a info attr to match on */
	}

	return NULL;
}

/** \brief Get the hwloc OS device object corresponding to the
 * LWCA device whose index is \p idx.
 *
 * Return the OS device object describing the LWCA device whose
 * index is \p idx. Return NULL if there is none.
 *
 * The topology \p topology does not necessarily have to match the current
 * machine. For instance the topology may be an XML import of a remote host.
 * I/O devices detection and the LWCA component must be enabled in the topology.
 *
 * \note The corresponding PCI device object can be obtained by looking
 * at the OS device parent object (unless PCI devices are filtered out).
 *
 * \note This function is identical to hwloc_lwdart_get_device_osdev_by_index().
 */
static __hwloc_inline hwloc_obj_t
hwloc_lwda_get_device_osdev_by_index(hwloc_topology_t topology, unsigned idx)
{
	hwloc_obj_t osdev = NULL;
	while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
		if (HWLOC_OBJ_OSDEV_COPROC == osdev->attr->osdev.type
		    && osdev->name
		    && !strncmp("lwca", osdev->name, 4)
		    && atoi(osdev->name + 4) == (int) idx)
			return osdev;
	}
	return NULL;
}

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_LWDA_H */
