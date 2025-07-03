/*
 * Copyright © 2012-2016 Inria.  All rights reserved.
 * See COPYING in top-level directory.
 */

/** \file
 * \brief Macros to help interaction between hwloc and the LWPU Management Library.
 *
 * Applications that use both hwloc and the LWPU Management Library may want to
 * include this file so as to get topology information for LWML devices.
 */

#ifndef HWLOC_LWML_H
#define HWLOC_LWML_H

#include <hwloc.h>
#include <hwloc/autogen/config.h>
#include <hwloc/helper.h>
#ifdef HWLOC_LINUX_SYS
#include <hwloc/linux.h>
#endif

#include <lwml.h>


#ifdef __cplusplus
extern "C" {
#endif


/** \defgroup hwlocality_lwml Interoperability with the LWPU Management Library
 *
 * This interface offers ways to retrieve topology information about
 * devices managed by the LWPU Management Library (LWML).
 *
 * @{
 */

/** \brief Get the CPU set of logical processors that are physically
 * close to LWML device \p device.
 *
 * Return the CPU set describing the locality of the LWML device \p device.
 *
 * Topology \p topology and device \p device must match the local machine.
 * I/O devices detection and the LWML component are not needed in the topology.
 *
 * The function only returns the locality of the device.
 * If more information about the device is needed, OS objects should
 * be used instead, see hwloc_lwml_get_device_osdev()
 * and hwloc_lwml_get_device_osdev_by_index().
 *
 * This function is lwrrently only implemented in a meaningful way for
 * Linux; other systems will simply get a full cpuset.
 */
static __hwloc_inline int
hwloc_lwml_get_device_cpuset(hwloc_topology_t topology __hwloc_attribute_unused,
			     lwmlDevice_t device, hwloc_cpuset_t set)
{
#ifdef HWLOC_LINUX_SYS
  /* If we're on Linux, use the sysfs mechanism to get the local cpus */
#define HWLOC_LWML_DEVICE_SYSFS_PATH_MAX 128
  char path[HWLOC_LWML_DEVICE_SYSFS_PATH_MAX];
  lwmlReturn_t lwres;
  lwmlPciInfo_t pci;

  if (!hwloc_topology_is_thissystem(topology)) {
    errno = EILWAL;
    return -1;
  }

  lwres = lwmlDeviceGetPciInfo(device, &pci);
  if (LWML_SUCCESS != lwres) {
    errno = EILWAL;
    return -1;
  }

  sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.0/local_cpus", pci.domain, pci.bus, pci.device);
  if (hwloc_linux_read_path_as_cpumask(path, set) < 0
      || hwloc_bitmap_iszero(set))
    hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#else
  /* Non-Linux systems simply get a full cpuset */
  hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
#endif
  return 0;
}

/** \brief Get the hwloc OS device object corresponding to the
 * LWML device whose index is \p idx.
 *
 * Return the OS device object describing the LWML device whose
 * index is \p idx. Returns NULL if there is none.
 *
 * The topology \p topology does not necessarily have to match the current
 * machine. For instance the topology may be an XML import of a remote host.
 * I/O devices detection and the LWML component must be enabled in the topology.
 *
 * \note The corresponding PCI device object can be obtained by looking
 * at the OS device parent object (unless PCI devices are filtered out).
 */
static __hwloc_inline hwloc_obj_t
hwloc_lwml_get_device_osdev_by_index(hwloc_topology_t topology, unsigned idx)
{
	hwloc_obj_t osdev = NULL;
	while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
                if (HWLOC_OBJ_OSDEV_GPU == osdev->attr->osdev.type
                    && osdev->name
		    && !strncmp("lwml", osdev->name, 4)
		    && atoi(osdev->name + 4) == (int) idx)
                        return osdev;
        }
        return NULL;
}

/** \brief Get the hwloc OS device object corresponding to LWML device \p device.
 *
 * Return the hwloc OS device object that describes the given
 * LWML device \p device. Return NULL if there is none.
 *
 * Topology \p topology and device \p device must match the local machine.
 * I/O devices detection and the LWML component must be enabled in the topology.
 * If not, the locality of the object may still be found using
 * hwloc_lwml_get_device_cpuset().
 *
 * \note The corresponding hwloc PCI device may be found by looking
 * at the result parent pointer (unless PCI devices are filtered out).
 */
static __hwloc_inline hwloc_obj_t
hwloc_lwml_get_device_osdev(hwloc_topology_t topology, lwmlDevice_t device)
{
	hwloc_obj_t osdev;
	lwmlReturn_t lwres;
	lwmlPciInfo_t pci;
	char uuid[64];

	if (!hwloc_topology_is_thissystem(topology)) {
		errno = EILWAL;
		return NULL;
	}

	lwres = lwmlDeviceGetPciInfo(device, &pci);
	if (LWML_SUCCESS != lwres)
		return NULL;

	lwres = lwmlDeviceGetUUID(device, uuid, sizeof(uuid));
	if (LWML_SUCCESS != lwres)
		uuid[0] = '\0';

	osdev = NULL;
	while ((osdev = hwloc_get_next_osdev(topology, osdev)) != NULL) {
		hwloc_obj_t pcidev = osdev->parent;
		const char *info;

		if (strncmp(osdev->name, "lwml", 4))
			continue;

		if (pcidev
		    && pcidev->type == HWLOC_OBJ_PCI_DEVICE
		    && pcidev->attr->pcidev.domain == pci.domain
		    && pcidev->attr->pcidev.bus == pci.bus
		    && pcidev->attr->pcidev.dev == pci.device
		    && pcidev->attr->pcidev.func == 0)
			return osdev;

		info = hwloc_obj_get_info_by_name(osdev, "LWIDIAUUID");
		if (info && !strcmp(info, uuid))
			return osdev;
	}

	return NULL;
}

/** @} */


#ifdef __cplusplus
} /* extern "C" */
#endif


#endif /* HWLOC_LWML_H */
