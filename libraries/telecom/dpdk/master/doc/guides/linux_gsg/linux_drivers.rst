..  SPDX-License-Identifier: BSD-3-Clause
    Copyright(c) 2010-2015 Intel Corporation.
    Copyright 2017 Mellanox Technologies, Ltd
    All rights reserved.

.. _linux_gsg_linux_drivers:

Linux Drivers
=============

Different PMDs may require different kernel drivers in order to work properly.
Depending on the PMD being used, a corresponding kernel driver should be loaded,
and network ports should be bound to that driver.

VFIO
----

VFIO is a robust and secure driver that relies on IOMMU protection.
To make use of VFIO, the ``vfio-pci`` module must be loaded:

.. code-block:: console

    sudo modprobe vfio-pci

VFIO kernel is usually present by default in all distributions,
however please consult your distributions documentation to make sure that is the case.

Since Linux version 5.7,
the ``vfio-pci`` module supports the creation of virtual functions.
After the PF is bound to ``vfio-pci`` module,
the user can create the VFs using the ``sysfs`` interface,
and these VFs will be bound to ``vfio-pci`` module automatically.

When the PF is bound to ``vfio-pci``,
by default it will have a randomly generated VF token.
For security reasons, this token is write only,
so the user cannot read it from the kernel directly.
To access the VFs, the user needs to create a new token,
and use it to initialize both VF and PF devices.
The tokens are in UUID format,
so any UUID generation tool can be used to create a new token.

This VF token can be passed to DPDK by using EAL parameter ``--vfio-vf-token``.
The token will be used for all PF and VF ports within the application.

#. Generate the VF token by uuid command

   .. code-block:: console

      14d63f20-8445-11ea-8900-1f9ce7d5650d

#. Load the ``vfio-pci`` module with ``enable_sriov`` parameter set

   .. code-block:: console

      sudo modprobe vfio-pci enable_sriov=1

#. Bind the PCI devices to ``vfio-pci`` driver

   .. code-block:: console

      ./usertools/dpdk-devbind.py -b vfio-pci 0000:86:00.0

#. Create the desired number of VF devices

   .. code-block:: console

      echo 2 > /sys/bus/pci/devices/0000:86:00.0/sriov_numvfs

#. Start the DPDK application that will manage the PF device

   .. code-block:: console

      <build_dir>/app/dpdk-testpmd -l 22-25 -n 4 -a 86:00.0 \
      --vfio-vf-token=14d63f20-8445-11ea-8900-1f9ce7d5650d --file-prefix=pf -- -i

#. Start the DPDK application that will manage the VF device

   .. code-block:: console

      <build_dir>/app/dpdk-testpmd -l 26-29 -n 4 -a 86:02.0 \
      --vfio-vf-token=14d63f20-8445-11ea-8900-1f9ce7d5650d --file-prefix=vf0 -- -i

To make use of full VFIO functionality,
both kernel and BIOS must support and be configured
to use IO virtualization (such as Intel® VT-d).

.. note::

   Linux versions earlier than version 3.6 do not support VFIO.

.. note::

   Linux versions earlier than version 5.7 do not support the creation of
   virtual functions within the VFIO framework.

.. note::

   In most cases, specifying "iommu=on" as kernel parameter should be enough to
   configure the Linux kernel to use IOMMU.

For proper operation of VFIO when running DPDK applications as a non-privileged user, correct permissions should also be set up.
For more information, please refer to :ref:`Running_Without_Root_Privileges`.

VFIO no-IOMMU mode
------------------

If there is no IOMMU available on the system, VFIO can still be used,
but it has to be loaded with an additional module parameter:

.. code-block:: console

   modprobe vfio enable_unsafe_noiommu_mode=1

Alternatively, one can also enable this option in an already loaded kernel module:

.. code-block:: console

   echo 1 > /sys/module/vfio/parameters/enable_unsafe_noiommu_mode

After that, VFIO can be used with hardware devices as usual.

.. note::

   It may be required to unload all VFIO related-modules before probing
   the module again with ``enable_unsafe_noiommu_mode=1`` parameter.

.. warning::

   Since no-IOMMU mode forgoes IOMMU protection, it is inherently unsafe.
   That said, it does make it possible for the user
   to keep the degree of device access and programming that VFIO has,
   in situations where IOMMU is not available.

UIO
---

In situations where using VFIO is not an option, there are alternative drivers one can use.
In many cases, the standard ``uio_pci_generic`` module included in the Linux kernel
can be used as a substitute for VFIO. This module can be loaded using the command:

.. code-block:: console

   sudo modprobe uio_pci_generic

.. note::

   ``uio_pci_generic`` module doesn't support the creation of virtual functions.

As an alternative to the ``uio_pci_generic``, there is the ``igb_uio`` module
which can be found in the repository `dpdk-kmods <http://git.dpdk.org/dpdk-kmods>`_.
It can be loaded as shown below:

.. code-block:: console

   sudo modprobe uio
   sudo insmod igb_uio.ko

.. note::

   If UEFI secure boot is enabled,
   the Linux kernel may disallow the use of UIO on the system.
   Therefore, devices for use by DPDK should be bound to the ``vfio-pci`` kernel module
   rather than any UIO-based module.
   For more details see :ref:`linux_gsg_binding_kernel` below.

.. note::

   If the devices used for DPDK are bound to the ``uio_pci_generic`` kernel module,
   please make sure that the IOMMU is disabled or is in passthrough mode.
   One can add ``intel_iommu=off`` or ``amd_iommu=off`` or ``intel_iommu=on iommu=pt``
   in GRUB command line on x86_64 systems,
   or add ``iommu.passthrough=1`` on aarch64 systems.

.. note::

   Using UIO drivers is inherently unsafe due to this method lacking IOMMU protection,
   and can only be done by root user.

.. _bifurcated_driver:

Bifurcated Driver
-----------------

PMDs which use the bifurcated driver co-exists with the device kernel driver.
On such model the NIC is controlled by the kernel, while the data
path is performed by the PMD directly on top of the device.

Such model has the following benefits:

 - It is secure and robust, as the memory management and isolation
   is done by the kernel.
 - It enables the user to use legacy linux tools such as ``ethtool`` or
   ``ifconfig`` while running DPDK application on the same network ports.
 - It enables the DPDK application to filter only part of the traffic,
   while the rest will be directed and handled by the kernel driver.
   The flow bifurcation is performed by the NIC hardware.
   As an example, using :ref:`flow_isolated_mode` allows to choose
   strictly what is received in DPDK.

More about the bifurcated driver can be found in
`Mellanox Bifurcated DPDK PMD
<https://www.dpdk.org/wp-content/uploads/sites/35/2016/10/Day02-Session04-RonyEfraim-Userspace2016.pdf>`__.

.. _linux_gsg_binding_kernel:

Binding and Unbinding Network Ports to/from the Kernel Modules
--------------------------------------------------------------

.. note::

   PMDs which use the bifurcated driver should not be unbound from their kernel drivers.
   This section is for PMDs which use the UIO or VFIO drivers.

As of release 1.4, DPDK applications no longer automatically unbind all supported network ports from the kernel driver in use.
Instead, in case the PMD being used use the VFIO or UIO drivers,
all ports that are to be used by a DPDK application must be bound to
the ``vfio-pci``, ``uio_pci_generic``, or ``igb_uio`` module
before the application is run.
For such PMDs, any network ports under Linux* control will be ignored and cannot be used by the application.

To bind ports to the ``vfio-pci``, ``uio_pci_generic`` or ``igb_uio`` module
for DPDK use, or to return ports to Linux control,
a utility script called ``dpdk-devbind.py`` is provided in the ``usertools`` subdirectory.
This utility can be used to provide a view of the current state of the network ports on the system,
and to bind and unbind those ports from the different kernel modules,
including the VFIO and UIO modules.
The following are some examples of how the script can be used.
A full description of the script and its parameters can be obtained
by calling the script with the ``--help`` or ``--usage`` options.
Note that the UIO or VFIO kernel modules to be used,
should be loaded into the kernel before running the ``dpdk-devbind.py`` script.

.. warning::

   Due to the way VFIO works, there are certain limitations
   to which devices can be used with VFIO.
   Mainly it comes down to how IOMMU groups work.
   Any Virtual Function device can usually be used with VFIO on its own,
   but physical devices may require either all ports bound to VFIO,
   or some of them bound to VFIO while others not being bound to anything at all.

   If your device is behind a PCI-to-PCI bridge,
   the bridge will then be part of the IOMMU group in which your device is in.
   Therefore, the bridge driver should also be unbound from the bridge PCI device
   for VFIO to work with devices behind the bridge.

.. warning::

   While any user can run the ``dpdk-devbind.py`` script
   to view the status of the network ports,
   binding or unbinding network ports requires root privileges.

To see the status of all network ports on the system:

.. code-block:: console

    ./usertools/dpdk-devbind.py --status

    Network devices using DPDK-compatible driver
    ============================================
    0000:82:00.0 '82599EB 10-GbE NIC' drv=uio_pci_generic unused=ixgbe
    0000:82:00.1 '82599EB 10-GbE NIC' drv=uio_pci_generic unused=ixgbe

    Network devices using kernel driver
    ===================================
    0000:04:00.0 'I350 1-GbE NIC' if=em0  drv=igb unused=uio_pci_generic *Active*
    0000:04:00.1 'I350 1-GbE NIC' if=eth1 drv=igb unused=uio_pci_generic
    0000:04:00.2 'I350 1-GbE NIC' if=eth2 drv=igb unused=uio_pci_generic
    0000:04:00.3 'I350 1-GbE NIC' if=eth3 drv=igb unused=uio_pci_generic

    Other network devices
    =====================
    <none>

To bind device ``eth1``,``04:00.1``, to the ``uio_pci_generic`` driver:

.. code-block:: console

    ./usertools/dpdk-devbind.py --bind=uio_pci_generic 04:00.1

or, alternatively,

.. code-block:: console

    ./usertools/dpdk-devbind.py --bind=uio_pci_generic eth1

To restore device ``82:00.0`` to its original kernel binding:

.. code-block:: console

    ./usertools/dpdk-devbind.py --bind=ixgbe 82:00.0

Troubleshooting VFIO
--------------------

In certain situations, using ``dpdk-devbind.py`` script
to bind a device to VFIO driver may fail.
The first place to check is the kernel messages:

.. code-block:: console

   dmesg | tail
   ...
   [ 1297.875090] vfio-pci: probe of 0000:31:00.0 failed with error -22
   ...

In most cases, the ``error -22`` indicates that the VFIO subsystem
could not be enabled because there is no IOMMU support.

To check whether the kernel has been booted with correct parameters,
one can check the kernel command-line:

.. code-block:: console

   cat /proc/cmdline

Please refer to earlier sections on how to configure kernel parameters
correctly for your system.

If the kernel is configured correctly, one also has to make sure that
the BIOS configuration has virtualization features (such as Intel® VT-d).
There is no standard way to check if the platform is configured correctly,
so please check with your platform documentation to see if it has such features,
and how to enable them.

In certain distributions, default kernel configuration is such that
the no-IOMMU mode is disabled altogether at compile time.
This can be checked in the boot configuration of your system:

.. code-block:: console

   cat /boot/config-$(uname -r) | grep NOIOMMU
   # CONFIG_VFIO_NOIOMMU is not set

If ``CONFIG_VFIO_NOIOMMU`` is not enabled in the kernel configuration,
VFIO driver will not support the no-IOMMU mode,
and other alternatives (such as UIO drivers) will have to be used.
