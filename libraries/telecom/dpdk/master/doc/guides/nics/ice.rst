..  SPDX-License-Identifier: BSD-3-Clause
    Copyright(c) 2018 Intel Corporation.

ICE Poll Mode Driver
======================

The ice PMD (**librte_net_ice**) provides poll mode driver support for
10/25/50/100 Gbps Intel® Ethernet 800 Series Network Adapters based on
the Intel Ethernet Controller E810 and Intel Ethernet Connection E822/E823.

Prerequisites
-------------

- Follow the DPDK :ref:`Getting Started Guide for Linux <linux_gsg>` to setup the basic DPDK environment.

- To get better performance on Intel platforms, please follow the "How to get best performance with NICs on Intel platforms"
  section of the :ref:`Getting Started Guide for Linux <linux_gsg>`.

- Please follow the matching list to download specific kernel driver, firmware and DDP package from
  `https://www.intel.com/content/www/us/en/search.html?ws=text#q=e810&t=Downloads&layout=table`.

- To understand what is DDP package and how it works, please review `Intel® Ethernet Controller E810 Dynamic
  Device Personalization (DDP) for Telecommunications Technology Guide <https://cdrdv2.intel.com/v1/dl/getContent/617015>`_.

- To understand DDP for COMMs usage with DPDK, please review `Intel® Ethernet 800 Series Telecommunication (Comms)
  Dynamic Device Personalization (DDP) Package <https://cdrdv2.intel.com/v1/dl/getContent/618651>`_.


Recommended Matching List
-------------------------

It is highly recommended to upgrade the ice kernel driver, firmware and DDP package
to avoid the compatibility issues with ice PMD.
Here is the suggested matching list which has been tested and verified.
The detailed information can refer to chapter Tested Platforms/Tested NICs in release notes.

   +-----------+---------------+-----------------+-----------+-----------+
   |    DPDK   | Kernel Driver | OS Default DDP  | COMMS DDP | Firmware  |
   +===========+===============+=================+===========+===========+
   |    20.11  |     1.3.0     |      1.3.20     |  1.3.24   |    2.3    |
   +-----------+---------------+-----------------+-----------+-----------+

Pre-Installation Configuration
------------------------------


Runtime Config Options
~~~~~~~~~~~~~~~~~~~~~~

- ``Safe Mode Support`` (default ``0``)

  If driver failed to load OS package, by default driver's initialization failed.
  But if user intend to use the device without OS package, user can take ``devargs``
  parameter ``safe-mode-support``, for example::

    -a 80:00.0,safe-mode-support=1

  Then the driver will be initialized successfully and the device will enter Safe Mode.
  NOTE: In Safe mode, only very limited features are available, features like RSS,
  checksum, fdir, tunneling ... are all disabled.

- ``Generic Flow Pipeline Mode Support`` (default ``0``)

  In pipeline mode, a flow can be set at one specific stage by setting parameter
  ``priority``. Lwrrently, we support two stages: priority = 0 or !0. Flows with
  priority 0 located at the first pipeline stage which typically be used as a firewall
  to drop the packet on a blocklist(we called it permission stage). At this stage,
  flow rules are created for the device's exact match engine: switch. Flows with priority
  !0 located at the second stage, typically packets are classified here and be steered to
  specific queue or queue group (we called it distribution stage), At this stage, flow
  rules are created for device's flow director engine.
  For none-pipeline mode, ``priority`` is ignored, a flow rule can be created as a flow director
  rule or a switch rule depends on its pattern/action and the resource allocation situation,
  all flows are virtually at the same pipeline stage.
  By default, generic flow API is enabled in none-pipeline mode, user can choose to
  use pipeline mode by setting ``devargs`` parameter ``pipeline-mode-support``,
  for example::

    -a 80:00.0,pipeline-mode-support=1

- ``Protocol extraction for per queue``

  Configure the RX queues to do protocol extraction into mbuf for protocol
  handling acceleration, like checking the TCP SYN packets quickly.

  The argument format is::

      -a 18:00.0,proto_xtr=<queues:protocol>[<queues:protocol>...]
      -a 18:00.0,proto_xtr=<protocol>

  Queues are grouped by ``(`` and ``)`` within the group. The ``-`` character
  is used as a range separator and ``,`` is used as a single number separator.
  The grouping ``()`` can be omitted for single element group. If no queues are
  specified, PMD will use this protocol extraction type for all queues.

  Protocol is : ``vlan, ipv4, ipv6, ipv6_flow, tcp, ip_offset``.

  .. code-block:: console

    dpdk-testpmd -a 18:00.0,proto_xtr='[(1,2-3,8-9):tcp,10-13:vlan]'

  This setting means queues 1, 2-3, 8-9 are TCP extraction, queues 10-13 are
  VLAN extraction, other queues run with no protocol extraction.

  .. code-block:: console

    dpdk-testpmd -a 18:00.0,proto_xtr=vlan,proto_xtr='[(1,2-3,8-9):tcp,10-23:ipv6]'

  This setting means queues 1, 2-3, 8-9 are TCP extraction, queues 10-23 are
  IPv6 extraction, other queues use the default VLAN extraction.

  The extraction metadata is copied into the registered dynamic mbuf field, and
  the related dynamic mbuf flags is set.

  .. table:: Protocol extraction : ``vlan``

   +----------------------------+----------------------------+
   |           VLAN2            |           VLAN1            |
   +======+===+=================+======+===+=================+
   |  PCP | D |       VID       |  PCP | D |       VID       |
   +------+---+-----------------+------+---+-----------------+

  VLAN1 - single or EVLAN (first for QinQ).

  VLAN2 - C-VLAN (second for QinQ).

  .. table:: Protocol extraction : ``ipv4``

   +----------------------------+----------------------------+
   |           IPHDR2           |           IPHDR1           |
   +======+=======+=============+==============+=============+
   |  Ver |Hdr Len|    ToS      |      TTL     |  Protocol   |
   +------+-------+-------------+--------------+-------------+

  IPHDR1 - IPv4 header word 4, "TTL" and "Protocol" fields.

  IPHDR2 - IPv4 header word 0, "Ver", "Hdr Len" and "Type of Service" fields.

  .. table:: Protocol extraction : ``ipv6``

   +----------------------------+----------------------------+
   |           IPHDR2           |           IPHDR1           |
   +=====+=============+========+=============+==============+
   | Ver |Traffic class|  Flow  | Next Header |   Hop Limit  |
   +-----+-------------+--------+-------------+--------------+

  IPHDR1 - IPv6 header word 3, "Next Header" and "Hop Limit" fields.

  IPHDR2 - IPv6 header word 0, "Ver", "Traffic class" and high 4 bits of
  "Flow Label" fields.

  .. table:: Protocol extraction : ``ipv6_flow``

   +----------------------------+----------------------------+
   |           IPHDR2           |           IPHDR1           |
   +=====+=============+========+============================+
   | Ver |Traffic class|            Flow Label               |
   +-----+-------------+-------------------------------------+

  IPHDR1 - IPv6 header word 1, 16 low bits of the "Flow Label" field.

  IPHDR2 - IPv6 header word 0, "Ver", "Traffic class" and high 4 bits of
  "Flow Label" fields.

  .. table:: Protocol extraction : ``tcp``

   +----------------------------+----------------------------+
   |           TCPHDR2          |           TCPHDR1          |
   +============================+======+======+==============+
   |          Reserved          |Offset|  RSV |     Flags    |
   +----------------------------+------+------+--------------+

  TCPHDR1 - TCP header word 6, "Data Offset" and "Flags" fields.

  TCPHDR2 - Reserved

  .. table:: Protocol extraction : ``ip_offset``

   +----------------------------+----------------------------+
   |           IPHDR2           |           IPHDR1           |
   +============================+============================+
   |       IPv6 HDR Offset      |       IPv4 HDR Offset      |
   +----------------------------+----------------------------+

  IPHDR1 - Outer/Single IPv4 Header offset.

  IPHDR2 - Outer/Single IPv6 Header offset.

  Use ``rte_net_ice_dynf_proto_xtr_metadata_get`` to access the protocol
  extraction metadata, and use ``RTE_PKT_RX_DYNF_PROTO_XTR_*`` to get the
  metadata type of ``struct rte_mbuf::ol_flags``.

  The ``rte_net_ice_dump_proto_xtr_metadata`` routine shows how to
  access the protocol extraction result in ``struct rte_mbuf``.

Driver compilation and testing
------------------------------

Refer to the document :ref:`compiling and testing a PMD for a NIC <pmd_build_and_test>`
for details.

Features
--------

Vector PMD
~~~~~~~~~~

Vector PMD for RX and TX path are selected automatically. The paths
are chosen based on 2 conditions.

- ``CPU``
  On the X86 platform, the driver checks if the CPU supports AVX2.
  If it's supported, AVX2 paths will be chosen. If not, SSE is chosen.

- ``Offload features``
  The supported HW offload features are described in the document ice_vec.ini.
  If any not supported features are used, ICE vector PMD is disabled and the
  normal paths are chosen.

Malicious driver detection (MDD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It's not appropriate to send a packet, if this packet's destination MAC address
is just this port's MAC address. If SW tries to send such packets, HW will
report a MDD event and drop the packets.

The APPs based on DPDK should avoid providing such packets.

Device Config Function (DCF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section demonstrates ICE DCF PMD, which shares the core module with ICE
PMD and iAVF PMD.

A DCF (Device Config Function) PMD bounds to the device's trusted VF with ID 0,
it can act as a sole controlling entity to exercise advance functionality (such
as switch, ACL) for the rest VFs.

The DCF PMD needs to advertise and acquire DCF capability which allows DCF to
send AdminQ commands that it would like to execute over to the PF and receive
responses for the same from PF.

.. _figure_ice_dcf:

.. figure:: img/ice_dcf.*

   DCF Communication flow.

#. Create the VFs::

      echo 4 > /sys/bus/pci/devices/0000\:18\:00.0/sriov_numvfs

#. Enable the VF0 trust on::

      ip link set dev enp24s0f0 vf 0 trust on

#. Bind the VF0,  and run testpmd with 'cap=dcf' devarg::

      dpdk-testpmd -l 22-25 -n 4 -a 18:01.0,cap=dcf -- -i

#. Monitor the VF2 interface network traffic::

      tcpdump -e -nn -i enp24s1f2

#. Create one flow to redirect the traffic to VF2 by DCF::

      flow create 0 priority 0 ingress pattern eth / ipv4 src is 192.168.0.2 \
      dst is 192.168.0.3 / end actions vf id 2 / end

#. Send the packet, and it should be displayed on tcpdump::

      sendp(Ether(src='3c:fd:fe:aa:bb:78', dst='00:00:00:01:02:03')/IP(src=' \
      192.168.0.2', dst="192.168.0.3")/TCP(flags='S')/Raw(load='XXXXXXXXXX'), \
      iface="enp24s0f0", count=10)

Sample Application Notes
------------------------

Vlan filter
~~~~~~~~~~~

Vlan filter only works when Promislwous mode is off.

To start ``testpmd``, and add vlan 10 to port 0:

.. code-block:: console

    ./app/dpdk-testpmd -l 0-15 -n 4 -- -i
    ...

    testpmd> rx_vlan add 10 0

Limitations or Known issues
---------------------------

The Intel E810 requires a programmable pipeline package be downloaded
by the driver to support normal operations. The E810 has a limited
functionality built in to allow PXE boot and other use cases, but the
driver must download a package file during the driver initialization
stage.

The default DDP package file name is ice.pkg. For a specific NIC, the
DDP package supposed to be loaded can have a filename: ice-xxxxxx.pkg,
where 'xxxxxx' is the 64-bit PCIe Device Serial Number of the NIC. For
example, if the NIC's device serial number is 00-CC-BB-FF-FF-AA-05-68,
the device-specific DDP package filename is ice-00ccbbffffaa0568.pkg
(in hex and all low case). During initialization, the driver searches
in the following paths in order: /lib/firmware/updates/intel/ice/ddp
and /lib/firmware/intel/ice/ddp. The corresponding device-specific DDP
package will be downloaded first if the file exists. If not, then the
driver tries to load the default package. The type of loaded package
is stored in ``ice_adapter->active_pkg_type``.

A symbolic link to the DDP package file is also ok. The same package
file is used by both the kernel driver and the DPDK PMD.
