..  SPDX-License-Identifier: BSD-3-Clause
    Copyright(c) 2017 Cavium, Inc

OCTEON TX SSOVF Eventdev Driver
===============================

The OCTEON TX SSOVF PMD (**librte_event_octeontx**) provides poll mode
eventdev driver support for the inbuilt event device found in the **Cavium OCTEON TX**
SoC family as well as their virtual functions (VF) in SR-IOV context.

More information can be found at `Cavium, Inc Official Website
<http://www.cavium.com/OCTEON-TX_ARM_Processors.html>`_.

Features
--------

Features of the OCTEON TX SSOVF PMD are:

- 64 Event queues
- 32 Event ports
- HW event scheduler
- Supports 1M flows per event queue
- Flow based event pipelining
- Flow pinning support in flow based event pipelining
- Queue based event pipelining
- Supports ATOMIC, ORDERED, PARALLEL schedule types per flow
- Event scheduling QoS based on event queue priority
- Open system with configurable amount of outstanding events
- HW accelerated dequeue timeout support to enable power management
- SR-IOV VF
- HW managed event timers support through TIMVF, with high precision and
  time granularity of 1us.
- Up to 64 event timer adapters.

Supported OCTEON TX SoCs
------------------------
- CN83xx

Prerequisites
-------------

See :doc:`../platform/octeontx` for setup information.


Initialization
--------------

The OCTEON TX eventdev is exposed as a vdev device which consists of a set
of SSO group and work-slot PCIe VF devices. On EAL initialization,
SSO PCIe VF devices will be probed and then the vdev device can be created
from the application code, or from the EAL command line based on
the number of probed/bound SSO PCIe VF device to DPDK by

* Ilwoking ``rte_vdev_init("event_octeontx")`` from the application

* Using ``--vdev="event_octeontx"`` in the EAL options, which will call
  rte_vdev_init() internally

Example:

.. code-block:: console

    ./your_eventdev_application --vdev="event_octeontx"


Enable TIMvf stats
------------------
TIMvf stats can be enabled by using this option, by default the stats are
disabled.

.. code-block:: console

    --vdev="event_octeontx,timvf_stats=1"


Limitations
-----------

Burst mode support
~~~~~~~~~~~~~~~~~~

Burst mode is not supported. Dequeue and Enqueue functions accepts only single
event at a time.

Rx adapter support
~~~~~~~~~~~~~~~~~~

When eth_octeontx is used as Rx adapter event schedule type
``RTE_SCHED_TYPE_PARALLEL`` is not supported.

Event timer adapter support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When timvf is used as Event timer adapter the clock source mapping is as
follows:

.. code-block:: console

        RTE_EVENT_TIMER_ADAPTER_CPU_CLK  = TIM_CLK_SRC_SCLK
        RTE_EVENT_TIMER_ADAPTER_EXT_CLK0 = TIM_CLK_SRC_GPIO
        RTE_EVENT_TIMER_ADAPTER_EXT_CLK1 = TIM_CLK_SRC_GTI
        RTE_EVENT_TIMER_ADAPTER_EXT_CLK2 = TIM_CLK_SRC_PTP

When timvf is used as Event timer adapter event schedule type
``RTE_SCHED_TYPE_PARALLEL`` is not supported.

Max number of events
~~~~~~~~~~~~~~~~~~~~

Max number of events in OCTEON TX Eventdev (SSO) are only limited by DRAM size
and they can be configured by passing limits to kernel bootargs as follows:

.. code-block:: console

        ssopf.max_events=4194304

The same can be verified by looking at the following sysfs entry:

.. code-block:: console

        # cat /sys/module/ssopf/parameters/max_events
        4194304

The maximum number of events that can be added to SSO by the event adapters such
as (Rx/Timer) should be limited to the above configured value.
