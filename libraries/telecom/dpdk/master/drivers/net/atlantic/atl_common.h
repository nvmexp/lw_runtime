/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2018 Aquantia Corporation
 */

#ifndef AQ_COMMON_H
#define AQ_COMMON_H

#define ATL_PMD_DRIVER_VERSION "0.6.7"

#define PCI_VENDOR_ID_AQUANTIA  0x1D6A

#define AQ_DEVICE_ID_0001	0x0001
#define AQ_DEVICE_ID_D100	0xD100
#define AQ_DEVICE_ID_D107	0xD107
#define AQ_DEVICE_ID_D108	0xD108
#define AQ_DEVICE_ID_D109	0xD109

#define AQ_DEVICE_ID_AQC100	0x00B1
#define AQ_DEVICE_ID_AQC107	0x07B1
#define AQ_DEVICE_ID_AQC108	0x08B1
#define AQ_DEVICE_ID_AQC109	0x09B1
#define AQ_DEVICE_ID_AQC111	0x11B1
#define AQ_DEVICE_ID_AQC112	0x12B1

#define AQ_DEVICE_ID_AQC100S	0x80B1
#define AQ_DEVICE_ID_AQC107S	0x87B1
#define AQ_DEVICE_ID_AQC108S	0x88B1
#define AQ_DEVICE_ID_AQC109S	0x89B1
#define AQ_DEVICE_ID_AQC111S	0x91B1
#define AQ_DEVICE_ID_AQC112S	0x92B1

#define AQ_DEVICE_ID_AQC111E	0x51B1
#define AQ_DEVICE_ID_AQC112E	0x52B1

#define HW_ATL_NIC_NAME "aQuantia AQtion 10Gbit Network Adapter"

#define AQ_HWREV_ANY	0
#define AQ_HWREV_1	1
#define AQ_HWREV_2	2

#define AQ_NIC_RATE_10G		BIT(0)
#define AQ_NIC_RATE_5G		BIT(1)
#define AQ_NIC_RATE_5G5R	BIT(2)
#define AQ_NIC_RATE_2G5		BIT(3)
#define AQ_NIC_RATE_1G		BIT(4)
#define AQ_NIC_RATE_100M	BIT(5)

#define AQ_NIC_RATE_EEE_10G	BIT(6)
#define AQ_NIC_RATE_EEE_5G	BIT(7)
#define AQ_NIC_RATE_EEE_2G5	BIT(8)
#define AQ_NIC_RATE_EEE_1G	BIT(9)


#define ATL_MAX_RING_DESC	(8 * 1024 - 8)
#define ATL_MIN_RING_DESC	32
#define ATL_RXD_ALIGN		8
#define ATL_TXD_ALIGN		8
#define ATL_TX_MAX_SEG		16

#define ATL_MAX_INTR_QUEUE_NUM  15

#define ATL_MISC_VEC_ID 10
#define ATL_RX_VEC_START 0

#define AQ_NIC_WOL_ENABLED           BIT(0)


#define AQ_NIC_FC_OFF    0U
#define AQ_NIC_FC_TX     1U
#define AQ_NIC_FC_RX     2U
#define AQ_NIC_FC_FULL   3U
#define AQ_NIC_FC_AUTO   4U


#define AQ_CFG_TX_FRAME_MAX  (16U * 1024U)
#define AQ_CFG_RX_FRAME_MAX  (2U * 1024U)

#define AQ_HW_MULTICAST_ADDRESS_MAX     32
#define AQ_HW_MAX_SEGS_SIZE    40

#define AQ_HW_MAX_RX_QUEUES    8
#define AQ_HW_MAX_TX_QUEUES    8
#define AQ_HW_MIN_RX_RING_SIZE 512
#define AQ_HW_MAX_RX_RING_SIZE 8192
#define AQ_HW_MIN_TX_RING_SIZE 512
#define AQ_HW_MAX_TX_RING_SIZE 8192

#define ATL_DEFAULT_RX_FREE_THRESH 64
#define ATL_DEFAULT_TX_FREE_THRESH 64

#define ATL_IRQ_CAUSE_LINK 0x8

#define AQ_HW_LED_BLINK    0x2U
#define AQ_HW_LED_DEFAULT  0x0U

#endif /* AQ_COMMON_H */
