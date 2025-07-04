/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2001-2020 Intel Corporation
 */

#include "igc_hw.h"
#include "igc_i225.h"
#include "igc_mac.h"
#include "igc_base.h"
#include "igc_manage.h"

/**
 *  igc_acquire_phy_base - Acquire rights to access PHY
 *  @hw: pointer to the HW structure
 *
 *  Acquire access rights to the correct PHY.
 **/
s32 igc_acquire_phy_base(struct igc_hw *hw)
{
	u16 mask = IGC_SWFW_PHY0_SM;

	DEBUGFUNC("igc_acquire_phy_base");

	if (hw->bus.func == IGC_FUNC_1)
		mask = IGC_SWFW_PHY1_SM;
	else if (hw->bus.func == IGC_FUNC_2)
		mask = IGC_SWFW_PHY2_SM;
	else if (hw->bus.func == IGC_FUNC_3)
		mask = IGC_SWFW_PHY3_SM;

	return hw->mac.ops.acquire_swfw_sync(hw, mask);
}

/**
 *  igc_release_phy_base - Release rights to access PHY
 *  @hw: pointer to the HW structure
 *
 *  A wrapper to release access rights to the correct PHY.
 **/
void igc_release_phy_base(struct igc_hw *hw)
{
	u16 mask = IGC_SWFW_PHY0_SM;

	DEBUGFUNC("igc_release_phy_base");

	if (hw->bus.func == IGC_FUNC_1)
		mask = IGC_SWFW_PHY1_SM;
	else if (hw->bus.func == IGC_FUNC_2)
		mask = IGC_SWFW_PHY2_SM;
	else if (hw->bus.func == IGC_FUNC_3)
		mask = IGC_SWFW_PHY3_SM;

	hw->mac.ops.release_swfw_sync(hw, mask);
}

/**
 *  igc_init_hw_base - Initialize hardware
 *  @hw: pointer to the HW structure
 *
 *  This inits the hardware readying it for operation.
 **/
s32 igc_init_hw_base(struct igc_hw *hw)
{
	struct igc_mac_info *mac = &hw->mac;
	s32 ret_val;
	u16 i, rar_count = mac->rar_entry_count;

	DEBUGFUNC("igc_init_hw_base");

	/* Setup the receive address */
	igc_init_rx_addrs_generic(hw, rar_count);

	/* Zero out the Multicast HASH table */
	DEBUGOUT("Zeroing the MTA\n");
	for (i = 0; i < mac->mta_reg_count; i++)
		IGC_WRITE_REG_ARRAY(hw, IGC_MTA, i, 0);

	/* Zero out the Unicast HASH table */
	DEBUGOUT("Zeroing the UTA\n");
	for (i = 0; i < mac->uta_reg_count; i++)
		IGC_WRITE_REG_ARRAY(hw, IGC_UTA, i, 0);

	/* Setup link and flow control */
	ret_val = mac->ops.setup_link(hw);
	/*
	 * Clear all of the statistics registers (clear on read).  It is
	 * important that we do this after we have tried to establish link
	 * because the symbol error count will increment wildly if there
	 * is no link.
	 */
	igc_clear_hw_cntrs_base_generic(hw);

	return ret_val;
}

/**
 * igc_power_down_phy_copper_base - Remove link during PHY power down
 * @hw: pointer to the HW structure
 *
 * In the case of a PHY power down to save power, or to turn off link during a
 * driver unload, or wake on lan is not enabled, remove the link.
 **/
void igc_power_down_phy_copper_base(struct igc_hw *hw)
{
	struct igc_phy_info *phy = &hw->phy;

	if (!(phy->ops.check_reset_block))
		return;

	/* If the management interface is not enabled, then power down */
	if (!phy->ops.check_reset_block(hw))
		igc_power_down_phy_copper(hw);
}

/**
 *  igc_rx_fifo_flush_base - Clean Rx FIFO after Rx enable
 *  @hw: pointer to the HW structure
 *
 *  After Rx enable, if manageability is enabled then there is likely some
 *  bad data at the start of the FIFO and possibly in the DMA FIFO.  This
 *  function clears the FIFOs and flushes any packets that came in as Rx was
 *  being enabled.
 **/
void igc_rx_fifo_flush_base(struct igc_hw *hw)
{
	u32 rctl, rlpml, rxdctl[4], rfctl, temp_rctl, rx_enabled;
	int i, ms_wait;

	DEBUGFUNC("igc_rx_fifo_flush_base");

	/* disable IPv6 options as per hardware errata */
	rfctl = IGC_READ_REG(hw, IGC_RFCTL);
	rfctl |= IGC_RFCTL_IPV6_EX_DIS;
	IGC_WRITE_REG(hw, IGC_RFCTL, rfctl);

	if (!(IGC_READ_REG(hw, IGC_MANC) & IGC_MANC_RCV_TCO_EN))
		return;

	/* Disable all Rx queues */
	for (i = 0; i < 4; i++) {
		rxdctl[i] = IGC_READ_REG(hw, IGC_RXDCTL(i));
		IGC_WRITE_REG(hw, IGC_RXDCTL(i),
				rxdctl[i] & ~IGC_RXDCTL_QUEUE_ENABLE);
	}
	/* Poll all queues to verify they have shut down */
	for (ms_wait = 0; ms_wait < 10; ms_wait++) {
		msec_delay(1);
		rx_enabled = 0;
		for (i = 0; i < 4; i++)
			rx_enabled |= IGC_READ_REG(hw, IGC_RXDCTL(i));
		if (!(rx_enabled & IGC_RXDCTL_QUEUE_ENABLE))
			break;
	}

	if (ms_wait == 10)
		DEBUGOUT("Queue disable timed out after 10ms\n");

	/* Clear RLPML, RCTL.SBP, RFCTL.LEF, and set RCTL.LPE so that all
	 * incoming packets are rejected.  Set enable and wait 2ms so that
	 * any packet that was coming in as RCTL.EN was set is flushed
	 */
	IGC_WRITE_REG(hw, IGC_RFCTL, rfctl & ~IGC_RFCTL_LEF);

	rlpml = IGC_READ_REG(hw, IGC_RLPML);
	IGC_WRITE_REG(hw, IGC_RLPML, 0);

	rctl = IGC_READ_REG(hw, IGC_RCTL);
	temp_rctl = rctl & ~(IGC_RCTL_EN | IGC_RCTL_SBP);
	temp_rctl |= IGC_RCTL_LPE;

	IGC_WRITE_REG(hw, IGC_RCTL, temp_rctl);
	IGC_WRITE_REG(hw, IGC_RCTL, temp_rctl | IGC_RCTL_EN);
	IGC_WRITE_FLUSH(hw);
	msec_delay(2);

	/* Enable Rx queues that were previously enabled and restore our
	 * previous state
	 */
	for (i = 0; i < 4; i++)
		IGC_WRITE_REG(hw, IGC_RXDCTL(i), rxdctl[i]);
	IGC_WRITE_REG(hw, IGC_RCTL, rctl);
	IGC_WRITE_FLUSH(hw);

	IGC_WRITE_REG(hw, IGC_RLPML, rlpml);
	IGC_WRITE_REG(hw, IGC_RFCTL, rfctl);

	/* Flush receive errors generated by workaround */
	IGC_READ_REG(hw, IGC_ROC);
	IGC_READ_REG(hw, IGC_RNBC);
	IGC_READ_REG(hw, IGC_MPC);
}
