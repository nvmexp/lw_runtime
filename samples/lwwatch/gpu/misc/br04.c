/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2001-2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwwatch.h"
#include "os.h"

//
// lw includes
//
#include "print.h"
#include "hal.h"
#include "fifo.h"
#include "diags.h"
#include "inst.h"
#include "dcb.h"
#include "gr.h"
#include "heap.h"
#include "i2c.h"
#include "exts.h"
#include "mmu.h"
#include "methodParse.h"
#include "lwwatch_pex.h"
#include "br04.h"

//
// Globals
//
LwU32 lwNumBR04s;
PLWWATCHTOPOLOGYNODESTRUCT lwBR04s[128] = { NULL };
PLWWATCHTOPOLOGYNODESTRUCT TopologyRoot = NULL;


static void
ClearTopologyNode
(
    PLWWATCHTOPOLOGYNODESTRUCT node
)
{
    LwU32 i;

    if (node == NULL)
        return;

    for (i = TOPOLOGY_NUM_DOWNSTREAM; i > 0; --i)
        ClearTopologyNode(node->down[i-1]);

    free(node);
}

static void
AddDeviceToTopologyNode
(
    PLWWATCHTOPOLOGYNODESTRUCT dest,
    PLWWATCHTOPOLOGYNODESTRUCT node
)
{
    int i;

    //
    // Check to see if node's PrimaryBus falls within the range of one of
    // dest's downstream nodes.  If so, add to the downstream node.
    //
    for (i = 0; i < TOPOLOGY_NUM_DOWNSTREAM; ++i)
    {
        if (dest->down[i] != NULL)
        {
            PLWWATCHTOPOLOGYNODESTRUCT TargetNode = dest->down[i];
            if (node->PrimaryBus >= TargetNode->SecondaryBus &&
              node->PrimaryBus <= TargetNode->SubordinateBus)
            {
                AddDeviceToTopologyNode(TargetNode, node);
                return;
            }
        }
    }

    //
    // Find an empty spot where we can place this node.
    //
    for (i = 0; i < TOPOLOGY_NUM_DOWNSTREAM; ++i)
    {
        if (dest->down[i] == NULL)
        {
            dest->down[i] = node;
            node->up = dest;
            return;
        }
    }

    dprintf("Out of bus slots adding device '%s' to topology hierarchy\n", node->name);
}

static void
PrintTopologyNode
(
    PLWWATCHTOPOLOGYNODESTRUCT node,
    LwU32 Level
)
{
    LwU32 i, j, children = 0;

    if (node == NULL)
        return;

    if (Level > 0)
    {
        dprintf("|");
        for (i = 1; i < Level; ++i)
            dprintf("  |");
    }

    dprintf("%s%s\n", (Level > 0) ? "--" : "", node->name);

    for (i = 0; i < TOPOLOGY_NUM_DOWNSTREAM; ++i)
    {
        if (node->down[i])
            ++children;
    }

    if (children > 0)
    {
        dprintf("|");
        for (i = 1; i <= Level; ++i)
            dprintf("  |");
        dprintf("\n");
    }


    for (i = 0; i < TOPOLOGY_NUM_DOWNSTREAM && children > 0; ++i)
    {
        PrintTopologyNode(node->down[i], Level+1);
        --children;
        if (children > 0)
        {
            for (j = 0; j <= Level; ++j)
                dprintf("|  ");
            if (j > 0)
                dprintf("\n");
        }
    }
}

static void
PrintTopology
(
    void
)
{
    dprintf("\n");
    PrintTopologyNode(TopologyRoot, 0);
}

static void
DumpCredits
(
    PhysAddr Address
)
{
    LwU32 reg = GPU_REG_RD32(Address);
    LwU32 Header = DRF_VAL(_BR04_XVU, _INT_FLOW_CTL, _DP0_TOO_CPL_H_SIGMA, reg);
    LwU32 Data = DRF_VAL(_BR04_XVU, _INT_FLOW_CTL, _DP0_TOO_CPL_D_SIGMA, reg);

    dprintf("        CPL Headers: %d    CPL Data: %d\n",  Header, Data);

    reg = GPU_REG_RD32(Address + 4);
    Header = DRF_VAL(_BR04_XVU, _INT_FLOW_CTL, _DP0_TOO_NP_H_SIGMA, reg);
    Data = DRF_VAL(_BR04_XVU, _INT_FLOW_CTL, _DP0_TOO_NP_D_SIGMA, reg);

    dprintf("         NP Headers: %d     NP Data: %d\n", Header, Data);

    reg = GPU_REG_RD32(Address + 8);
    Header = DRF_VAL(_BR04_XVU, _INT_FLOW_CTL, _DP0_TOO_PW_H_SIGMA, reg);
    Data = DRF_VAL(_BR04_XVU, _INT_FLOW_CTL, _DP0_TOO_PW_D_SIGMA, reg);

    dprintf("         PW Headers: %d     PW Data: %d\n", Header, Data);
}


//
// global functions
//

void
br04ClearTopology
(
    void
)
{
    ClearTopologyNode(TopologyRoot);
    TopologyRoot = NULL;
    lwNumBR04s = 0;
}

PLWWATCHTOPOLOGYNODESTRUCT
br04AddDeviceToTopology
(
    const char *name,
    LwU32 Bar0,
    LwU32 PrimaryBus,
    LwU32 SecondaryBus,
    LwU32 SubordinateBus,
    LwU32 PortID
)
{
    LwU32 i;
    PLWWATCHTOPOLOGYNODESTRUCT tmp = NULL;

    if (TopologyRoot == NULL)
    {
        TopologyRoot = (PLWWATCHTOPOLOGYNODESTRUCT) malloc(sizeof(LWWATCHTOPOLOGYNODESTRUCT));

        if (TopologyRoot == NULL)
        {
            dprintf("Out of memory creating topology hierarchy\n");
            return NULL;
        }

        strcpy(TopologyRoot->name, "PCI Express Bus");
        TopologyRoot->Bar0 = 0;
        TopologyRoot->PrimaryBus = 0xffffffff;
        TopologyRoot->SecondaryBus = 0;
        TopologyRoot->SubordinateBus = 0xffffffff;
        TopologyRoot->PortID = 0xffffffff;
        TopologyRoot->up = NULL;
    
        for (i = 0; i < TOPOLOGY_NUM_DOWNSTREAM; ++i)
            TopologyRoot->down[i] = NULL;
    }

    tmp = (PLWWATCHTOPOLOGYNODESTRUCT) malloc(sizeof(LWWATCHTOPOLOGYNODESTRUCT));

    if (tmp == NULL)
    {
        br04ClearTopology();
        dprintf("Out of memory adding device to topology hierarchy\n");
        return NULL;
    }

    strcpy(tmp->name, name);
    tmp->Bar0 = Bar0;
    tmp->PrimaryBus = PrimaryBus;
    tmp->SecondaryBus = SecondaryBus;
    tmp->SubordinateBus = SubordinateBus;
    tmp->PortID = PortID;
    tmp->up = NULL;

    for (i = 0; i < TOPOLOGY_NUM_DOWNSTREAM; ++i)
        tmp->down[i] = NULL;


    AddDeviceToTopologyNode(TopologyRoot, tmp);
    return tmp;
}


void
br04Init
(
    LwU32 PcieConfigSpaceBase
)
{
    br04ClearTopology();
    ScanLWTopology(PcieConfigSpaceBase);
}

void
br04DisplayTopology
(
    void
)
{
    if (lwNumBR04s == 0)
    {
        dprintf("Use 'br04init <PcieConfigSpaceBase>' first.\n");
        dprintf("PcieConfigSpaceBase defaults to 0xe0000000 if unspecified.\n");
        return;
    }

    PrintTopology();
}

void
br04DumpBoard
(
    LwU32 bid
)
{
    LwU32 reg, ConnectedPorts, i;
    PhysAddr Bar0;

    if (lwNumBR04s == 0)
    {
        dprintf("Use 'br04init <PcieConfigSpaceBase>' first.\n");
        dprintf("PcieConfigSpaceBase defaults to 0xe0000000 if unspecified.\n");
        return;
    }

    if (bid >= lwNumBR04s)
    {
        dprintf("Invalid Board ID\n");
        return;
    }

    Bar0 = lwBR04s[bid]->Bar0;

    //
    // GPU_REG_RD32 automatically adds lwBar0. Rather than code a new function
    // for a different Bar0, we just remove it.
    //

    dprintf("\nBAR0 Address: " PhysAddr_FMT "\n", Bar0);

    reg = GPU_REG_RD32(Bar0 + LW_BR04_XVU_BUS - lwBar0); // Read register LW_BR04_XVU_BUS

    dprintf("Primary Bus: %d\n", DRF_VAL(_BR04_XVU, _BUS, _PRI_NUMBER, reg));
    dprintf("Secondary Bus: %d\n", DRF_VAL(_BR04_XVU, _BUS, _SEC_NUMBER, reg));
    dprintf("Subordinate Bus: %d\n", DRF_VAL(_BR04_XVU, _BUS, _SUB_NUMBER, reg));

    reg = GPU_REG_RD32(Bar0 + LW_BR04_XVU_DEV_ID - lwBar0); // Read register LW_BR04_XVU_DEV_ID

    dprintf("Device ID: 0x%04X\n", DRF_VAL(_BR04_XVU, _DEV_ID, _DEVICE_ID, reg));

    reg = GPU_REG_RD32(Bar0 + LW_BR04_XVU_HGPU_CTRL - lwBar0); // Read register LW_BR04_XVU_HGPU_CTRL

    dprintf("Hidden: %c\n", DRF_VAL(_BR04_XVU, _HGPU_CTRL, _EN, reg) ? 'Y' : 'N');

    reg = GPU_REG_RD32(Bar0 + LW_BR04_XVU_MCC_PF_REMAP_0 - lwBar0); // Read register LW_BR04_XVU_MCC_PF_REMAP_0
    if (DRF_VAL(_BR04_XVU, _MCC_PF_REMAP_0, _SRC_LOWER_BASE, reg) <=
      DRF_VAL(_BR04_XVU, _MCC_PF_REMAP_0, _SRC_LOWER_LIMIT, reg))
    {
        reg = GPU_REG_RD32(Bar0 + LW_BR04_XVU_MCC_PF_REDIR_BASE(0) - lwBar0); // Read register LW_BR04_XVU_MCC_PF_REDIR_BASE(0)

        dprintf("Broadcast Mode: %s\n", DRF_VAL(_BR04_XVU, _MCC_PF_REDIR_BASE, _ACCESS, reg) ? "Disabled" : "Enabled");
    }
    else
    {
        dprintf("Broadcast Mode: Disabled\n");
    }

    ConnectedPorts = 0;
    for (i = 0; i < TOPOLOGY_NUM_DOWNSTREAM; ++i)
        if (lwBR04s[bid]->down[i] != NULL && lwBR04s[bid]->down[i]->down[0] != NULL)
            ++ConnectedPorts;

    dprintf("# Connected Ports: %d\n", ConnectedPorts);

    reg = GPU_REG_RD32(Bar0 + LW_BR04_XVU_LINK_CAP - lwBar0); // Read register LW_BR04_XVU_LINK_CAP

    switch (DRF_VAL(_BR04_XVU, _LINK_CAP, _MAX_LINK_SPEED, reg))
    {
        case LW_BR04_XVU_LINK_CAP_MAX_LINK_SPEED_2P5G:
            dprintf("Maximum Upstream Link Speed: 2.5 Gb/s (PCI Express Generation 1)\n");
            break;

        case LW_BR04_XVU_LINK_CAP_MAX_LINK_SPEED_5P0G:
            dprintf("Maximum Upstream Link Speed: 5.0 Gb/s (PCI Express Generation 2)\n");
            break;

        default:
            dprintf("Maximum Upstream Link Speed: Unknown\n");
    }

    dprintf("Maximum Upstream Link Width: %dx\n", DRF_VAL(_BR04_XVU, _LINK_CAP, _MAX_LINK_WIDTH, reg));

    reg = GPU_REG_RD32(Bar0 + LW_BR04_XVU_LINK_CTRLSTAT - lwBar0); // Read register LW_BR04_XVU_LINK_CTRLSTAT

    switch (DRF_VAL(_BR04_XVU, _LINK_CTRLSTAT, _LINK_SPEED, reg))
    {
        case LW_BR04_XVU_LINK_CTRLSTAT_LINK_SPEED_2P5G:
            dprintf("Negotiated Upstream Link Speed: 2.5 Gb/s (PCI Express Generation 1)\n");
            break;

        case LW_BR04_XVU_LINK_CTRLSTAT_LINK_SPEED_5P0G:
            dprintf("Negotiated Upstream Link Speed: 5.0 Gb/s (PCI Express Generation 2)\n");
            break;

        default:
            dprintf("Negotiated Upstream Link Speed: Unknown\n");
    }

    dprintf("Negotiated Upstream Link Width: %dx\n", DRF_VAL(_BR04_XVU, _LINK_CTRLSTAT, _NEGO_LINK_WIDTH, reg));

    dprintf("Pending Errors:\n");
    reg = GPU_REG_RD32(Bar0 + LW_BR04_XVU_DEV_CTRLSTAT - lwBar0); // Read register LW_BR04_XVU_DEV_CTRLSTAT
    if (reg & (DRF_SHIFTMASK(LW_BR04_XVU_DEV_CTRLSTAT_CORR_ERR_RPT_EN) |
      DRF_SHIFTMASK(LW_BR04_XVU_DEV_CTRLSTAT_NO_FATAL_ERR_RPT_EN) |
      DRF_SHIFTMASK(LW_BR04_XVU_DEV_CTRLSTAT_FATAL_ERR_RPT_EN) |
      DRF_SHIFTMASK(LW_BR04_XVU_DEV_CTRLSTAT_UNSUPP_REQ_RPT_EN)))
    {
        if (DRF_VAL(_BR04_XVU, _DEV_CTRLSTAT, _CORR_ERR_RPT_EN, reg))
            dprintf("    Correctable Error Detected");
        if (DRF_VAL(_BR04_XVU, _DEV_CTRLSTAT, _NO_FATAL_ERR_RPT_EN, reg))
            dprintf("    Non-Fatal Error Detected");
        if (DRF_VAL(_BR04_XVU, _DEV_CTRLSTAT, _FATAL_ERR_RPT_EN, reg))
            dprintf("    Fatal Error Detected");
        if (DRF_VAL(_BR04_XVU, _DEV_CTRLSTAT, _UNSUPP_REQ_RPT_EN, reg))
            dprintf("    Unsupported Request Detected");
    }
    else
        dprintf("    No errors detected\n");

    dprintf("Credits:\n");
    dprintf("    Upstream Port allocation to Downstream Port 0:\n");
    DumpCredits(Bar0 + LW_BR04_XVU_INT_FLOW_CTL_UP0_TOO_CPL(0) - lwBar0);
    dprintf("    Upstream Port allocation to Downstream Port 1:\n");
    DumpCredits(Bar0 + LW_BR04_XVU_INT_FLOW_CTL_UP0_TOO_CPL(1) - lwBar0);
    dprintf("    Upstream Port allocation to Downstream Port 2:\n");
    DumpCredits(Bar0 + LW_BR04_XVU_INT_FLOW_CTL_UP0_TOO_CPL(2) - lwBar0);
    dprintf("    Upstream Port allocation to Downstream Port 3:\n");
    DumpCredits(Bar0 + LW_BR04_XVU_INT_FLOW_CTL_UP0_TOO_CPL(3) - lwBar0);
    dprintf("    Upstream Port allocation to Message Handler:\n");
    DumpCredits(Bar0 + LW_BR04_XVU_INT_FLOW_CTL_UP0_TOO_CPL(5) - lwBar0);

    dprintf("\n    Message Handler allocation to Downstream Port 0:\n");
    DumpCredits(Bar0 + LW_BR04_XVU_INT_FLOW_CTL_MH0_TOO_CPL(0) - lwBar0);
    dprintf("    Message Handler allocation to Downstream Port 1:\n");
    DumpCredits(Bar0 + LW_BR04_XVU_INT_FLOW_CTL_MH0_TOO_CPL(1) - lwBar0);
    dprintf("    Message Handler allocation to Downstream Port 2:\n");
    DumpCredits(Bar0 + LW_BR04_XVU_INT_FLOW_CTL_MH0_TOO_CPL(2) - lwBar0);
    dprintf("    Message Handler allocation to Downstream Port 3:\n");
    DumpCredits(Bar0 + LW_BR04_XVU_INT_FLOW_CTL_MH0_TOO_CPL(3) - lwBar0);
    dprintf("    Message Handler allocation to Upstream Port:\n");
    DumpCredits(Bar0 + LW_BR04_XVU_INT_FLOW_CTL_MH0_TOO_CPL(4) - lwBar0);
}

void
br04DumpPort
(
    LwU32 bid,
    LwU32 portid
)
{
    LwU32 reg, i;
    PhysAddr UPBar0, DPBar0;
    PLWWATCHTOPOLOGYNODESTRUCT node;

    if (lwNumBR04s == 0)
    {
        dprintf("Use 'br04init <PcieConfigSpaceBase>' first.\n");
        dprintf("PcieConfigSpaceBase defaults to 0xe0000000 if unspecified.\n");
        return;
    }

    if (bid >= lwNumBR04s)
    {
        dprintf("Invalid Board ID\n");
        return;
    }

    UPBar0 = lwBR04s[bid]->Bar0;
    for (i=0; i<TOPOLOGY_NUM_DOWNSTREAM && (lwBR04s[bid]->down[i] == NULL ||
      lwBR04s[bid]->down[i]->PortID != portid); ++i);

    if (i>=TOPOLOGY_NUM_DOWNSTREAM || lwBR04s[bid]->down[i] == NULL ||
      lwBR04s[bid]->down[i]->PortID != portid)
    {
        dprintf("Invalid Port ID\n");
        return;
    }
        
    DPBar0 = lwBR04s[bid]->down[i]->Bar0;

    //
    // GPU_REG_RD32 automatically adds lwBar0. Rather than code a new function
    // for a different Bar0, we just remove it.
    //

    reg = GPU_REG_RD32(DPBar0 + LW_BR04_XVD_BUS - lwBar0); // Read register LW_BR04_XVD_BUS

    dprintf("Primary Bus: %d\n", DRF_VAL(_BR04_XVD, _BUS, _PRI_NUMBER, reg));
    dprintf("Secondary Bus: %d\n", DRF_VAL(_BR04_XVD, _BUS, _SEC_NUMBER, reg));
    dprintf("Subordinate Bus: %d\n", DRF_VAL(_BR04_XVD, _BUS, _SUB_NUMBER, reg));

    reg = GPU_REG_RD32(DPBar0 + LW_BR04_XVD_DEV_ID - lwBar0); // Read register LW_BR04_XVD_DEV_ID

    dprintf("Device ID: 0x%04X\n", DRF_VAL(_BR04_XVD, _DEV_ID, _DEVICE_ID, reg));

    reg = GPU_REG_RD32(DPBar0 + LW_BR04_XVD_REV_CC - lwBar0); // Read register LW_BR04_XVD_REV_CC

    dprintf("Revision ID: Major 0x%04X, Minor 0x%04X\n",
      DRF_VAL(_BR04_XVD, _REV_CC, _MAJOR_REVISION_ID, reg),
      DRF_VAL(_BR04_XVD, _REV_CC, _MINOR_REVISION_ID, reg));

    // Look at downstream GPU or BR04 attached to this port
    node = lwBR04s[bid]->down[i]->down[0];
    dprintf("Connected: %s\n", node ? node->name : "<none>");

    reg = GPU_REG_RD32(DPBar0 + LW_BR04_XVD_LINK_CAP - lwBar0); // Read register LW_BR04_XVD_LINK_CAP

    switch (DRF_VAL(_BR04_XVD, _LINK_CAP, _MAX_LINK_SPEED, reg))
    {
        case LW_BR04_XVD_LINK_CAP_MAX_LINK_SPEED_2P5G:
            dprintf("Maximum Link Speed: 2.5 Gb/s (PCI Express Generation 1)\n");
            break;

        case LW_BR04_XVD_LINK_CAP_MAX_LINK_SPEED_5P0G:
            dprintf("Maximum Link Speed: 5.0 Gb/s (PCI Express Generation 2)\n");
            break;

        default:
            dprintf("Maximum Link Speed: Unknown\n");
    }

    dprintf("Maximum Link Width: %dx\n", DRF_VAL(_BR04_XVD, _LINK_CAP, _MAX_LINK_WIDTH, reg));

    reg = GPU_REG_RD32(DPBar0 + LW_BR04_XVD_LINK_CTRLSTAT - lwBar0); // Read register LW_BR04_XVD_LINK_CTRLSTAT

    switch (DRF_VAL(_BR04_XVD, _LINK_CTRLSTAT, _LINK_SPEED, reg))
    {
        case LW_BR04_XVD_LINK_CTRLSTAT_LINK_SPEED_2P5G:
            dprintf("Negotiated Link Speed: 2.5 Gb/s (PCI Express Generation 1)\n");
            break;

        case LW_BR04_XVD_LINK_CTRLSTAT_LINK_SPEED_5P0G:
            dprintf("Negotiated Link Speed: 5.0 Gb/s (PCI Express Generation 2)\n");
            break;

        default:
            dprintf("Negotiated Upstream Link Speed: Unknown\n");
    }

    dprintf("Negotiated Upstream Link Width: %dx\n", DRF_VAL(_BR04_XVD, _LINK_CTRLSTAT, _NEGO_LINK_WIDTH, reg));

    dprintf("Credits:\n");

    switch (portid)
    {
        case 0:
            dprintf("    Downstream Port 0 allocation to Downstream Port 0:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP0_TOO_CPL(0) - lwBar0);
            dprintf("    Downstream Port 0 allocation to Downstream Port 1:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP0_TOO_CPL(1) - lwBar0);
            dprintf("    Downstream Port 0 allocation to Downstream Port 2:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP0_TOO_CPL(2) - lwBar0);
            dprintf("    Downstream Port 0 allocation to Downstream Port 3:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP0_TOO_CPL(3) - lwBar0);
            dprintf("    Downstream Port 0 allocation to Upstream Port:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP0_TOO_CPL(4) - lwBar0);
            dprintf("    Downstream Port 0 allocation to Message Handler:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP0_TOO_CPL(5) - lwBar0);
            break;

        case 1:
            dprintf("    Downstream Port 1 allocation to Downstream Port 0:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP1_TOO_CPL(0) - lwBar0);
            dprintf("    Downstream Port 1 allocation to Downstream Port 1:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP1_TOO_CPL(1) - lwBar0);
            dprintf("    Downstream Port 1 allocation to Downstream Port 2:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP1_TOO_CPL(2) - lwBar0);
            dprintf("    Downstream Port 1 allocation to Downstream Port 3:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP1_TOO_CPL(3) - lwBar0);
            dprintf("    Downstream Port 1 allocation to Upstream Port:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP1_TOO_CPL(4) - lwBar0);
            dprintf("    Downstream Port 1 allocation to Message Handler:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP1_TOO_CPL(5) - lwBar0);
            break;

        case 2:
            dprintf("    Downstream Port 2 allocation to Downstream Port 0:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP2_TOO_CPL(0) - lwBar0);
            dprintf("    Downstream Port 2 allocation to Downstream Port 1:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP2_TOO_CPL(1) - lwBar0);
            dprintf("    Downstream Port 2 allocation to Downstream Port 2:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP2_TOO_CPL(2) - lwBar0);
            dprintf("    Downstream Port 2 allocation to Downstream Port 3:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP2_TOO_CPL(3) - lwBar0);
            dprintf("    Downstream Port 2 allocation to Upstream Port:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP2_TOO_CPL(4) - lwBar0);
            dprintf("    Downstream Port 2 allocation to Message Handler:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP2_TOO_CPL(5) - lwBar0);
            break;

        case 3:
            dprintf("    Downstream Port 3 allocation to Downstream Port 0:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP3_TOO_CPL(0) - lwBar0);
            dprintf("    Downstream Port 3 allocation to Downstream Port 1:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP3_TOO_CPL(1) - lwBar0);
            dprintf("    Downstream Port 3 allocation to Downstream Port 2:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP3_TOO_CPL(2) - lwBar0);
            dprintf("    Downstream Port 3 allocation to Downstream Port 3:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP3_TOO_CPL(3) - lwBar0);
            dprintf("    Downstream Port 3 allocation to Upstream Port:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP3_TOO_CPL(4) - lwBar0);
            dprintf("    Downstream Port 3 allocation to Message Handler:\n");
            DumpCredits(UPBar0 + LW_BR04_XVU_INT_FLOW_CTL_DP3_TOO_CPL(5) - lwBar0);
            break;
        
        default:
            dprintf("Unsupported Port ID\n");
    }
}
