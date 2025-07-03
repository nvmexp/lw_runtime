/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _BR04_H_
#define _BR04_H_

#include "br04/br04_ref.h"

//
// topology support
//

#define TOPOLOGY_NUM_DOWNSTREAM 4

typedef struct _topology_node_struct
{
    char name[64];
    LwU32 Bar0;
    LwU32 PrimaryBus, SecondaryBus, SubordinateBus, PortID;
    struct _topology_node_struct *up, *down[TOPOLOGY_NUM_DOWNSTREAM];
} LWWATCHTOPOLOGYNODESTRUCT, *PLWWATCHTOPOLOGYNODESTRUCT;

extern LwU32 lwNumBR04s;
extern PLWWATCHTOPOLOGYNODESTRUCT TopologyRoot;
extern PLWWATCHTOPOLOGYNODESTRUCT lwBR04s[128];


//
// global functions
//

void br04ClearTopology(void);
PLWWATCHTOPOLOGYNODESTRUCT br04AddDeviceToTopology(const char *name, LwU32 Bar0, LwU32 PrimaryBus, LwU32 SecondaryBus, LwU32 SubordinateBus, LwU32 PortID);
void br04Init(LwU32 PcieConfigSpaceBase);
void br04DisplayTopology(void);
void br04DumpBoard(LwU32 bid);
void br04DumpPort(LwU32 bid, LwU32 portid);


#endif // _BR04_H_
