/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// ttsu@lwpu.com - July 2008
// cdb.cpp - module for reading cli database data
//   -  hyperlink and table feature added
//         djovanovic@lwpu.com - August 2012
//
//*****************************************************

#include "os.h"
#include "CSymHelpers.h"
#include "CSymModule.h"
#include "CSymType.h"
#include "cdb.h"
#include "lwrmTableField.h"

struct objEntryCache
{
    ULONG   hObject;
    ULONG   objType;
    ULONG64 entryAddr;
    ULONG64 objAddr;
};

//
// Enumerate tree (starting at the node with specified value)
//
ULONG
btreeEnumStart
(
    ULONG64  keyOffset,
    ULONG64 *pNode,
    ULONG64  root
)
{
    CSymModule  kmdModule(g_KMDModuleName);
    CSymType    node(&kmdModule, "PNODE");
    *pNode = 0;

    // initialized ?
    if (root)
    {
        ULONG64 current = root;

        while(current)
        {
            if (keyOffset < node.readULONG64(current, "keyStart"))
            {
                *pNode = current;
                current = node.readVirtualPointer(current, "left");
            }
            else if (keyOffset > node.readULONG64(current, "keyEnd"))
            {
                current = node.readVirtualPointer(current, "right");
            }
            else
            {
                *pNode = current;
                break;
            }
        }

        return 0;
    }
    return 0;
}

ULONG
btreeEnumNext
(
    ULONG64 *pNode,
    ULONG64  root
)
{
    CSymModule  kmdModule(g_KMDModuleName);
    CSymType    node(&kmdModule, "PNODE");
    // no nodes ?
    ULONG64 current = 0;

    if (root && *pNode)
    {
        // if we don't have a right subtree return the parent
        current = *pNode;

        // pick the leftmost node of the right subtree ?
        if (node.readVirtualPointer(current, "right"))
        {
            current = node.readVirtualPointer(current, "right");
            for(;node.readVirtualPointer(current, "left");)
            {
                current = node.readVirtualPointer(current, "left");
            }
        }
        else
        {
            // go up until we find the right inorder node
            for(current = node.readVirtualPointer(current, "parent"); current; current = node.readVirtualPointer(current, "parent"))
            {
                if (node.readULONG64(current, "keyStart") > node.readULONG64((*pNode), "keyEnd"))
                {
                    break;
                }
            }
        }
    }
    *pNode = current;

    return 0;
}

int __cdecl compObjTypeAscend(const void *elem1, const void *elem2 )
{
    const objEntryCache *pEntry1 = (const objEntryCache*)elem1,
                        *pEntry2 = (const objEntryCache*)elem2;
    return pEntry1->objType - pEntry2->objType;
}

static char * _getObjType(char *cliEnum)
{
    static struct
    {
        char enumName[64];
        char typeName[64];
    }
    mapping[] =
    {
        {"cliClient"                 , "CLIENTINFO"},
        {"cliDevice"                 , "CLI_DEVICE_INFO"},
        {"cliGsync"                  , "CLI_GSYNC_INFO"},
        {"cliGvo"                    , "CLI_GVO_INFO"},
        {"cliGvi"                    , "CLI_GVI_INFO"},
        {"cliSyscon"                 , "CLI_SYSCON_INFO"},
        {"cliEvent"                  , "CLI_EVENT_INFO"},
        {"cliDma"                    , "CLI_DMA_INFO"},
        {"cliSubdevice"              , "CLI_SUBDEVICE_INFO"},
        {"cliChannel"                , "CLI_FIFO_INFO"},
        {"cliContextShare"           , "CLI_CONTEXT_SHARE_INFO"},
        {"cliMemory"                 , "CLI_MEMORY_INFO"},
        {"cliIfb"                    , "CLI_IFB_INFO"},
        {"cliDisp"                   , "CLI_DISP_INFO"},
        {"cliDispChannel"            , "DISP_CHANNEL_INFO"},
        {"cliDispChannelDmaControl"  , "DISP_CHANNEL_DMA_CONTROL_INFO"},
        {"cliPerfBuffer"             , "CLI_PERFBUFFER_INFO"},
        {"cliP2P"                    , "CLI_P2P_INFO"},
        {"cliThirdPartyP2P"          , "CLI_THIRD_PARTY_P2P_INFO"},
        {"cliObject"                 , "OBJECT"},
        {"cliSmu"                    , "CLI_SUBDEVICE_INFO"},
        {"cliSubdeviceDiag"          , "CLI_SUBDEVICE_DIAG_INFO"},
        {"cliSubdevicePmu"           , "CLI_SUBDEVICE_INFO"},
        {"cliHdacodecObject"         , "CLI_HDACODEC_INFO"},
        {"cliDispCommonObject"       , "CLI_DISP_INFO"},
        {"cliChannelRunlist"         , "CLI_CHANNEL_RUNLIST_INFO"},
        {"cliZbc"                    , "CLI_SUBDEVICE_INFO"},
        {"cliAvp"                    , "CLI_SUBDEVICE_AVP_INFO"},
        {"cliDFD"                    , "CLI_DFD_INFO"},
        {"cliTimer"                  , "CLI_SUBDEVICE_INFO"},
        {"cliDeviceMemBusMapping"    , "CLI_BUS_MAPPING_INFO"},
        {"cliChannelGroup"           , "CLI_CHANNEL_GROUP_INFO"},
        {"cliDebugBuffer"            , "CLI_DEBUG_BUFFER_INFO"},
        {"cliDebugger"               , "CLI_DEBUGGER_INFO"},
        {"cliI2c"                    , "CLI_SUBDEVICE_INFO"},
        {"cliSubdeviceEngine"        , "CLI_SUBDEVICE_ENGINE_INFO"},
        {"cliSyncpoint"              , "SYNCPOINT_INFO"},
        {"cliSyncpointBase"          , "SYNCPOINT_BASE_INFO"},
        {"cliDispSfUser"             , "CLI_SUBDEVICE_INFO"},
        {"cliDeferredApi"            , "CLI_DEFERRED_API"},
        {"cliPerfmon"                , "CLI_PERFMON_INFO"},
        {"cliVASpace"                , "CLI_VASPACE_INFO"},
        {"cliVgpu"                   , "CLI_VGPU_INFO"}
    };
    ULONG mappingSize = sizeof(mapping)/sizeof(*mapping);
    for (ULONG i = 0; i < mappingSize; i++)
    {
        if (strcmp(cliEnum, mapping[i].enumName) == 0)
        {
            return mapping[i].typeName;
        }
    }
    return NULL;
}

// 
// dumpClientDB()
//  - Dumps the client database from the given root
//  Returns :   void
//  Params  :   rootObjAddr - virtual address of root object
//              cliHandle   - client handle; 0 to dump all clients.
//              flags       - print control flags
//
void
dumpClientDB(
    ULONG64 rootObjAddr,
    ULONG   cliHandle,
    ULONG   flags
)
{
    CSymModule  kmdModule(g_LwrrentModuleName);
    CSymType    clientInfo(&kmdModule, "CLIENTINFO");
    CSymType    node(&kmdModule, "PNODE");
    CSymType    entry(&kmdModule, "PCLI_OBJ_REGISTER_ENTRY");
    CSymType    elementTypeEnum(&kmdModule, "ClientElementType");
    ULONG64     cliInfoAddr = 0;
    ULONG64     objTreeRootAddr = 0;
    ULONG64     objNodeAddr = 0;
    ULONG64     objEntryAddr = 0;
    ULONG64     objHandle = 0;
    ULONG64     objAddr = 0;
    ULONG       elementState = 0;
    ULONG       elementType = 0;
    ULONG       clientHandle = 0;
    ULONG       clientCount = 0;
    ULONG       clientObjCount = 0;
    ULONG       lastObjType;
    ULONG       lastObjTypeCount;
    ULONG       j;
    ULONG       tableOutput = !(cliHandle || (flags & LWWATCH_CDB_DUMP_FLAGS_VERBOSE));
    char        elementTypeName[64];
    objEntryCache *entryCache;
    AddressLinkField addrHyperlink(14, 22, 1, 1);

    LwU32 size = elementTypeEnum.getSize();


    //  this code is added here temporary until I find a better place to put it
    if(tableOutput)
    {
        StructULONGField hClientField(13, "CLIENTINFO", "hClient");
        StructULONGField classField(8, "CLIENTINFO", "Class", 2);
        StructULONGField procIDField(9, "CLIENTINFO", "ProcID");
        StructULONGField flagsField(8, "CLIENTINFO", "Flags", 1);
        StructULONGField uniqueObjHandleField(18, "CLIENTINFO", "UniqueObjHandle", 2);
        TableField       privilegeField(12, 2);
        AddressLinkField objRegisterField(13, 21, 2);
        if (IsPtr64())
        {
            dprintf("     clientInfo         hClient    Class   ProcID   Flags   UniqueObjHandle   Privilege    Object Register\n"
                    "-------------------- ------------ ------- -------- ------- ----------------- ----------- --------------------\n");
        }
        else
        {
            dprintf(" clientInfo     hClient    Class   ProcID   Flags   UniqueObjHandle   Privilege   Object Register\n"
                    "------------ ------------ ------- -------- ------- ----------------- ----------- -----------------\n");
        }
        
        for (int i = 0; i < NUM_CLIENTS; i++)
        {
            cliInfoAddr = rootObjAddr + clientInfo.getSize() * i;
            elementState = clientInfo.readULONG(cliInfoAddr, "elementState");
            if (elementState == clientElementEnabled)
            {
                LwU32   data;

                addrHyperlink.print("dt %s!CLIENTINFO %#llx", 
                                    cliInfoAddr,
                                    g_LwrrentModuleName,
                                    cliInfoAddr);
                hClientField.print("0x%x", cliInfoAddr); 
                classField.print("%-3d", cliInfoAddr);
                procIDField.print("0x%x", cliInfoAddr);
                flagsField.print("0x%x", cliInfoAddr);
                uniqueObjHandleField.print("0x%x", cliInfoAddr);
                data = clientInfo.readULONG(cliInfoAddr, "Privilege");
                privilegeField.print("%s", data?"true":"false");
                data = clientInfo.readULONG(cliInfoAddr, "hClient");
                objRegisterField.print("!lw.dumpclientdb -h %x -v", cliInfoAddr, data);

                dprintf("\n");
            }
        }
        return;
    }
    
    for (int i = 0; i < NUM_CLIENTS; i++)
    {
        cliInfoAddr = rootObjAddr + clientInfo.getSize() * i;
        elementState = clientInfo.readULONG(cliInfoAddr, "elementState");
        if (elementState == clientElementEnabled)
        {
            clientHandle = clientInfo.readULONG(cliInfoAddr, "hClient");

            if (cliHandle != 0 && clientHandle != cliHandle)
            {
                continue;
            }

            clientCount++;

            dprintf("  hClient: ");
          
            addrHyperlink.print("!lw.dumpclientdb -h %#llxx -v", clientHandle, clientHandle);

            dprintf("\n");

            clientObjCount = 0;
            objTreeRootAddr = clientInfo.readVirtualPointer(cliInfoAddr, "CliObjectRegister");

            btreeEnumStart(0, &objNodeAddr, objTreeRootAddr);
            while (objNodeAddr != 0)
            {
                btreeEnumNext(&objNodeAddr, objTreeRootAddr);
                clientObjCount++;
            }

            entryCache = new objEntryCache[clientObjCount];
            if (entryCache == 0)
            {
                dprintf("ERROR: insufficient memory\n");
                return;
            }

            clientObjCount = 0;
            btreeEnumStart(0, &objNodeAddr, objTreeRootAddr);
            while (objNodeAddr != 0)
            {
                objEntryAddr = node.readVirtualPointer(objNodeAddr, "Data");

                entryCache[clientObjCount].hObject = (ULONG)node.readULONG64(objNodeAddr, "keyStart");
                entryCache[clientObjCount].entryAddr = objEntryAddr;
                entryCache[clientObjCount].objType = entry.readULONG(objEntryAddr, "elementType");
                entryCache[clientObjCount++].objAddr = entry.readVirtualPointer(objEntryAddr, "ptrToObject");

                btreeEnumNext(&objNodeAddr, objTreeRootAddr);
            }

            qsort(entryCache, clientObjCount, sizeof(objEntryCache), compObjTypeAscend);

            lastObjType = 0xFFFFFFFF;
            lastObjTypeCount = 0xFFFFFFFF;
            for (j = 0; j < clientObjCount; j++)
            {
                if (lastObjType != entryCache[j].objType)
                {
                    if (lastObjTypeCount != 0xFFFFFFFF)
                    {
                        // retrieve the name of the property
                        elementTypeEnum.getConstantName(entryCache[j-1].objType, elementTypeName, sizeof(elementTypeName));
                        dprintf("    %s: %d allocated\n", elementTypeName, lastObjTypeCount);

                        if (flags & LWWATCH_CDB_DUMP_FLAGS_VERBOSE)
                        {
                            for (ULONG k = j - lastObjTypeCount; k < j; k++)
                            {
                                AddressLinkField ptrToObject(10, 18);
                                dprintf("      h%-12s: 0x%8x  ptrToObject: ",
                                        elementTypeName + 3, entryCache[k].hObject, entryCache[k].objAddr);
                                ptrToObject.print("dt %s!%s %#llx", entryCache[k].objAddr,
                                                                    g_LwrrentModuleName,
                                                                    _getObjType(elementTypeName),
                                                                    entryCache[k].objAddr);
                                dprintf("\n");
                            }
                        }
                    }
                    lastObjTypeCount = 0;
                    lastObjType = entryCache[j].objType;
                }
                lastObjTypeCount++;
            }

            // retrieve the name of the property
            elementTypeEnum.getConstantName(entryCache[j-1].objType, elementTypeName, sizeof(elementTypeName));
            dprintf("    %s: %d allocated\n", elementTypeName, lastObjTypeCount);

            if (flags & LWWATCH_CDB_DUMP_FLAGS_VERBOSE)
            {
                for (ULONG k = j - lastObjTypeCount; k < j; k++)
                {
                    AddressLinkField ptrToObject(10, 18);
                    dprintf("      h%-12s: 0x%8x  ptrToObject: ",
                            elementTypeName + 3, entryCache[k].hObject, entryCache[k].objAddr);
                    ptrToObject.print("dt %s!%s %#llx -r", entryCache[k].objAddr,
                                                        g_LwrrentModuleName,
                                                        _getObjType(elementTypeName),
                                                        entryCache[k].objAddr);
                    dprintf("\n");
                }
            }

            dprintf("  Total objects allocated by this client : %d\n\n", clientObjCount);

            delete[] entryCache;
        }
    }

    dprintf("Total clients found : %d\n", clientCount);


    return;
}
