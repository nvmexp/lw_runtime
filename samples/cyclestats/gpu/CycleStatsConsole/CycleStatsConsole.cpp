 /************************ BEGIN COPYRIGHT NOTICE ***************************\
|*                                                                           *|
|* Copyright 2003-2010 by LWPU Corporation.  All rights reserved.  All     *|
|* information contained herein is proprietary and confidential to LWPU    *|
|* Corporation.  Any use, reproduction, or disclosure without the written    *|
|* permission of LWPU Corporation is prohibited.                           *|
|*                                                                           *|
 \************************** END COPYRIGHT NOTICE ***************************/

#include <windows.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lw32.h>
#include <lwos.h>
#include <lwcm.h>
#include <lwrmapi.h>

#include "CycleStatsConsole.h"
#include "lwPerfMonHW.h"

// LwIoControl is in lwarch32.dll, which isn't on winnt machines.
// Let's try replacing this with a null function, to see if it's ever actually called.
void __stdcall LwIoControl(int a, void *v) {a;v;}

// resman client connection
LwU32           g_hClient = ~0;
LwU32           g_deviceNumber = 0;
LwU32           g_chipId = 0;
LwU32           g_chipIdOverride = 0;
void           *g_bar0 = NULL;
LwU32           g_fakeBar0 = false;
LwU32           g_accessBAR0ViaRM = false;
LwU32           g_parseOnly = false;
LwU32           g_testAllFrames = false;
LwU32           g_printToDebug = false;
LwU32           g_preciseSleep = false;
LwPMResource   *g_lwpm = NULL;
LwU32           g_pollInterval = 1000;
LwU32           g_pmDynFrameModulo = ~0;

/** switches to the next set of experiments (iterates over the "frames")
  *
  */
void log(const char *format, ...)
{
    va_list va;

    va_start(va, format);
    char tmp[8192];
    int replaceStringLen = _vsnprintf(tmp, sizeof(tmp), format, va);
    strcpy(&tmp[sizeof(tmp)-4], "...");
    va_end(va);

    if (g_printToDebug)
    {
        OutputDebugStringA(tmp);
    }
    else
    {
        printf("%s", tmp);
    }
}

/** switches to the next set of experiments (iterates over the "frames")
  *
  */
bool advanceFrame
(
    LwPMResource::PMInterface     **ppPMRead,
    LwU32                           pmCount,
    LwPMResource::PokeInterface   **ppPokeRead,
    LwU32                           pokeCount
)
{
    LwU32 oldFrameModulo = g_pmDynFrameModulo;
    bool wrapAround = true;
    if (pmCount)
    {
        for(LwU32 i=0; i<pmCount; i++)
        {
            LwPMResource::PMInterface *pi = ppPMRead[i];
            if (!pi->isStatic() && pi->isAllocated() && (pi->getFrameModulo()==oldFrameModulo))
            {
                pi->freeResources();
            }
            if (!pi->isStatic() && (pi->getFrameModulo()==(oldFrameModulo+1)))
            {
                wrapAround = false;
            }
        }
    }


    if (wrapAround || (g_pmDynFrameModulo == ~0))
    {
        g_pmDynFrameModulo = 0;
    }
    else
    {
        g_pmDynFrameModulo++;
    }

    // allocate new batch of pm configurations
    LwU32 id = g_pmDynFrameModulo;
    if (pmCount)
    {
        // first bind *all* mux at once
        g_lwpm->beginMultiConfigMuxSelection();
        for (LwU32 i=0; i<pmCount; i++)
        {
            LwPMResource::PMInterface *pi = ppPMRead[i];
            if (!pi->isStatic() && (pi->getFrameModulo()==id))
            {
                if (!g_lwpm->addMultiConfigMuxSelection(pi->getConfig()))
                {
                    log("** advanceFrame(): failed addMultiConfigMuxSelection() '%s'\n", g_lwpm->getConfigName(i));
                }
            }
        }
        if (!g_lwpm->bindMultiConfigMuxSelection())
        {
            log("** advanceFrame(): can't resolve watchbus/instance select register conflicts!\n");
        }

        // now allocate each config
        for (LwU32 i=0; i<pmCount; i++)
        {
            LwPMResource::PMInterface *pi = ppPMRead[i];
            if (!pi->isStatic() && (pi->getFrameModulo()==id))
            {
                if (!pi->allocateResources(g_lwpm))
                {
                    log("** advanceFrame(): allocateResources() failed for '%s'\n", pi->getTitle());
                }
            }
        }

        // unbind all global mux since after the allocation locked the mux/instance selects
        g_lwpm->unbindMultiConfigMuxSelection();
        g_lwpm->endMultiConfigMuxSelection();
    }

    // per frame poke registers...
    for(LwU32 i=0; i<pokeCount; i++)
    {
        LwPMResource::PokeInterface *p = ppPokeRead[i];
        if (!p->isStatic() && (p->getFrameModulo()==id))
        {
             p->pokeValue(LWSTREAMID_CPU_0);
        }
    }

    return wrapAround;
}

/**
  *
  */
static void run(const char *pmConfigStr)
{
    log("--- config string ---\n");
    log("%s", pmConfigStr);
    log("\n");

    // lock (single gpu only)
    g_lwpm = LwPMResource::create((void*)(size_t)g_chipId, LWSTREAMID_CPU_0, 0x1);
    if (g_lwpm)
    {
        g_lwpm->initialize();
        if (g_lwpm->isInitialized())
        {
            LwPMResource::PMInterface **ppPMRead = NULL;
            LwPMResource::PokeInterface **ppPokeRead = NULL;
            LwU32 pmNextFreeRead = 0;
            LwU32 pokeNextFreeRead = 0;
            log("--- parser log ---\n");
            if (strlen(pmConfigStr) && g_lwpm->parseConfig(pmConfigStr))
            {
                // any config to setup ?
                log("--- config creation log ---\n");
                LwU32 n = g_lwpm->getConfigCount();
                if (n)
                {
                    ppPMRead = new LwPMResource::PMInterface*[n];
                    ppPokeRead = new LwPMResource::PokeInterface*[n];
                }
                if (ppPMRead && ppPokeRead)
                {
                    // first bind *all* mux at once
                    g_lwpm->beginMultiConfigMuxSelection();
                    for(LwU32 i=0; i<n; i++)
                    {
                        if (g_lwpm->isConfigStatic(i) && g_lwpm->needsGPUPerfmon(i))
                        {
                            if (!g_lwpm->addMultiConfigMuxSelection(g_lwpm->getConfig(i)))
                            {
                                log("** failed addMultiConfigMuxSelection('%s')\n", g_lwpm->getConfigName(i));
                            }
                        }
                    }
                    if (!g_lwpm->bindMultiConfigMuxSelection())
                    {
                        log("** bindMultiConfigMuxSelection(): can't resolve watchbus/instance select register conflicts!\n");
                    }

                    // now allocate each config
                    for(LwU32 i=0; i<n; i++)
                    {
                        if (g_lwpm->needsGPUPerfmon(i))
                        {
                            LwPMResource::PMInterface *ri = g_lwpm->createGPUPerfMonConfig(i);
                            if (ri)
                            {
                                ppPMRead[pmNextFreeRead] = ri;
                                pmNextFreeRead++;
                            }
                            else
                            {
                                log("** failed to create/commit config '%s'\n", g_lwpm->getConfigName(i));
                            }
                        }
                        if (g_lwpm->needsGPUPoke(i))
                        {
                            LwPMResource::PokeInterface *ri = g_lwpm->createGPUPokeConfig(i);
                            if (ri)
                            {
                                ppPokeRead[pokeNextFreeRead] = ri;
                                pokeNextFreeRead++;
                            }
                            else
                            {
                                log("** failed to create/commit config '%s'\n", g_lwpm->getConfigName(i));
                            }
                        }
                    }

                    // unbind all global mux since the allocations locked the mux/instance selects
                    g_lwpm->unbindMultiConfigMuxSelection();
                    g_lwpm->endMultiConfigMuxSelection();

                    // print results
                    log("--- config dump ---\n");
                    for(LwU32 j=0; j<pmNextFreeRead; j++)
                    {
                        log("PM-Event%d:\tsetup=%s\n", j, ppPMRead[j]->getDesc());
                    }
                    for(LwU32 j=0; j<pokeNextFreeRead; j++)
                    {
                        log("PM-Poke%d:\tsetup=%s\n", j, ppPokeRead[j]->getDesc());
                    }
                    g_lwpm->clearConfig();

                    // one-time programming...
                    if (pokeNextFreeRead)
                    {
                        // poke registers
                        for(LwU32 i=0; i<pokeNextFreeRead; i++)
                        {
                            LwPMResource::PokeInterface *p = ppPokeRead[i];
                            if (p->isStatic())
                            {
                                p->pokeValue(LWSTREAMID_CPU_0);
                            }
                        }
                    }
                }

                if (g_testAllFrames)
                {
                    log("--- iterating over all frames once ---\n");
                    for(;;)
                    {
                        if (advanceFrame(ppPMRead, pmNextFreeRead, ppPokeRead, pokeNextFreeRead))
                        {
                            break;
                        }
                    }
                }

                // loop until termination
                if (!g_parseOnly)
                {
                    log("--- running results ---\n");
                    advanceFrame(ppPMRead, pmNextFreeRead, ppPokeRead, pokeNextFreeRead);
                    g_lwpm->trigger(LWSTREAMID_CPU_0, LwPMResource::TRIGGER_NOW);
                    for(;;)
                    {
                        // update every second once
                        Sleep(g_pollInterval);

                        g_lwpm->trigger(LWSTREAMID_CPU_0, LwPMResource::TRIGGER_NOW);

                        // print current pm results
                        log("new perfmon poll results:\n");
                        for(LwU32 i=0; i<pmNextFreeRead; i++)
                        {
                            if (ppPMRead[i]->isAllocated())
                            {
                                LwPMResult result;
                                ppPMRead[i]->begin(LWSTREAMID_CPU_0, 0);
                                ppPMRead[i]->end(&result, LWSTREAMID_CPU_0);

                                // flush the various pipelines queues
                                g_lwpm->processPendingResults();
                                assert(result.status == LwPMResult::LWPMSTATUS_RECV_U64);
                                log("  %s = %.3fM events (%lld) \n", ppPMRead[i]->getTitle(), (double)result.u64/(1000*1000), result.u64);
                            }
                        }

                        // apply "few frame" logic
                        advanceFrame(ppPMRead, pmNextFreeRead, ppPokeRead, pokeNextFreeRead);
                    }
                }
                // free resources
                for(LwU32 i=0; i<pmNextFreeRead; i++)
                {
                    assert(ppPMRead[i]);
                    delete ppPMRead[i];
                }
                if (ppPMRead)
                {
                    delete[] ppPMRead;
                }
                for(LwU32 i=0; i<pokeNextFreeRead; i++)
                {
                    assert(ppPokeRead[i]);
                    delete ppPokeRead[i];
                }
                if (ppPokeRead)
                {
                    delete[] ppPokeRead;
                }
            }
        }
        g_lwpm->destroy();
        delete g_lwpm;
        g_lwpm = NULL;
    }
}

/**
  *
  */
static void __cdecl exitNotifcation(void)
{
    if (g_hClient != ~0)
    {
        LwRmFree(g_hClient, g_hClient, g_hClient);
        g_hClient = ~0;
    }
}

/**
  *
  */
static void printUsage()
{
    printf("usage: cyclestats -pmconfig <filename>\n");
    printf("                  [-delay <time(ms) between polls>]\n");
    printf("                  [-gpu <arch|impl>]\n");
    printf("                  [-fakeBAR0]\n");
    printf("                  [-accessBAR0ViaRM]\n");
    printf("                  [-preciseSleep]\n");
    printf("                  [-parseonly]\n");
    printf("                  [-testAllFrames]\n");
    printf("                  [-device <deviceNumber, 0..7>]\n");
    printf("                  [-printToDebug]\n");
}

/**
  *
  */
int main(int argc, char* argv[])
{
    char *pmConfigStr = NULL;
    for(int k=1; k<argc; k++)
    {
        if ((k+1 < argc) && _stricmp(argv[k], "-pmconfig") == 0)
        {
            FILE *f = fopen(argv[k+1], "rb");
            if (f)
            {
                // determine file size
                fseek(f, 0, SEEK_END);
                unsigned int fileSize = ftell(f);
                fseek(f, 0, SEEK_SET);

                pmConfigStr = new char[fileSize + 1];
                if (pmConfigStr)
                {
                    memset(pmConfigStr, 0, fileSize + 1);
                    size_t bytesRead = fread(pmConfigStr, 1, fileSize, f);
                    if (!bytesRead)
                    {
                        printf("failed to read any data\n");
                    }
                }
                else
                {
                    printf("out of memory (%d bytes). \n", fileSize + 1);
                }
                fclose(f);
            }
            else
            {
                printf("failed to open '%s'.\n", argv[k + 1]);
            }
            k++;
            continue;
        }
        if ((k+1 < argc) && _stricmp(argv[k], "-gpu") == 0)
        {
            if (sscanf(argv[k+1], "%x", &g_chipIdOverride) != 1)
            {
                printf("failed to parse '%s'.\n", argv[k+1]);
            }
            k++;
            continue;
        }
        if ((k+1 < argc) && _stricmp(argv[k], "-device") == 0)
        {
            if (sscanf(argv[k+1], "%x", &g_deviceNumber) != 1)
            {
                printf("failed to parse '%s'.\n", argv[k+1]);
            }
            k++;
            continue;
        }
        if (_stricmp(argv[k], "-parseonly") == 0)
        {
            g_parseOnly = true;
        }
        if (_stricmp(argv[k], "-testAllFrames") == 0)
        {
            g_testAllFrames = true;
        }
        if (_stricmp(argv[k], "-printToDebug") == 0)
        {
            g_printToDebug = true;
        }
        if (_stricmp(argv[k], "-fakeBAR0") == 0)
        {
            g_fakeBar0 = true;
        }
        if (_stricmp(argv[k], "-accessBAR0ViaRM") == 0)
        {
            g_accessBAR0ViaRM = true;
        }
        if (_stricmp(argv[k], "-preciseSleep") == 0)
        {
            g_preciseSleep = true;
        }
        if ((k+1 < argc) && _stricmp(argv[k], "-delay") == 0)
        {
            g_pollInterval = atoi(argv[k+1]);
        }
    }

    if (!pmConfigStr)
    {
        printUsage();
        return 0;
    }

    // check parsed arguments
    if (g_deviceNumber >= 8)
    {
        printf("** device number is illegal (%d >= 8)!\n", g_deviceNumber);
        return 100;
    }

    // alloce root/client
    if (!g_fakeBar0 && LwRmAllocRoot(&g_hClient) != LWOS_STATUS_SUCCESS)
    {
        printf("** LwRmAllocRoot failed!\n");
        return 100;
    }

    // allocate from the desktop/hDC on display0
    unsigned char devName[32];
    memset(devName, 0, sizeof(devName));
    if (g_fakeBar0 || LwRmAlloc(g_hClient, g_hClient, CONSOLE_DEVICE_HANDLE_ID, LW01_DEVICE_0 + g_deviceNumber, devName) == LWOS_STATUS_SUCCESS)
    {
        // identify chip
        LwU32 arch = ~0;
        LwU32 impl = ~0;
        if (!g_fakeBar0)
        {
            if (LwRmConfigGet(g_hClient, CONSOLE_DEVICE_HANDLE_ID, LW_CFG_ARCHITECTURE, &arch) != LWOS_STATUS_SUCCESS)
            {
                log("** LwRmConfigGet failed!\n");
                return 101;
            }

            // implementation
            if (LwRmConfigGet(g_hClient, CONSOLE_DEVICE_HANDLE_ID, LW_CFG_IMPLEMENTATION, &impl) != LWOS_STATUS_SUCCESS)
            {
                log("** LwRmConfigGet failed!\n");
                return 102;
            }
            g_chipId = arch | impl;
        }
        if (g_chipIdOverride)
        {
            g_chipId = g_chipIdOverride;
            log("** forcing gpu = 0x%x\n", g_chipId);
        }
        atexit(exitNotifcation);

        // create a mapping of BAR0
        void *pAddress = NULL;
        if (g_fakeBar0)
        {
            unsigned int bar0Size = 64*1024*1024;
            pAddress = new char[bar0Size];
            if (pAddress)
            {
                memset(pAddress, 0, bar0Size);
            }
            g_bar0 = pAddress;
        }
        else
        {
            // PMDriverInterface::reservePerfMonHardware()
            LwU32 dwOldValue = 0;
            LwRmConfigSet(g_hClient, CONSOLE_DEVICE_HANDLE_ID, LW_CFG_RESERVE_PERFMON_HW, 1, &dwOldValue);

            LwU32 limit = 0;
            if (LwRmAllocMemory(g_hClient, CONSOLE_DEVICE_HANDLE_ID, CONSOLE_MAPPING_ID, LW01_MEMORY_LOCAL_PRIVILEGED,
                                DRF_DEF(OS02, _FLAGS, _ALLOC, _NONE),
                                &pAddress, &limit) == LWOS_STATUS_SUCCESS)
            {
                if (LwRmMapMemory(g_hClient, CONSOLE_DEVICE_HANDLE_ID, CONSOLE_MAPPING_ID, 0, limit, &pAddress,
                                LW04_MAP_MEMORY_FLAGS_NONE) == LWOS_STATUS_SUCCESS)
                {
                    // start
                    log("mapped BAR0 to 0x%p-0x%p (%.2fMB)\n", (void*)pAddress, (void*)((char*)pAddress+limit), limit*(1.0f/(1024.0f*1024.0f)));
                    g_bar0 = pAddress;
                }
            }
        }

        if (g_preciseSleep)
        {
            log("** forcing timer resolution to 1ms\n");
            timeBeginPeriod(1);
        }

        // a BAR0 mapping is mandatory
        if (pAddress)
        {
            run(pmConfigStr);
        }

        if (g_preciseSleep)
        {
            timeEndPeriod(1);
        }

        if (g_fakeBar0)
        {
            if (pAddress)
            {
                delete[] pAddress;
            }
        }
        else
        {
            // PMDriverInterface::releasePerfMonHardware()
            LwU32 dwOldValue = 0;
            LwRmConfigSet(g_hClient, CONSOLE_DEVICE_HANDLE_ID, LW_CFG_RESERVE_PERFMON_HW, 0, &dwOldValue);
        }
    }

    delete[] pmConfigStr;
    pmConfigStr = NULL;

    // free all
    if (!g_fakeBar0)
    {
        LwRmFree(g_hClient, g_hClient, g_hClient);
        g_hClient = ~0;
    }
    return 0;
}
