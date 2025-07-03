////////////////////////////////////////////////////////////////////////////////
// Filename: main.cpp
////////////////////////////////////////////////////////////////////////////////
#include "systemclass.h"
#include <assert.h>
#include <string>
#include <windows.h>

#include "stdafx.h"
#include "CoprocStatistics.h"
#include "ntddvdeo.h"
#include "cfgmgr32.h"

// GPULatencyTestsApp
// Version : 2.0

// Keeping this enum in case if we want to add more modes in future.
typedef enum _TestModeEnum
{
    SINGLE_PASS,
}
TestModeEnum;

#define I_SWITCH_USAGE_STRING "numTrangles numFrames delayInterval numWasteMB" 

int main(int argc, char *argv[])
{
    SystemClass* System = NULL;
    bool result;
    FILE* fp;

    errno_t err;
    err = fopen_s(&fp,"main_log.txt","w+");

    // default test paramters
    DWORD initialTris = 1;
    DWORD numCycles = 1800;
    DWORD delayInterval = 1000;
    bool bIsFullscreen = false;
    bool bGetGOLDStats = false;
    DWORD numWasteMB = 512;
	bool forced = false;
    TestModeEnum tMode = SINGLE_PASS;

    bool bSystemInitDone = false;

    bool bResetCycles = true;
    LwU64 cycleCount = 0;

    System = new SystemClass;
    if (!System)
    {
        fclose(fp);
        return 0;
    }

    System->RedirectIOToConsole();
        
    int nArg;
    for( nArg = 1; nArg < argc; nArg++ )
    {
        if(strcmp(argv[nArg], "/i") == 0)
        {   // specify initial condition
            if(nArg+4 >= argc) // needs four additional parameters
            {
                fprintf(fp, "\n /i usage: %s \n", I_SWITCH_USAGE_STRING );
                System->errorParam(fp);
                goto Shutdwn;
            }

            nArg++;
            initialTris = strtol(argv[nArg], 0, 0); // TODO: should initialTris be sanitized?

            nArg++;
            numCycles = strtol(argv[nArg], 0, 0); // TODO: should numCycles be sanitized?

            nArg++;
            delayInterval = strtol(argv[nArg], 0, 0); // TODO: should delayInterval be sanitized?

            nArg++;
            numWasteMB = strtol(argv[nArg], 0, 0); // TODO: should numWasteMB be sanitized?
            if(numWasteMB > 512)
            {
                fprintf(fp, "\n Wasted Memory clamped to max 512 MB\n");
                numWasteMB = 512;
            }
            else if(numWasteMB == 0)
            {
                fprintf(fp, "\n Wasted Memory clamped to min 1x1x1x4Bpp\n");
            }
            else
            {
                fprintf(fp, "\n Wasted Memory : %d MB \n", numWasteMB );
            }
        }
        else if (strcmp(argv[nArg], "/h") == 0)//help
        {
            fprintf(fp, "\n /i initial settings usage: %s \n", I_SWITCH_USAGE_STRING );
            goto Shutdwn;
        }
        else
        {
            fprintf(fp, "\n unknown switch : %s \n", argv[nArg]);
            System->errorParam(fp);
            goto Shutdwn;
        }

    }

    result = System->Initialize(initialTris, numCycles, delayInterval, bIsFullscreen, bGetGOLDStats, numWasteMB);
    bSystemInitDone = true;

    if( bResetCycles == true )
    {
        System->ClearCoprocCycles(bResetCycles, cycleCount);
    }

	// kept this switch case in case, we want to add more cases to this Customer version of GPULatencyTest app. 
    switch(tMode)
    {
    case SINGLE_PASS:
        System->Run(forced); //no optional parameters
        break;

	default: 
		fprintf(fp, "\n Please provide command line as /i usage: %s \n", I_SWITCH_USAGE_STRING );
    }
     
    // Shutdown and release the system object.
    Shutdwn:
    if( bSystemInitDone ) System->Shutdown(bIsFullscreen);
    delete System;
    System = 0;
    fprintf(fp,"\n Ops completed \n");
    fclose(fp);
    return 0;
}