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

typedef enum _TestModeEnum
{
    SINGLE_PASS,
    CALIBRATE,
    RANDOM,
    TARGET
}
TestModeEnum;

#define I_SWITCH_USAGE_STRING "numTrangles numFrames delayInterval fullscreen GOLD numWasteMB" 
#define C_SWITCH_USAGE_STRING "calibrateTime" 
#define T_SWITCH_USAGE_STRING "targetCycles targetStepAmount targetSearchDirection"
#define R_SWITCH_USAGE_STRING "randomMaxTriangles randomMaxDelayInterval randomHoldInterval"
#define P_SWITCH_USAGE_STRING "PowerCycles"

int main(int argc, char *argv[])
{
    SystemClass* System = NULL;
    bool result;
    FILE* fp;

    errno_t err;
    err = fopen_s(&fp,"main_log.txt","w+");

    // default test paramters
    DWORD initialTris = 1;
    DWORD numCycles = 10;
    DWORD delayInterval = 10000;
    bool bIsFullscreen = false;
    bool bGetGOLDStats = false;
    float CalibrateTime = 1.f;
    DWORD numWasteMB = 512;
    bool forced = false;
    bool DoCalibrate = 0;
    TestModeEnum tMode = SINGLE_PASS;

    // target mode parameters w/o defaults
    DWORD targetCycles;
    DWORD targetStepAmount;
    int targetSearchDirection;

    // random mode parameters w/o defaults
    DWORD randomMaxTriangles;
    DWORD randomMaxDelayInterval;
    DWORD randomHoldInterval;

    bool bSystemInitDone = false;

    bool bResetCycles = false;
    LwU64 cycleCount = 0;
    CLEAR_TOOL_STATS doNotclearStat = CLEAR_END_AND_START;

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
            if(nArg+6 >= argc) // needs six additional parameters
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
            bIsFullscreen = strtol(argv[nArg], 0, 0) == 0 ? false : true;

            nArg++;
            bGetGOLDStats = strtol(argv[nArg], 0, 0) == 0 ? false : true;

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
        else if(strcmp(argv[nArg], "/f") == 0) //forced
        {
            forced = true;
        }
        else if(strcmp(argv[nArg], "/c") == 0) //calibrate
        {
            if( tMode != SINGLE_PASS )
            {
                fprintf(fp, "\n /c cannot be used with /t or /r \n" );
                System->errorParam(fp);
                goto Shutdwn;
            }
            tMode = CALIBRATE;

            if(nArg+1 >= argc) // needs one additional parameter
            {
                fprintf(fp, "\n /c usage: %s \n", C_SWITCH_USAGE_STRING );
                System->errorParam(fp);
                goto Shutdwn;
            }

            nArg++;

            DoCalibrate = true;
            CalibrateTime = std::stof(argv[nArg], NULL); // Should this be error checked?
        }
        else if(strcmp(argv[nArg], "/t") == 0)
        {
            if( tMode != SINGLE_PASS )
            {
                fprintf(fp, "\n /t cannot be used with /c or /r \n" );
                System->errorParam(fp);
                goto Shutdwn;
            }
            tMode = TARGET;

            if(nArg+3 >= argc) // needs 3 additional parameters
            {
                fprintf(fp, "\n /t usage: %s \n", T_SWITCH_USAGE_STRING );
                System->errorParam(fp);
                goto Shutdwn;
            }

            nArg++;
            targetCycles = strtol(argv[nArg], 0, 0); // TODO: should targetCycles be sanitized?

            nArg++;
            targetStepAmount = strtol(argv[nArg], 0, 0); // TODO: should targetStepAmount be sanitized?

            nArg++;
            targetSearchDirection = strtol(argv[nArg], 0, 0); // TODO: should targetSearchDirection be sanitized?

            fprintf(fp, "\n Target cycles : %d\n", targetCycles);
            fprintf(fp, "\n Step amount   : %d ms \n", targetStepAmount);
            fprintf(fp, "\n Search direction is %d\n", targetSearchDirection);

            if(targetSearchDirection > 1)
            {
                targetSearchDirection = 1;
                fprintf(fp, "\n Target Search Direction clamped to : %d \n", targetSearchDirection);
            }
            else if(targetSearchDirection < -1)
            {
                targetSearchDirection = -1;
                fprintf(fp, "\n Target Search Direction clamped to : %d \n", targetSearchDirection);
            }
            else
            {
                fprintf(fp, "\n Target Search Direction : %d \n", targetSearchDirection);
            }
        }
        else if (strcmp(argv[nArg], "/r") == 0)//random
        {
            if( tMode != SINGLE_PASS )
            {
                fprintf(fp, "\n /r cannot be used with /c or /t \n" );
                System->errorParam(fp);
                goto Shutdwn;
            }
            tMode = RANDOM;

            if(nArg+3 >= argc) // needs 3 additional parameters
            {
                fprintf(fp, "\n /r usage: %s \n", R_SWITCH_USAGE_STRING );
                System->errorParam(fp);
                goto Shutdwn;
            }

            nArg++;
            randomMaxTriangles = strtol(argv[nArg], 0, 0);
            
            nArg++;
            randomMaxDelayInterval = strtol(argv[nArg], 0, 0);
            
            nArg++;
            randomHoldInterval = strtol(argv[nArg], 0, 0);

            fprintf(fp, "\n Max Triangles         : %d \n", randomMaxTriangles);
            fprintf(fp, "\n Max Delay Interval    : %d \n", randomMaxDelayInterval);
            fprintf(fp, "\n HoldInterval          : %d \n", randomHoldInterval);
        }
        else if (strcmp(argv[nArg], "/p") == 0)//power cycles reset
        {
            if(nArg+1 >= argc) // needs one additional parameter
            {
                fprintf(fp, "\n /p usage: %s \n", P_SWITCH_USAGE_STRING );
                System->errorParam(fp);
                goto Shutdwn;
            }

            bResetCycles = true;

            nArg++;
            cycleCount = strtol(argv[nArg], 0, 0); // TODO: should cycleCount be sanitized?
        }
        else if (strcmp(argv[nArg], "/cs") == 0) // Clear stats before and after the Run
        {
            nArg++;
            doNotclearStat = (CLEAR_TOOL_STATS)strtol(argv[nArg], 0, 0);
            if(doNotclearStat > CLEAR_END_AND_START)
            {
                doNotclearStat = CLEAR_END_AND_START;
            }
            nArg++;
        }
        else if (strcmp(argv[nArg], "/h") == 0)//help
        {
            fprintf(fp, "\n /f forced P state\n" );
            fprintf(fp, "\n /i initial settings usage: %s \n", I_SWITCH_USAGE_STRING );
            fprintf(fp, "\n /c calibarte usage: %s \n", C_SWITCH_USAGE_STRING );
            fprintf(fp, "\n /t target transitions usage: %s \n", T_SWITCH_USAGE_STRING );
            fprintf(fp, "\n /r random testing usage: %s \n", R_SWITCH_USAGE_STRING );
            fprintf(fp, "\n /p usage: %s \n", P_SWITCH_USAGE_STRING );
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

    switch(tMode)
    {
    case SINGLE_PASS:
        System->Run(forced, doNotclearStat); //no optional parameters
        break;
    case CALIBRATE:
        System->Run(forced, doNotclearStat);
        break;
    case RANDOM:
        System->Run_MinMaxHold(forced, randomMaxTriangles, randomMaxDelayInterval, randomHoldInterval, doNotclearStat);
        break;
    case TARGET:
        System->Run_TargetCycles(forced, targetCycles, targetStepAmount, targetSearchDirection, doNotclearStat);
        break;
    }
     
    if (DoCalibrate)
    {
        System->Calibrate(0, bIsFullscreen, CalibrateTime);
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