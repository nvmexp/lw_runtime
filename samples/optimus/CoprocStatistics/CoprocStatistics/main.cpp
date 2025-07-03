#include "stdafx.h"

#include "CoprocStatistics.h"
#include <InitGuid.h>
#include "ntddvdeo.h"
#include "cfgmgr32.h"

// Copied from the winblue WDK's sdkddkver.h
#define _WIN32_WINNT_WINBLUE                0x0603

#define LW_COPROC_STATE_CHANGE_LOG_COUNT    32

#define COPROC_DEFAULT_LIFE_EXPECTANCY    157680000
#define SEC_TO_MILLISEC_UNITS           1000ll  // seconds to milliseconds units

int _tmain(int argc, _TCHAR* argv[])
{

    wchar_t* str = isWinBlue() ? L"Running on Windows Blue\n" : L"Running on OS < Windows Blue \n";
    print(true, true, NULL, str);
	bool bResetStats = false;
    bool bAbbreviated = false;
    bool bResetCycles = false;
    LwU64 cycleCount = 0;
    LwU64 maxLoop = 1; // by default stats will be printed for 1 time.
    LwU32 timeLapse = 0;

	//_asm int 3;
    _TCHAR* target = NULL;
    _TCHAR cmdLine[MAX_PATH];

    int i = 1;

    for (; i<argc; ++i)
    {
        if (!_tcscmp(argv[i], _T("/s")) ||
            !_tcscmp(argv[i], _T("/S")) )
        {
            if( !DisableGC6())
            {
                print(true, true, NULL, L"Disable GC6 failed \n" );
                return 0;
            }
        }
        else if (!_tcscmp(argv[i], _T("/t")) ||
            !_tcscmp(argv[i], _T("/T")) )
        {
            if( !ResetTest())
            {
                print(true, true, NULL, L"Reset GC6 Test failed\n" );
                return 0;
            }
            print(true, true, NULL, L"Reset GC6 test was called. Not collecting stats \n");
            return 0;
        }
        else if (!_tcscmp(argv[i], _T("/p")) ||
            !_tcscmp(argv[i], _T("/P")) )
        {
            // By using atexit, the DisplayEscape_Pause() function will be called
            // automatically when the program exits.
            atexit(DisplayEscape_Pause);
        }
        else if (!_tcscmp(argv[i], _T("/w")) ||
            !_tcscmp(argv[i], _T("/W")) )
        {   // pause immediately, instead of at exit
            DisplayEscape_Pause();
        }
        else if (!_tcscmp(argv[i], _T("/r")) ||
            !_tcscmp(argv[i], _T("/R")) )
        {
            bResetStats = true;
        }
        else if (!_tcscmp(argv[i], _T("/b")) ||
            !_tcscmp(argv[i], _T("/B")) )
        {
            bAbbreviated = true;
        }
        else if (!_tcscmp(argv[i], _T("/c")) ||
            !_tcscmp(argv[i], _T("/C")) )
        {
            if( i < (argc - 1) )
            {
                bResetCycles = true;
                cycleCount = 0;
                if ( swscanf_s( argv[i+1], L"%llx", &cycleCount, sizeof(argv[i+1]) ) != 1 )
                {
                    bResetCycles = false;
                    break;
                }

                if( bResetCycles == true ) i++;
            }
        }
        else if(!_tcscmp(argv[i], _T("/l")) ||
            !_tcscmp(argv[i], _T("/L")) )
        {
            // validate arguments
            // CoprocStatistics.exe /l <number of loops> <time-lapse in loop> <summaried or details output>
            if(argc < 5)
            {
                print(true, true, NULL, L"Need more command line inputs\n");
                return 0;
            }
            i++;
            maxLoop = _wtoi(argv[i]);
            i++;
            timeLapse = _wtoi(argv[i]);
            i++;
            bAbbreviated = (_wtoi(argv[i]) == 0)? false : true;
            i++;
            bResetCycles = false;
            break;
        }
        else
        {
            break;
        }
    }

    for (; i<argc; ++i)
    {
        if( target == NULL )
        {
            target = argv[i];
            _tcsnccpy_s( cmdLine, target, sizeof(cmdLine) );
        }
        else
        {
            _tcsncat_s( cmdLine, L" ", sizeof(cmdLine) );
            _tcsncat_s( cmdLine, argv[i], sizeof(cmdLine));
        }
    }

    vector<GPU>                         gpuList;
    bool                                gc6SupportStatus = false;
    LwAPI_Status                        coprocInfoStatus = LWAPI_ERROR;
    if(!fetchGpuList(gpuList))
    {
        print(true, true, NULL, L"No GPU has been found\n");
        return 0;
    }
    if( target != NULL )
    {
        // clear cycle and stats for all GPU's
        for (std::vector<GPU>::iterator it = gpuList.begin() ; it != gpuList.end(); ++it)
        {
            GPU gpu = *it;
            if(bResetCycles)
            {
                if(showCoprocCycleInformation(gpu.hPhyGPU, bResetCycles, cycleCount,false, NULL))
                {
                    print(true, true, NULL, L"clearing cycle stats has been failed\n");
                }
            }
            if(bResetStats)
            {
                if(clearCoprocStats(gpu.hPhyGPU))
                {
                    print(true, true, NULL, L"clearing coproc stats has been failed\n");
                }
            }
        }
		HANDLE hJob = CreateJobObject(NULL, L"DisplayEscapeJob");
        if( hJob == NULL )
        {
            print(true, true, NULL, L"CreatJobObject failed\n" );
        }
        else
        {
            STARTUPINFO startupInfo;
            PROCESS_INFORMATION processInfo;

            ZeroMemory(&startupInfo, sizeof(startupInfo));
            startupInfo.cb = sizeof(startupInfo);
            ZeroMemory(&processInfo, sizeof(processInfo));

            if( !CreateProcess( NULL, cmdLine, NULL, NULL, TRUE, CREATE_SUSPENDED | CREATE_BREAKAWAY_FROM_JOB, NULL, NULL, &startupInfo, &processInfo ) )
            {
                print(true, true, NULL, L"CreateProcess failed.\n");
               // dumpLastError();
            }
            else if ( !AssignProcessToJobObject(hJob, processInfo.hProcess) )
            {
                print(true, true, NULL, L"AssignProcessToJobObject failed\n");
               // dumpLastError();
            }
            else if( ResumeThread( processInfo.hThread ) == INFINITE )
            {
                print(true, true, NULL, L"ResumeThread failed\n" );
                dumpLastError();
            }

            HANDLE childProcess = processInfo.hProcess;

            while( childProcess )
            {
                // wait for the child process to exit
                WaitForSingleObject( childProcess, INFINITE );
                childProcess = 0;

                // the child process may have created more processes, so look for those
                JOBOBJECT_BASIC_PROCESS_ID_LIST pidList = {0};
                if (!QueryInformationJobObject(hJob, JobObjectBasicProcessIdList, &pidList, sizeof(pidList), NULL))
                {
                    print(true, true, NULL, L"AssignProcessToJobObject failed\n");
                    //dumpLastError();
                }
                else if( pidList.NumberOfProcessIdsInList )
                {
                    childProcess = OpenProcess( SYNCHRONIZE, FALSE, (DWORD)(pidList.ProcessIdList[0]));
                    if( !childProcess )
                    {
                        print(true, true, NULL, L"OpenProcess failed, Extra child escaped!\n" );
                       // dumpLastError();
                    }
                }
            }
        }
    }
    for(int indexLoop = 1; indexLoop<= maxLoop; indexLoop++)
    {
        print(true, true, NULL, L"=========================================================================\n");
        for (std::vector<GPU>::iterator it = gpuList.begin() ; it != gpuList.end(); ++it)
        {
            GPU gpu = *it;
            print(true, true, NULL, L"*****************************************************************\nGPU Name\t\t:\t%S\n",gpu.gpuName);
            print(true, true, NULL, L"*****************************************************************\n");
            
            if(!showCoprocCycleInformation(gpu.hPhyGPU, bResetCycles, cycleCount, true, NULL))
            {
                print(true, true, NULL, L"Showing the coproc cycle is failed\n");
            }
            print(true, true, NULL, L"*****************************************************************\n");
            
            getAndShowCoprocInfo(gpu.hPhyGPU, gc6SupportStatus, coprocInfoStatus, NULL);
            if(gc6SupportStatus)
            {
                getAndShowGC6Statistics(gpu.hPhyGPU, bAbbreviated, NULL);
                print(true, true, NULL, L"*****************************************************************\n\n");
            }
            getAndShowGOLDStatistics(gpu.hPhyGPU, bAbbreviated, bResetStats, NULL);
            print(true, true, NULL, L"*****************************************************************\n\n");
        }
        print(true, true, NULL, L"=========================================================================\n");
        Sleep(timeLapse);
    }
    return 0;
}