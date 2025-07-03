/***************************************************************************\
|*                                                                           *|
|*       Copyright 1993-2015 LWPU, Corporation.  All rights reserved.      *|
|*                                                                           *|
|*     NOTICE TO USER:   The source code  is copyrighted under  U.S. and     *|
|*     international laws.  Users and possessors of this source code are     *|
|*     hereby granted a nonexclusive,  royalty-free copyright license to     *|
|*     use this code in individual and commercial software.                  *|
|*                                                                           *|
|*     Any use of this source code must include,  in the user dolwmenta-     *|
|*     tion and  internal comments to the code,  notices to the end user     *|
|*     as follows:                                                           *|
|*                                                                           *|
|*       Copyright 1993-2013 LWPU, Corporation.  All rights reserved.      *|
|*                                                                           *|
|*     LWPU, CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY     *|
|*     OF  THIS SOURCE  CODE  FOR ANY PURPOSE.  IT IS  PROVIDED  "AS IS"     *|
|*     WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.  LWPU, CORPOR-     *|
|*     ATION DISCLAIMS ALL WARRANTIES  WITH REGARD  TO THIS SOURCE CODE,     *|
|*     INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGE-     *|
|*     MENT,  AND FITNESS  FOR A PARTICULAR PURPOSE.   IN NO EVENT SHALL     *|
|*     LWPU, CORPORATION  BE LIABLE FOR ANY SPECIAL,  INDIRECT,  INCI-     *|
|*     DENTAL, OR CONSEQUENTIAL DAMAGES,  OR ANY DAMAGES  WHATSOEVER RE-     *|
|*     SULTING FROM LOSS OF USE,  DATA OR PROFITS,  WHETHER IN AN ACTION     *|
|*     OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF     *|
|*     OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.     *|
|*                                                                           *|
|*     U.S. Government  End  Users.   This source code  is a "commercial     *|
|*     item,"  as that  term is  defined at  48 C.F.R. 2.101 (OCT 1995),     *|
|*     consisting  of "commercial  computer  software"  and  "commercial     *|
|*     computer  software  documentation,"  as such  terms  are  used in     *|
|*     48 C.F.R. 12.212 (SEPT 1995)  and is provided to the U.S. Govern-     *|
|*     ment only as  a commercial end item.   Consistent with  48 C.F.R.     *|
|*     12.212 and  48 C.F.R. 227.7202-1 through  227.7202-4 (JUNE 1995),     *|
|*     all U.S. Government End Users  acquire the source code  with only     *|
|*     those rights set forth herein.                                        *|
|*                                                                           *|
\***************************************************************************/


#include "stdafx.h"

#include "CoprocStatistics.h"
#include <InitGuid.h>
#include "ntddvdeo.h"
#include "cfgmgr32.h"
#include "lwDbg.h"

// Copied from the winblue WDK's sdkddkver.h
#define _WIN32_WINNT_WINBLUE                0x0603

void print(bool printToDebug, bool printToFile, FILE *outFile, wchar_t *format, ...)
{
    wchar_t outString[1024];

    va_list arguments;

    va_start(arguments, format);
    vswprintf_s(outString, sizeof(outString)/sizeof(outString[0]), format, arguments);
    va_end(arguments);

    if(printToDebug) 
        OutputDebugStringW( outString );

    // print in CMD
    fputws( outString, stdout);

    if(printToFile)
    {
        if(outFile)
            fputws( outString, outFile );
        else
            OutputDebugStringW(L"Output File Stream not specified");
    }
}

bool ResetTest()
{
    PFND3DKMT_ESCAPE pEscape = NULL;
    D3DKMT_HANDLE hAdapter = NULL;

    if(!populate(&pEscape, &hAdapter))
    {
        print(true, true, NULL, L"populate(hAdapter, pEscape) failed");
        return false;
    }

    LWL_ESC_COMMON_SET_GC6_RESET_PENDING resetTest;
    memset(&resetTest, 0, sizeof(resetTest) );

    LWL_ESC_INIT( &resetTest, sizeof(LWL_ESC_COMMON_SET_GC6_RESET_PENDING), LWL_PRIV_CALLER_ID_WILDCARD, LWL_ESC_ID_COMMON_SET_GC6_RESET_TEST );

    D3DKMT_ESCAPE kmtEscape;
    memset( &kmtEscape, 0, sizeof(kmtEscape) );
    kmtEscape.hAdapter = hAdapter;
    kmtEscape.hDevice = NULL;
    kmtEscape.Type = D3DKMT_ESCAPE_DRIVERPRIVATE;
    kmtEscape.Flags.HardwareAccess = 0;
    kmtEscape.pPrivateDriverData = &resetTest;
    kmtEscape.PrivateDriverDataSize = sizeof(LWL_ESC_COMMON_SET_GC6_RESET_PENDING);

    NTSTATUS status = pEscape( &kmtEscape );
    if( !SUCCEEDED(status) )
    {
        print(true, true, NULL, L"pEscape failed\n" );
        return false;
    }

    return true;
}

// Suspend GC6 by grabbing a refcount for the duration of the process. 
// Meant for testing to see if GC6 is affecting function or perf of some app
bool DisableGC6()
{
    vector<GPU> gpuList;
    if(!fetchGpuList( gpuList))
    {
        print(true, true, NULL, L"Not able to fatch any GPU\n");
        return false;
    }
    LwAPI_Status Status = LWAPI_ERROR;

    GPU gpu;
    int gpuIndex=1,select=0;
    LWAPI_COPROC_REGISTER_PROCESS coprocData  = {0};
    coprocData.version                        = LWAPI_COPROC_REGISTER_PROCESS_VER1;
    coprocData.processSettings                = LWL_COPROC_REGISTER_PROCESS_DISABLE_GC6;
    coprocData.processSettingsMask            = LWL_COPROC_REGISTER_PROCESS_DISABLE_GC6;
    for (std::vector<GPU>::iterator it = gpuList.begin() ; it != gpuList.end(); ++it)
    {
        gpu = *it;
        wprintf(L"GPU %d :\t\t:\t%S\n", gpuIndex, gpu.gpuName);
        gpuIndex++;
    }
    print(true, true, NULL, L"\nSelect GPU number to Enable/Disble the GC6 for pertilwlar GPU.\nPress 0 to Enable/Disble the GC6 for all GPU's\n");
    scanf_s("%d", &select);
    gpuIndex = 0;
    for (std::vector<GPU>::iterator it = gpuList.begin() ; it != gpuList.end(); ++it)
    {
        gpuIndex++;
        if(select !=0 && select != gpuIndex) continue;
        gpu = *it;
        Status = LwAPI_Coproc_RegisterProcess(gpu.hPhyGPU, &coprocData);
        if(Status != LWAPI_OK)
        {
            print(true, true, NULL, L"Selected GPU %d: Fail to Enable/Disble the GC6 with Error: %d\n", gpuIndex,Status);
        }
        else
        {
            print(true, true, NULL, L"Selected GPU %d: Pass to Enable/Disble the GC6\n", gpuIndex);
        }
    }
    print(true, true, NULL, L"\n\n");
    return true;
}

void formatTime( wchar_t *timeString, LwU64 time, size_t numChars )
{
    LwU32 time32;

    if( time >= 1000ll * 1000ll * 1000ll * 60ll * 60ll * 24ll ) // dy
    {
        time32 = (LwU32)(time / (1000ll * 1000ll * 1000ll * 60ll * 60ll * 24ll));
        swprintf_s( timeString, numChars, L"%4ddy", time32 );
    }
    else if( time >= 1000ll * 1000ll * 1000ll * 60ll * 60ll ) // hr
    {
        time32 = (LwU32)(time / (1000ll * 1000ll * 1000ll * 60ll * 60ll));
        swprintf_s( timeString, numChars, L"%4dhr", time32 );
    }
    else if( time >= 1000ll * 1000ll * 1000ll * 60ll ) //  m
    {
        time32 = (LwU32)(time / (1000ll * 1000ll * 1000ll * 60ll));
        swprintf_s( timeString, numChars, L"%4d m", time32 );
    }
    else if( time >= 1000ll * 1000ll * 1000ll ) //  s
    {
        time32 = (LwU32)(time / (1000ll * 1000ll * 1000ll));
        swprintf_s( timeString, numChars, L"%4d s", time32 );
    }
    else if( time >= 1000ll * 1000ll ) // ms
    {
        time32 = (LwU32)(time / (1000ll * 1000ll));
        swprintf_s( timeString, numChars, L"%4dms", time32 );
    }
    else if( time > 1000ll ) //us
    {
        time32 = (LwU32) time / 1000ll;
        swprintf_s( timeString, numChars, L"%4dus", time32 );
    }
    else // nano seconds
    {
        time32 = (LwU32) time;
        swprintf_s( timeString, numChars, L"%4dns", time32 );
    }
}

void DebugDumpHistogram( LwS32 *histogram, LwU32 timeIncrement, FILE *outFile)
{
    wchar_t timeStartString[128];
    wchar_t timeEndString[128];

    for( unsigned int i=0; i < LW_COPROC_STATE_CHANGE_LOG_COUNT; i++ )
    {
        if(i == 0) // special bucket implying it took no duration for this event
        {
            print(true, true, outFile, L"%4d, %s - %s\n", histogram[i], L"0ns", L"0ns");
            if(histogram[i] > 0)
            {
                print(true, true, outFile, L"-> WOW, there was an event which took 0 duration\n");
            }
            continue;
        }

        int effectiveBucket  = i - 1; // this aclwrately represents the tick duration since bucket 0 is reserved
        LONGLONG tickStart = 1 << effectiveBucket;
        LONGLONG tickEnd = (1 << (effectiveBucket+1)) - 1;

        LONGLONG timeStart = effectiveBucket == 0 ? 1 : tickStart * timeIncrement;
        LONGLONG timeEnd = tickEnd * timeIncrement;

        formatTime( timeStartString, timeStart, sizeof(timeStartString)/sizeof(timeStartString[0]) );
        formatTime( timeEndString, timeEnd, sizeof(timeEndString)/sizeof(timeEndString[0]) );
        // we just use i - 1 for callwlating the correct duration
        print(true, true, outFile, L"%4d, %s - %s\n", histogram[i], timeStartString, timeEndString); 
    }
}
wchar_t *coprocStateToString( LWL_COPROC_POWER_STATE state )
{
    switch( state )
    {
    case     LWL_COPROC_POWER_STATE_UNKNOWN :
        return L"UNKNOWN";
    case     LWL_COPROC_POWER_STATE_ON :
        return L"ON";
    case     LWL_COPROC_POWER_STATE_GOLD :
        return L"GCOFF";
    case     LWL_COPROC_POWER_STATE_EXITING_GOLD :
        return L"EXITING_GCOFF";
    case     LWL_COPROC_POWER_STATE_READY_GOLD :
        return L"READY_GOLD";
    case     LWL_COPROC_POWER_STATE_ENTERING_GOLD :
        return L"ENTERING_GCOFF";
    case     LWL_COPROC_POWER_STATE_WAITING_FOR_SVC :
        return L"WAITING_FOR_SVC";
    case     LWL_COPROC_POWER_STATE_GC6 :
        return L"GC6";
    case     LWL_COPROC_POWER_STATE_EXITING_GC6 :
        return L"EXITING_GC6";
    case     LWL_COPROC_POWER_STATE_READY_GC6 :
        return L"READY_GC6";
    case     LWL_COPROC_POWER_STATE_ENTERING_GC6 :
        return L"ENTERING_GC6";
    }

    return L"SomethingHasGoneReallyBadlyWrong";
}

wchar_t *coprocStateToString( LW_COPROC_POWER_STATE state )
{
    switch( state )
    {
    case     LW_COPROC_POWER_STATE_ERROR :
        return L"UNKNOWN";
    case     LW_COPROC_POWER_STATE_ON :
        return L"ON";
    case     LW_COPROC_POWER_STATE_GOLD :
        return L"GCOFF";
    case     LW_COPROC_POWER_STATE_EXITING_GOLD :
        return L"EXITING_GCOFF";
    case     LW_COPROC_POWER_STATE_READY_GOLD :
        return L"READY_GCOFF";
    case     LW_COPROC_POWER_STATE_ENTERING_GOLD :
        return L"ENTERING_GCOFF";
    case     LW_COPROC_POWER_STATE_WAITING_FOR_SVC :
        return L"WAITING_FOR_SVC";
    case     LW_COPROC_POWER_STATE_GC6 :
        return L"GC6";
    case     LW_COPROC_POWER_STATE_EXITING_GC6 :
        return L"EXITING_GC6";
    case     LW_COPROC_POWER_STATE_READY_GC6 :
        return L"READY_GC6";
    case     LW_COPROC_POWER_STATE_ENTERING_GC6 :
        return L"ENTERING_GC6";
    }

    return L"SomethingHasGoneReallyBadlyWrong";
}

void dumpLastError() 
{ 
    LPVOID lpMsgBuf; 
    DWORD dw = GetLastError(); 

    FormatMessage( 
        FORMAT_MESSAGE_ALLOCATE_BUFFER | 
        FORMAT_MESSAGE_FROM_SYSTEM, 
        NULL, 
        dw, 
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), 
        (LPTSTR) &lpMsgBuf, 
        0, NULL ); 
    print(true, true, NULL, L"%ls\n", lpMsgBuf );
    OutputDebugString( (LPTSTR)lpMsgBuf ); 

    LocalFree(lpMsgBuf); 
    return;
} 

bool populate(PFND3DKMT_ESCAPE* ppEscape, D3DKMT_HANDLE* phAdapter)
{
    if(isWinBlue()) print(true, true, NULL, L" !!! Warning: This is being run on winblue and will wake up the dGPU !!! \n");

    static HINSTANCE hInst = LoadLibrary( L"gdi32.dll" );
    if (hInst == NULL)
    {
        print(true, true, NULL, L"could not open gdi32.dll");
        return false;
    }

    static PFND3DKMT_OPENADAPTERFROMDEVICENAME pOpenAdapterFromDeviceName = (PFND3DKMT_OPENADAPTERFROMDEVICENAME)GetProcAddress(hInst, "D3DKMTOpenAdapterFromDeviceName");
    if (pOpenAdapterFromDeviceName == NULL)
    {
        print(true, true, NULL, L"could not GetProcAddress of D3DKMTOpenAdapterFromDeviceName");
        return false;
    }

    static D3DKMT_HANDLE hAdapter = OpenFirstLwidiaAdapter( pOpenAdapterFromDeviceName );
    if( phAdapter == NULL )
    {
        print(true, true, NULL, L"OpenFirstLwidiaAdapter failed");
        return false;
    }

    *phAdapter = hAdapter;

    static PFND3DKMT_ESCAPE pEscape = (PFND3DKMT_ESCAPE)GetProcAddress(hInst, "D3DKMTEscape");
    if (pEscape == NULL)
    {
        print(true, true, NULL, L"could not GetProcAddress of D3DKMTEscape");
        return false;
    }

    *ppEscape = pEscape;

    return true;
}

D3DKMT_HANDLE OpenFirstLwidiaAdapter( PFND3DKMT_OPENADAPTERFROMDEVICENAME pOpenAdapterFromDeviceName )
{
    D3DKMT_HANDLE hAdapter = NULL;
    WCHAR *pLwidiaDevice = L"\\\\\?\\pci#ven_10de";
    size_t len = wcsnlen( pLwidiaDevice, 100 );

    HDEVINFO hDeviceInfo = SetupDiGetClassDevs(&GUID_DISPLAY_DEVICE_ARRIVAL, NULL, NULL, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE );
    if (hDeviceInfo == ILWALID_HANDLE_VALUE)
    {
        print(true, true, NULL, L"could not SetupDiGetClassDevs");
        return hAdapter;
    }

    SP_DEVICE_INTERFACE_DATA DeviceInterfaceData;
    DeviceInterfaceData.cbSize = sizeof(DeviceInterfaceData);

    //now enum through all the devices in that info set
    for(DWORD MemberIndex = 0; SetupDiEnumDeviceInterfaces(hDeviceInfo,NULL, &GUID_DISPLAY_DEVICE_ARRIVAL, MemberIndex++, &DeviceInterfaceData); )
    {
        DWORD dwBufferSize = 0;

        // Get the interface detail buffer size
        SetupDiGetDeviceInterfaceDetail(hDeviceInfo, &DeviceInterfaceData, NULL, 0, &dwBufferSize, NULL);

        PSP_DEVICE_INTERFACE_DETAIL_DATA pInterfaceDetail = (PSP_DEVICE_INTERFACE_DETAIL_DATA) new char[dwBufferSize];

        if (pInterfaceDetail != NULL)
        {
            pInterfaceDetail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

            // Get the interface detail
            SetupDiGetDeviceInterfaceDetail(hDeviceInfo, &DeviceInterfaceData, pInterfaceDetail, dwBufferSize, NULL, NULL);

            // Is this the LW device - search for the hardware string inside DevicePath
            if( !wcsncmp(pInterfaceDetail->DevicePath, pLwidiaDevice, len ) )
            {
                D3DKMT_OPENADAPTERFROMDEVICENAME openArgsFromDeviceName;
                memset(&openArgsFromDeviceName, 0, sizeof(openArgsFromDeviceName));              

                openArgsFromDeviceName.pDeviceName = pInterfaceDetail->DevicePath;
                NTSTATUS status = pOpenAdapterFromDeviceName(&openArgsFromDeviceName);

                if( SUCCEEDED(status) )
                {
                    hAdapter = openArgsFromDeviceName.hAdapter;
                    delete (char*) pInterfaceDetail;
                    pInterfaceDetail = NULL;
                    break;
                }
                else
                {
                    print(true, true, NULL, L"failed to OpenAdapterFromDeviceName\n" );
                }
            }

            delete (char*) pInterfaceDetail;
            pInterfaceDetail = NULL;
        }
    }

    SetupDiDestroyDeviceInfoList(hDeviceInfo);

    return hAdapter;
}

bool isWinBlue()
{
    OSVERSIONINFOEXW osvi = { sizeof(osvi), 0, 0, 0, 0, {0}, 0, 0 };

    DWORDLONG        const dwlConditionMask = VerSetConditionMask(
        VerSetConditionMask(
        VerSetConditionMask(
            0, VER_MAJORVERSION, VER_GREATER_EQUAL),
               VER_MINORVERSION, VER_GREATER_EQUAL),
               VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL);

    // HIBYTE(_WIN32_WINNT_WINBLUE), LOBYTE(_WIN32_WINNT_WINBLUE), 0
    osvi.dwMajorVersion = HIBYTE(_WIN32_WINNT_WINBLUE);
    osvi.dwMinorVersion = LOBYTE(_WIN32_WINNT_WINBLUE);
    osvi.wServicePackMajor = 0;

    return VerifyVersionInfoW(&osvi, VER_MAJORVERSION | VER_MINORVERSION | VER_SERVICEPACKMAJOR, dwlConditionMask) != FALSE;
} // isWinBlue

void updateRtd3SupportInfo(void  *pRtd3, LW_COPROC_INFO coprocInfo,
                           FILE                   *outFile)
{
    LW_RTD3_SUPPORT_INF_V1 rtd3SupportInfo = *((LW_RTD3_SUPPORT_INF_V1*)pRtd3);
    print(true, true, outFile, L"*****************************************************************\n");
    print(true, true, outFile, L"Rtd3 DR Key Info\n");
    print(true, true, outFile, L"\tRtd3 regkey Value(hex) : \t%x\n", rtd3SupportInfo.DRKey.value);
    print(true, true, outFile, L"\tRtd3 Enable : \t\t\t%u\n", rtd3SupportInfo.DRKey.details.enable);
    print(true, true, outFile, L"\tOverrideRM : \t\t\t%u\n", rtd3SupportInfo.DRKey.details.overrideRM);
    print(true, true, outFile, L"\tOverridePlaform : \t\t%u\n", rtd3SupportInfo.DRKey.details.overridePlaform);
    print(true, true, outFile, L"\tTurnONCorePower : \t\t%u\n", rtd3SupportInfo.DRKey.details.turnONCorePower);
    print(true, true, outFile, L"\tEnableASPM : \t\t\t%u\n", rtd3SupportInfo.DRKey.details.enableASPM);

    print(true, true, outFile, L"*****************************************************************\n");
    print(true, true, outFile, L"Rtd3 Support Info\n");
    print(true, true, outFile, L"\tDRKeyExists : \t\t\t%d\n", rtd3SupportInfo.config.DRKeyExists);
    print(true, true, outFile, L"\tplatformDSMSupport : \t\t%d\n", rtd3SupportInfo.config.platformDSMSupport);
    print(true, true, outFile, L"\tPerstDelayNegotiationFailed : \t%d\n", rtd3SupportInfo.config.PerstDelayNegotiationFailed);
    print(true, true, outFile, L"\tChipSupport : \t\t\t%d\n", rtd3SupportInfo.config.chipSupport);
    print(true, true, outFile, L"\tGC6Support : \t\t\t%d\n", rtd3SupportInfo.config.GC6Support);
    print(true, true, outFile, L"\tGCOFFSupport : \t\t\t%d\n", rtd3SupportInfo.config.GCOFFSupport);
    print(true, true, outFile, L"\tAUXPowerGC6Negotiated : \t%d\n", rtd3SupportInfo.config.AUXPowerGC6Negotiated);
    print(true, true, outFile, L"\tAUXPowerGCOFFNegotiated : \t%d\n", rtd3SupportInfo.config.AUXPowerGCOFFNegotiated);
    print(true, true, outFile, L"\tCorePowerLwtAllowed : \t\t%d\n", rtd3SupportInfo.config.corePowerLwtAllowed);
    print(true, true, outFile, L"\tPowerManageWithxUSB : \t\t%d\n", rtd3SupportInfo.config.powerManageWithxUSB);
    print(true, true, outFile, L"\tNoD3OnShortIdleWithxUSBPort : \t%d\n", rtd3SupportInfo.config.noD3OnShortIdleWithxUSBPort);
    print(true, true, outFile, L"\tNoD3WithxUSBDeviceConnect : \t%d\n", rtd3SupportInfo.config.noD3WithxUSBDeviceConnect);
    print(true, true, outFile, L"\tNoD3OnWithxUSBAtD0State : \t%d\n", rtd3SupportInfo.config.noD3OnWithxUSBAtD0State);
    print(true, true, outFile, L"\tNoD3OnWithHDAAtD0State : \t%d\n", rtd3SupportInfo.config.noD3OnWithHDAAtD0State);
    print(true, true, outFile, L"\tGC6TotalBoardPowerMilliWatts : \t%d\n", rtd3SupportInfo.GC6TotalBoardPowerMilliWatts);
    print(true, true, outFile, L"\tGCOffTotalBoardPowerMilliWatts :%d\n", rtd3SupportInfo.GCOffTotalBoardPowerMilliWatts);
    print(true, true, outFile, L"\tPerstDelayMicroSecs : \t\t%d\n", rtd3SupportInfo.perstDelayMicroSecs);

    if(coprocInfo.version >= LW_COPROC_INFO_VER_7)
    {
        LW_RTD3_SUPPORT_INF_V2 rtd3SupportInfo = *((LW_RTD3_SUPPORT_INF_V2*)pRtd3);
        PCIEPOWERCONTROL_KEY_ORDER order = rtd3SupportInfo.identifiedKeyOrder;
        switch( order )
        {
            case PCIEPOWERCONTROL_KEY_ORDER_CHIPSET_GPU_ID:
                {
                    print(true, true, outFile, L"\tRegKeyIdentifiedOrder : \tPCIEPOWERCONTROL_KEY_ORDER_CHIPSET_GPU_ID\n");
                }
                break;
            case PCIEPOWERCONTROL_KEY_ORDER_WILDCARD:
                {
                    print(true, true, outFile, L"\tRegKeyIdentifiedOrder : \tPCIEPOWERCONTROL_KEY_ORDER_WILDCARD\n");
                }
                break;
            case PCIEPOWERCONTROL_KEY_ORDER_CHIPSET_ID:
                {
                    print(true, true, outFile, L"\tRegKeyIdentifiedOrder : \tPCIEPOWERCONTROL_KEY_ORDER_CHIPSET_ID\n");
                }
                break;
            default:
                {
                    print(true, true, outFile, L"\tRegKeyIdentifiedOrder : \tPCIEPOWERCONTROL_KEY_ORDER_NONE\n");
                }
                break;
        }
		PCIEPOWERCONTROL_KEY_LOCATION location = rtd3SupportInfo.identifiedKeyLocation;
        switch( location )
        {
            case PCIEPOWERCONTROL_KEY_LOCATION_GLOBAL:
                {
                   print(true, true, outFile, L"\tRegKeyIdentifiedLocation : \tPCIEPOWERCONTROL_KEY_LOCATION_GLOBAL\n"); 
                }
                break;
            case PCIEPOWERCONTROL_KEY_LOCATION_ADAPTER:
                {
                    print(true, true, outFile, L"\tRegKeyIdentifiedLocation : \tPCIEPOWERCONTROL_KEY_LOCATION_ADAPTER\n");
                }
                break;
            case PCIEPOWERCONTROL_KEY_LOCATION_UEFI:
                {
                    print(true, true, outFile, L"\tRegKeyIdentifiedLocation : \tPCIEPOWERCONTROL_KEY_LOCATION_UEFI\n");
                }
                break;
            case PCIEPOWERCONTROL_KEY_LOCATION_DR:
                {
                    print(true, true, outFile, L"\tRegKeyIdentifiedLocation : \tPCIEPOWERCONTROL_KEY_LOCATION_DR\n");
                }
                break;    
            default:
                {
                    print(true, true, outFile, L"\tRegKeyIdentifiedLocation : \tPCIEPOWERCONTROL_KEY_LOCATION_NOT_PRESENT\n");
                }
                break;
        }
	}
} // updateRtd3SupportInfo

//*****************************************************************************
// Function:  getAndShowCoprocInfo()
//
// Routine Description:
//
//      Get and show the coproc info using LwAPI_Coproc_GetCoprocInfoEx.
//      GC6 support status. If GC6 is not supported, show the reason.
//      Display RTD3 support info and support type.
//
// Arguments:
//
//      [IN]  hPhyGPU           -   Handle of GPU
//      [OUT] gc6SupportStatus  -   GC6 support status
//      [OUT] coprocInfoStatus  -   coproc API status
//      [IN]  outFile           -   Output File pointer
//
// Return Value:
//
//      None
//
//*****************************************************************************

void getAndShowCoprocInfo(LwPhysicalGpuHandle   hPhyGPU, 
                          bool&                 gc6SupportStatus, 
                          LwAPI_Status          coprocInfoStatus,
                          FILE                  *outFile)
{
    LW_COPROC_INFO    coprocInfo       =   {0};
    LW_COPROC_INFO_V7 coprocInfoVer7   =   {0};
	LW_COPROC_INFO_V6 coprocInfoVer6   =   {0};
    LW_COPROC_INFO_V4 coprocInfoVer4   =   {0};
    coprocInfoVer7.version             =   LW_COPROC_INFO_VER_7;
    print(true, true, outFile, L"Coproc Info: \n");
	coprocInfoStatus                   =   LwAPI_Coproc_GetCoprocInfoEx(hPhyGPU, (LW_COPROC_INFO*)&coprocInfoVer7);
    if(coprocInfoStatus != LWAPI_OK)
    {   
        // fall back to version LW_COPROC_INFO_VER_6.
	    coprocInfoVer6.version             =   LW_COPROC_INFO_VER_6;
        coprocInfoStatus                   =   LwAPI_Coproc_GetCoprocInfoEx(hPhyGPU, (LW_COPROC_INFO*)&coprocInfoVer6);
        if(coprocInfoStatus != LWAPI_OK)
        {
            // falll back to version LW_COPROC_INFO_VER_4.
            coprocInfoVer4.version             =   LW_COPROC_INFO_VER_4;
            coprocInfoStatus = LwAPI_Coproc_GetCoprocInfoEx(hPhyGPU, (LW_COPROC_INFO*)&coprocInfoVer4);
            if(coprocInfoStatus == LWAPI_OK)
            {
                coprocInfo.version = LW_COPROC_INFO_VER_4;    
                memcpy_s(&coprocInfo, sizeof(LW_COPROC_INFO_V4), &coprocInfoVer4, sizeof(LW_COPROC_INFO_V4));
                print(true, true, outFile, L"Coproc Info Ver: LW_COPROC_INFO_VER_4\n");
            }
            else
            {
                print(true, true, outFile, L"failed coproc Info Status\n");
                return;
            }
        }
        else
        {
            coprocInfo.version = LW_COPROC_INFO_VER_6;
            memcpy_s(&coprocInfo, sizeof(LW_COPROC_INFO_V6), &coprocInfoVer6, sizeof(LW_COPROC_INFO_V6));
            print(true, true, outFile, L"Coproc Info Ver: LW_COPROC_INFO_VER_6\n");
        }
    }
    else
    {
        coprocInfo.version = LW_COPROC_INFO_VER_7;
        memcpy_s(&coprocInfo, sizeof(LW_COPROC_INFO_V7), &coprocInfoVer7, sizeof(LW_COPROC_INFO_V7));
        print(true, true, outFile, L"Coproc Info Ver: LW_COPROC_INFO_VER_7\n");
    }

    print(true, true, outFile, L"\tcoprocStatusMask : \t%d\n", coprocInfo.coprocStatusMask);

    // Print the GPU current state.
    if( coprocInfo.lwrrentState == LW_COPROC_STATE_DGPU_GOLD )
    {
        print(true, true, outFile, L"\tLwrrent GPU State : \tLW_COPROC_STATE_DGPU_GCOFF\n");
    }
    else if( coprocInfo.lwrrentState == LW_COPROC_STATE_DGPU_ON )
    {
        print(true, true, outFile, L"\tLwrrent GPU State : \tLW_COPROC_STATE_DGPU_ON\n");
    }
    else if (coprocInfo.lwrrentState == LW_COPROC_STATE_DGPU_GC6)
    {
        print(true, true, outFile, L"\tLwrrent GPU State : \tLW_COPROC_STATE_DGPU_GC6\n");
    }
    // Print the GPU current state.
    if( coprocInfo.lastState == LW_COPROC_STATE_DGPU_GOLD )
    {
        print(true, true, outFile, L"\tLast GPU State : \tLW_COPROC_STATE_DGPU_GCOFF\n");
    }
    else if( coprocInfo.lastState == LW_COPROC_STATE_DGPU_ON )
    {
        print(true, true, outFile, L"\tLast GPU State : \tLW_COPROC_STATE_DGPU_ON\n");
    }
    else if (coprocInfo.lastState == LW_COPROC_STATE_DGPU_GC6)
    {
        print(true, true, outFile, L"\tLast GPU State : \tLW_COPROC_STATE_DGPU_GC6\n");
    }

    if(coprocInfo.JTFlags & LW_JT_FLAGS_GC6_ENABLED)
    {
        if (coprocInfo.JTFlags & LW_JT_FLAGS_SUPPORTS_GC6_TDR)
        {
            gc6SupportStatus = true;
            print(true, true, outFile, L"\tGC6 staus: \t\tSupported\n");
        }
    }
    else
    {
        print(true, true, outFile, L"GC6 is not supported with Reasons:");
        gc6SupportStatus                        = false;
        LW_DIAG_GC6_DEBUG_INFO pGC6DebugInfo    = {0};
        pGC6DebugInfo.version                   = LW_DIAG_GC6_DEBUG_INFO_VER;
        LwAPI_Status pGC6DebugStatus            = LwAPI_Diag_GetGC6DebugInfo(hPhyGPU, &pGC6DebugInfo);
        if(pGC6DebugStatus == LWAPI_OK)
        {
            if ((!(pGC6DebugInfo.gc6DebugInfo & LW_GC6_DEBUG_INFO_SBIOS_ENABLED)) && (!(pGC6DebugInfo.gc6DebugInfo & LW_GC6_DEBUG_INFO_VBIOS_FBCLAMP_ENABLED)))
            {
                print(true, true, outFile, L"\tGC6_STATUS_NOSUPPORT_VBIOS_SBIOS\n");
            }
            else if (!(pGC6DebugInfo.gc6DebugInfo & LW_GC6_DEBUG_INFO_SBIOS_ENABLED))
            {
                print(true, true, outFile, L"\tGC6_STATUS_NOSUPPORT_SBIOS\n");
            }
            else if(!(pGC6DebugInfo.gc6DebugInfo & LW_GC6_DEBUG_INFO_VBIOS_FBCLAMP_ENABLED))
            {
                print(true, true, outFile, L"\tGC6_STATUS_NOSUPPORT_VBIOS\n");
            }
        }
    }

    print(true, true, outFile, L"\tRTD3 Support Status: \t%d\n", coprocInfo.isRTD3Supported);

    switch (coprocInfo.platformD3ColdSupportType)
    {
        case LW_D3COLD_SUPPORT_GC_OFF_1_0:
            {
                print(true, true, outFile, L"\tD3 Support Type: \tLW_D3COLD_SUPPORT_GC_OFF_1_0\n");
            }
            break;
        case LW_D3COLD_SUPPORT_GC_OFF_3_0:
            {
                print(true, true, outFile, L"\tD3 Support Type: \tLW_D3COLD_SUPPORT_GC_OFF_3_0\n");
            }
            break;
        default:
            {
                print(true, true, outFile, L"\tD3 Support Type: \tLW_D3COLD_SUPPORT_NONE\n");
            }
            break;
    }
    // print device count on GPU
    print(true, true, outFile, L"\tDevice Count: \t\t%d\n", coprocInfo.deviceCount);
    print(true, true, outFile, L"\tAllocation Count: \t%d\n", coprocInfo.allocationCount);
    print(true, true, outFile, L"\tActive Entry Count: \t%d\n", coprocInfo.activeEntryPointCount);
    if(coprocInfo.version <= LW_COPROC_INFO_VER_4)
    {
        return;
    }
    // Below this is version 6 specific data points.
    //Type of SBIOS support for HDA config
    switch(coprocInfo.sbiosHDASupport)
    {
        case LW_SBIOS_OPTIMUS_HDA_CAP_ON_BOOT_HDA_DISABLED:
            {
                print(true, true, outFile, L"\tSBIOS support type for HDA config: \tLW_SBIOS_OPTIMUS_HDA_CAP_ON_BOOT_HDA_DISABLED\n");
            }
            break;
        case LW_SBIOS_OPTIMUS_HDA_CAP_ON_BOOT_HDA_PRESENT:
            {
                print(true, true, outFile, L"\tSBIOS support type for HDA config: \tLW_SBIOS_OPTIMUS_HDA_CAP_ON_BOOT_HDA_PRESENT\n");
            }
            break;
        default:
            {
                print(true, true, outFile, L"\tSBIOS support type for HDA config: \tLW_SBIOS_OPTIMUS_HDA_CAP_NONE\n");
            }
            break;
    }
    updateRtd3SupportInfo(&coprocInfo.RTD3SupportInfo, coprocInfo, outFile);
}

//*****************************************************************************
// Function:  getAndShowGC6Statistics()
//
// Routine Description:
//
//      Display the GC6 related information using LwAPI_GPU_GetGC6Statistics
//
// Arguments:
//
//      [IN]  hPhyGPU           -   Handle of GPU.
//      [IN]  bAbbreviated      -   if True: display abbreviated info of GC6.
//                                  if False : display GC6 info with histograms as well.
//      [IN]  outFile           -   Output File pointer.
//
// Return Value:
//
//      None
//
//*****************************************************************************
void getAndShowGC6Statistics(LwPhysicalGpuHandle hPhyGPU, bool bAbbreviated, FILE *outFile)
{
    LW_GPU_GC6_STATISTICS gc6Stats             = {0};

    // Call with the lwapi version 11.
    LW_GPU_GC6_STATISTICS_V11 gc6Stats_v11     = {0};
    gc6Stats_v11.version                       = LW_GPU_GC6_STATISTICS_VER11;
    gc6Stats_v11.bEnableLPWRInfo               = LW_GPU_FEATURE_LPWR_INFO_DISABLE;
	LwAPI_Status status = LwAPI_GPU_GetGC6Statistics(hPhyGPU, (LW_GPU_GC6_STATISTICS*) &gc6Stats_v11);
    
    if(status == LWAPI_OK)
    {
        memcpy_s(&gc6Stats, sizeof(LW_GPU_GC6_STATISTICS_V11), &gc6Stats_v11, sizeof(LW_GPU_GC6_STATISTICS_V11));
    }
    else
    {
        // if lwapi version 11 failed, fallback to lwapi version 9
        // this lwapi will work with r418_00 and pre r418_00 release driver.
        LW_GPU_GC6_STATISTICS_V9 gc6Stats_v9    = {0};
        gc6Stats_v9.version                       = LW_GPU_GC6_STATISTICS_VER9;
        gc6Stats_v9.bEnableLPWRInfo               = LW_GPU_FEATURE_LPWR_INFO_DISABLE;
        status = LwAPI_GPU_GetGC6Statistics(hPhyGPU, (LW_GPU_GC6_STATISTICS*) &gc6Stats_v9);
        if(status == LWAPI_OK)
        {
            memcpy_s(&gc6Stats, sizeof(LW_GPU_GC6_STATISTICS_V9), &gc6Stats_v9, sizeof(LW_GPU_GC6_STATISTICS_V9));
        }
    }
    
    if(status == LWAPI_OK)
    {
        LONGLONG totalTime;
        float fTotalTime;
        LONG totalGC6TransitionCount;
        gc6Stats.timeIncrement = gc6Stats.timeIncrement ? gc6Stats.timeIncrement : 1; // TODO: Fix this once timeInrement is being fixed from KMD
        totalTime = gc6Stats.lwrrentTime - gc6Stats.clearTime;
        
        if(totalTime == 0)
        {
            print(true, true, outFile, L"**GC6** \nclear time and current time are the same.\nNo new event has happened since, hence no stats to report.\n");
            return ;
        }
        fTotalTime = (float)totalTime * (float)gc6Stats.timeIncrement;    // from clock ticks to nano second units
        fTotalTime /= (float)(1000ll * 1000ll); // time in ms (milli seconds)
        print(true, true, outFile, L"*****************************************************************\nCoproc Time Stats: \n");
        print(true, true, outFile, L"\tclearTime: \t%-lld \n\tlwrrentTime: \t%lld \n\tticks: \t\t%d \n\tSX: \t\t%d \n \tTotal Time(ms): %f\n",
                gc6Stats.clearTime, gc6Stats.lwrrentTime, gc6Stats.timeIncrement, gc6Stats.timeInSx, fTotalTime );
        // total GC6 D3COLD+D3HOT count. This doesn't include the bounce GC6 counts.
        // GC6TransitionCount is D3COLD GC6 TransitionCount count.
        print(true, true, outFile, L"*****************************************************************\n\n**GC6**\n");
        if(gc6Stats.version >= LW_GPU_GC6_STATISTICS_VER11)
        {
            totalGC6TransitionCount = gc6Stats.exitD3HotGC6Count + gc6Stats.GC6TransitionCount;
            print(true, true, outFile, L"GC6 lwAPI LW_GPU_GC6_STATISTICS version 11\n");
        }
        else
        {
            totalGC6TransitionCount = gc6Stats.GC6TransitionCount;
            print(true, true, outFile, L"GC6 lwAPI LW_GPU_GC6_STATISTICS version 9\n");
        }
        print(true, true, outFile, L"*****************************************************************\n");
        print(true, true, outFile, L"GC6 Stats:\n \tLwrrent Mask:\t\t 0x%08lx \n\tLwmulative Mask:\t 0x%08lx \n\tD3COLD GC6 Transitions:\t\t %d \n"
                                   L"\tBounceCount:\t\t\t%d \n",
                gc6Stats.refCountGC6Mask,
                gc6Stats.refCountGC6MaskLwmulative,
                gc6Stats.GC6TransitionCount,
                gc6Stats.GC6BounceCount);

        if(gc6Stats.version >= LW_GPU_GC6_STATISTICS_VER11)
        {
            print(true, true, outFile, L"\tD3HOT GC6 transitions:\t\t%d\n",
                gc6Stats.exitD3HotGC6Count);
        }
        
        print(true, true, outFile, L"\nTimes in Ms: \n \t"
                L"%-8s,         \t percent, total, minTime, aveTime, maxTime \n \t"
                    L"In__Idle,          \t %f, %d, %d, %d, %d \n \t"
                    L"In___GC6,          \t %f, %d, %d, %d, %d \n \t"
                    L"EnterGC6,          \t %f, %d, %d, %d, %d \n \t"
                    L"Exit_GC6_D3COLD,   \t %f, %d, %d, %d, %d \n \t"
                    L"ExitEvnt,          \t %f, %d, %d, %d, %d \n \t"
                    L"RM_Exit_GC6_D3COLD,\t %f, %d, %d, %d, %d \n \t"
                    L"RM_SREnt,          \t %f, %d, %d, %d, %d \n \t"
                    L"RM__SREx,          \t %f, %d, %d, %d, %d \n",
                L"Function:",
                // Duration of idle time waiting for the transition to GC6
                (float)(totalGC6TransitionCount*gc6Stats.avgTimeInIdleToEnterGC6Ms * 100l) / fTotalTime, 
                gc6Stats.totTimeInIdleToEnterGC6Ms, gc6Stats.minTimeInIdleToEnterGC6Ms, gc6Stats.avgTimeInIdleToEnterGC6Ms, gc6Stats.maxTimeInIdleToEnterGC6Ms, // Idle time to reach GC6
                // Time spent in GC6
                (float)(totalGC6TransitionCount*gc6Stats.avgTimeInGC6Ms * 100l) / fTotalTime, 
                gc6Stats.totTimeInGC6Ms, gc6Stats.minTimeInGC6Ms, gc6Stats.avgTimeInGC6Ms, gc6Stats.maxTimeInGC6Ms, 
                // Time in the ENTERING_GC6 state
                (float)(totalGC6TransitionCount*gc6Stats.avgTimeEnteringGC6Ms * 100l) / fTotalTime, 
                gc6Stats.totTimeEnteringGC6Ms, gc6Stats.minTimeEnteringGC6Ms, gc6Stats.avgTimeEnteringGC6Ms, gc6Stats.maxTimeEnteringGC6Ms,
                // Time in the EXITING_GC6 state
                (float)(totalGC6TransitionCount*gc6Stats.avgTimeExitingGC6Ms * 100l) / fTotalTime, 
                gc6Stats.totTimeExitingGC6Ms, gc6Stats.minTimeExitingGC6Ms, gc6Stats.avgTimeExitingGC6Ms, gc6Stats.maxTimeExitingGC6Ms,
                // Time spent waiting for the thread to wake after the exit GC6 event is signalled
                (float)(totalGC6TransitionCount * gc6Stats.avgTimeExitEventGC6Ms * 100l) / fTotalTime, 
                gc6Stats.totTimeExitEventGC6Ms, gc6Stats.minTimeExitEventGC6Ms, gc6Stats.avgTimeExitEventGC6Ms, gc6Stats.maxTimeExitEventGC6Ms,
                // Time spent in the RM control call to exit GC6
                (float)(totalGC6TransitionCount * gc6Stats.avgTimeRmExitGC6DurationMs * 100l) / fTotalTime, 
                gc6Stats.totTimeRmExitGC6DurationMs, gc6Stats.minTimeRmExitGC6DurationMs, gc6Stats.avgTimeRmExitGC6DurationMs, gc6Stats.maxTimeRmExitGC6DurationMs,
                // Time spent entering self refresh
                (float)(totalGC6TransitionCount * gc6Stats.avgTimeRmEnterSRDurationMs * 100l) / fTotalTime, 
                gc6Stats.totTimeRmEnterSRDurationMs, gc6Stats.minTimeRmEnterSRDurationMs, gc6Stats.avgTimeRmEnterSRDurationMs, gc6Stats.maxTimeRmEnterSRDurationMs,
                // Time spent exiting self refresh
                (float)(totalGC6TransitionCount * gc6Stats.avgTimeRmExitSRDurationMs * 100l) / fTotalTime, 
                gc6Stats.totTimeRmExitSRDurationMs, gc6Stats.minTimeRmExitSRDurationMs, gc6Stats.avgTimeRmExitSRDurationMs, gc6Stats.maxTimeRmExitSRDurationMs);

        if(gc6Stats.version >= LW_GPU_GC6_STATISTICS_VER11)
        {
            print(true, true, outFile, 
                    L" \tExit_GC6_D3HOT,    \t %f, %d, %d, %d, %d \n"
                    L" \tRM_Exit_GC6_D3HOT, \t %f, %d, %d, %d, %d \n",
                    // Time spent D3 Hot GC6 exit
                    (float)(totalGC6TransitionCount * gc6Stats.avgTimeExitD3HotGC6DurationMs * 100l) / fTotalTime, 
                    gc6Stats.totTimeExitD3HotGC6DurationMs, gc6Stats.minTimeExitD3HotGC6DurationMs, gc6Stats.avgTimeExitD3HotGC6DurationMs, gc6Stats.maxTimeExitD3HotGC6DurationMs,
                    // RM Time spent D3 Hot GC6 exit
                    (float)(totalGC6TransitionCount * gc6Stats.avgTimeRmExitD3HotGC6DurationMs * 100l) / fTotalTime, 
                    gc6Stats.totTimeRmExitD3HotGC6DurationMs, gc6Stats.minTimeRmExitD3HotGC6DurationMs, gc6Stats.avgTimeRmExitD3HotGC6DurationMs, gc6Stats.maxTimeRmExitD3HotGC6DurationMs );
        }

        if(bAbbreviated) return ;

        print(true, true, outFile, L"\n*****************************************************************\nStay in GC6 Histogram\n" );
        DebugDumpHistogram( gc6Stats.GC6Histogram, gc6Stats.timeIncrement,outFile);

        print(true, true, outFile, L"\nEntering GC6 Histogram\n" );
        DebugDumpHistogram( gc6Stats.enteringGC6Histogram, gc6Stats.timeIncrement, outFile);

        print(true, true, outFile, L"\nD3COLD Exiting GC6 Histogram\n" );
        DebugDumpHistogram( gc6Stats.exitingGC6Histogram, gc6Stats.timeIncrement, outFile);

        if(gc6Stats.version >= LW_GPU_GC6_STATISTICS_VER11)
        {
            print(true, true, outFile,L"\nD3HOT Exiting GC6 Histogram\n" );
            DebugDumpHistogram( gc6Stats.exitD3HotGC6Histogram, gc6Stats.timeIncrement, outFile);
        }

        print(true, true, outFile,L"\nidle To Enter GC6 Histogram\n" );
        DebugDumpHistogram( gc6Stats.idleToEnterGC6MsHistogram, gc6Stats.timeIncrement, outFile);
    }
    else
    {
        print(true, true, outFile,L"\nGC6 lwapi failed\n" );
    }
    return ;
}


//*****************************************************************************
// Function:  getAndShowGOLDStatistics()
//
// Routine Description:
//
//      Display the GCOFF related information using LwAPI_Coproc_GetGoldStatisticsEx
//
// Arguments:
//
//      [IN]  hPhyGPU           -   Handle of GPU.
//      [IN]  bAbbreviated      -   if True: display abbreviated info of GC6.
//                                  if False : display GC6 info with histograms as well.
//      [IN]  bResetStats       -   if True: All GC6 and GCOFF stats will be reset to default values.
//                                  if False : No change happen.
//      [IN]  outFile           -   Output File pointer.
//
// Return Value:
//
//      None
//
//*****************************************************************************
void getAndShowGOLDStatistics(LwPhysicalGpuHandle hPhyGPU, bool bAbbreviated, bool bResetStats, FILE *outFile)
{
    LW_COPROC_GOLD_STATISTICS goldStats     = {0};

    // Call with the LW_COPROC_GOLD_STATISTICS lwapi with version 3.
    LW_COPROC_GOLD_STATISTICS_V3 goldStats_v3   = {0};
    goldStats_v3.version                        = LW_COPROC_GOLD_STATISTICS_VER3;
    LwAPI_Status status = LwAPI_Coproc_GetGoldStatisticsEx(hPhyGPU, bResetStats,(LW_COPROC_GOLD_STATISTICS*) &goldStats_v3);

    if(status == LWAPI_OK)
    {
        memcpy_s(&goldStats, sizeof(LW_COPROC_GOLD_STATISTICS_V3), &goldStats_v3, sizeof(LW_COPROC_GOLD_STATISTICS_V3));
    }
    else
    {
        LW_COPROC_GOLD_STATISTICS_V2 goldStats_v2     = {0};
        goldStats_v2.version                          = LW_COPROC_GOLD_STATISTICS_VER2;
        status = LwAPI_Coproc_GetGoldStatisticsEx(hPhyGPU, bResetStats,(LW_COPROC_GOLD_STATISTICS*) &goldStats_v2);
        if(status == LWAPI_OK)
        {
            memcpy_s(&goldStats, sizeof(LW_COPROC_GOLD_STATISTICS_V2), &goldStats_v2, sizeof(LW_COPROC_GOLD_STATISTICS_V2));
        }
    }
    

    if(status ==  LWAPI_OK)
    {
        wchar_t timeStringDelta[128],timeStringEvent[128];
        int i;
        int startIndex,endIndex;
        print(true, true, outFile, L"*****************************************************************\n");
        print(true, true, outFile, L"**GCOFF**\n");
        LONGLONG totalTime;
        float fTotalTime;
        LONG totalGCOFFCount = 0;
        goldStats.timeIncrement = goldStats.timeIncrement ? goldStats.timeIncrement : 1; // TODO: Fix this once timeInrement is being fixed from KMD
        totalTime = goldStats.lwrrentTime - goldStats.clearTime;

        if(totalTime == 0)
        {
            print(true, true, outFile, L"clear time and current time are the same.\nNo new event has happened since, hence no stats to report.\n");
            return ;
        }
        // total GCOFF D3COLD+D3HOT count.
        // dwGoldTransitionCount is D3COLD GCOFF TransitionCount count.
        if(goldStats.version >= LW_COPROC_GOLD_STATISTICS_VER3)
        {
            print(true, true, outFile, L"GCOFF lwAPI LW_COPROC_GOLD_STATISTICS version 3\n");
            totalGCOFFCount = goldStats.dwGoldTransitionCount + goldStats.exitD3HotGCOFFCount;
        }
        else
        {
            print(true, true, outFile, L"GCOFF lwAPI LW_COPROC_GOLD_STATISTICS version 2\n");
            totalGCOFFCount = goldStats.dwGoldTransitionCount;
        }
        fTotalTime = (float)totalTime * (float)goldStats.timeIncrement;    // from clock ticks to nano second units
        fTotalTime /= (float)(1000ll * 1000ll); // time in ms (milli seconds)
        print(true, true, outFile, L"*****************************************************************\n");

        print(true, true, outFile, L"GCOFF Stats:\n\tlwrrenotMask: \t\t0x%08lx \n\tlwmulativeMask: \t0x%08lx\n\tD3COLD GCOFF Transitions: \t%d \n",
            goldStats.refCountGoldMask, goldStats.refCountGoldMaskLwmulative,
            goldStats.dwGoldTransitionCount);
        if(goldStats.version >= LW_COPROC_GOLD_STATISTICS_VER3)
        {
            print(true, true, outFile, L"\tD3HOT GCOFF Transitions: \t%d \n",
                    goldStats.exitD3HotGCOFFCount);
        }
        print(true, true, outFile,L"\nTimes in milliseconds: \n"
                                 L"%-8s, \tpercent, total, minTime, avgTime, maxTime \n \t"
                                 L"In___GCOFF,          \t%f, %d, %d, %d, %d \n \t"
                                 L"EnterGCOFF,          \t%f, %d, %d, %d, %d \n \t"
                                 L"D3COLD_Exit_GCOFF,   \t%f, %d, %d, %d, %d \n \t"
                                 L"RM_D3COLD_Exit_GCOFF,\t%f, %d, %d, %d, %d \n",
                                 L"Function:",
            // In GCOFF duration stats
            (float)(totalGCOFFCount * goldStats.dwAvgTimeInGoldMs * 100l) / fTotalTime, // percent in GCOFF
            goldStats.totTimeInGoldMs, goldStats.dwMinTimeInGoldMs, goldStats.dwAvgTimeInGoldMs, goldStats.dwMaxTimeInGoldMs, 
            // Entering GCOFF duration stats
            (float)(totalGCOFFCount * goldStats.dwAvgTimeEnteringGoldMs * 100l) / fTotalTime,  // percent in Entering GCOFF
            goldStats.totTimeEnteringGoldMs, goldStats.dwMinTimeEnteringGoldMs, goldStats.dwAvgTimeEnteringGoldMs, goldStats.dwMaxTimeEnteringGoldMs, 
            // Exting from D3COLD GCOFF duration stats
            (float)(totalGCOFFCount * goldStats.dwAvgTimeExitingGoldMs * 100l) / fTotalTime, 
            goldStats.totTimeExitingGoldMs, goldStats.dwMinTimeExitingGoldMs, goldStats.dwAvgTimeExitingGoldMs, goldStats.dwMaxTimeExitingGoldMs,
            // Exting from D3COLD GCOFF RM duration stats
            (float)(totalGCOFFCount * goldStats.avgTimeRmExitGCOFFDurationMs * 100l) / fTotalTime, 
            goldStats.totTimeRmExitGCOFFDurationMs, goldStats.minTimeRmExitGCOFFDurationMs, goldStats.avgTimeRmExitGCOFFDurationMs, goldStats.maxTimeRmExitGCOFFDurationMs);

        if(goldStats.version >= LW_COPROC_GOLD_STATISTICS_VER3)
        {
            print(true, true, outFile, L"\tD3HOT_Exit_GCOFF,    \t%f, %d, %d, %d, %d \n \t"
                                       L"RM_D3HOT_Exit_GCOFF, \t%f, %d, %d, %d, %d \n \n",
                // Exting from D3HOT GCOFF duration stats
                (float)(totalGCOFFCount * goldStats.avgTimeExitD3HotGCOFFDurationMs * 100l) / fTotalTime, 
                goldStats.totTimeExitD3HotGCOFFDurationMs, goldStats.minTimeExitD3HotGCOFFDurationMs, goldStats.avgTimeExitD3HotGCOFFDurationMs, goldStats.maxTimeExitD3HotGCOFFDurationMs,
                // Exting from D3HOT GCOFF RM duration stats
                (float)(totalGCOFFCount * goldStats.avgTimeRmExitD3HotGCOFFDurationMs * 100l) / fTotalTime, 
                goldStats.totTimeRmExitD3HotGCOFFDurationMs, goldStats.minTimeRmExitD3HotGCOFFDurationMs, goldStats.avgTimeExitD3HotGCOFFDurationMs, goldStats.maxTimeRmExitD3HotGCOFFDurationMs);
        }
        if( bAbbreviated ) return;

        print(true, true, outFile, L"GCOFF Histogram\n" );
        DebugDumpHistogram( goldStats.goldHistogram, goldStats.timeIncrement, outFile );

        print(true, true, outFile, L"\nEntering GCOFF Histogram\n" );
        DebugDumpHistogram( goldStats.enteringGoldHistogram, goldStats.timeIncrement, outFile);

        print(true, true, outFile, L"\nD3COLD Exiting GCOFF Histogram\n" );
        DebugDumpHistogram( goldStats.exitingGoldHistogram, goldStats.timeIncrement, outFile);

        if(goldStats.version >= LW_COPROC_GOLD_STATISTICS_VER3)
        {
            print(true, true, outFile, L"\nD3HOT Exiting GCOFF Histogram\n" );
            DebugDumpHistogram( goldStats.exitD3HotGCOFFHistogram, goldStats.timeIncrement, outFile);
        }
        // add state change log here
        print(true, true, outFile, L"\n\n%12s, %5s, %13s, %8s, %10s\n", L"Timestamp",L"Delta",L"Event",L"Reason",L"Timeline\n" );

        startIndex = -1; endIndex=-1;

        for( i = 0; i <LW_COPROC_STATE_CHANGE_LOG_COUNT; i++ )
        {
            if( goldStats.stateChangeLog[i].timeStampMicroseconds )
            {
                if( (endIndex < 0) || (goldStats.stateChangeLog[i].timeStampMicroseconds > goldStats.stateChangeLog[endIndex].timeStampMicroseconds) )
                {
                    endIndex = i;
                }
                if( (startIndex < 0) || (goldStats.stateChangeLog[i].timeStampMicroseconds < goldStats.stateChangeLog[startIndex].timeStampMicroseconds) )
                {
                    startIndex = i;
                }
            }
        }
        if( startIndex >= 0 )
        {
            unsigned __int64 lastTime = goldStats.stateChangeLog[startIndex].timeStampMicroseconds;
            unsigned __int64 elapsedTime,EventTime;
            for( i = startIndex + LW_COPROC_STATE_CHANGE_LOG_COUNT; i > startIndex; i-- )
            {
                int targetIndex = i % LW_COPROC_STATE_CHANGE_LOG_COUNT; 
                if( goldStats.stateChangeLog[targetIndex].timeStampMicroseconds )
                {
                    elapsedTime = goldStats.stateChangeLog[targetIndex].timeStampMicroseconds - lastTime;
                    elapsedTime *= 10ll; // colwert to 100ns

                    // LW_COPROC_POWER_STATE_WAITING_FOR_SVC
                    wchar_t * targetPowerString = coprocStateToString( goldStats.stateChangeLog[targetIndex].coprocPowerState );
                    EventTime = goldStats.stateChangeLog[targetIndex].timeStampMicroseconds-goldStats.stateChangeLog[startIndex].timeStampMicroseconds;
                    EventTime *= 10ll;
                    formatTime( timeStringDelta, elapsedTime, sizeof(timeStringDelta)/sizeof(timeStringDelta[0]) );
                    formatTime( timeStringEvent, EventTime, sizeof(timeStringEvent)/sizeof(timeStringEvent[0]) );
                    print(true, true, outFile, L"%12llu, %5s, %13s, 0x%08lx, %5s\n", goldStats.stateChangeLog[targetIndex].timeStampMicroseconds,
                            timeStringDelta, targetPowerString, goldStats.stateChangeLog[targetIndex].coprocReason,timeStringEvent  );
                    lastTime = goldStats.stateChangeLog[targetIndex].timeStampMicroseconds;
                }
                else
                {
                    break;
                }
            }
        }
    }
    else
    {
        print(true, true, outFile, L"GCOFF API failed\n" );
    }
}

//*****************************************************************************
// Function:  showCoprocCycleInformation()
//
// Routine Description:
//
//      Display the coproc cycle related information using LwAPI_Coproc_ControlLimitedCycles
//
// Arguments:
//
//      [IN]  hPhyGPU          -   Handle of GPU.
//      [IN]  bResetCycles     -   Reset the cycle to cycleCount.
//      [IN]  cycleCount       -   reset the cycle to cycleCount.
//      [IN]  printStats       -   if True: All GC6 and GCOFF stats will be reset to default values.
//                                 if False : No change happen.
//      [IN]  outFile          -   Output File pointer.
//
// Return Value:
//
//      None
//
//*****************************************************************************
bool showCoprocCycleInformation(LwPhysicalGpuHandle hPhyGPU, bool bResetCycles, LwU64 cycleCount, bool printStats, FILE *outFile)
{
    LwAPI_Status status = LWAPI_ERROR;
    LWAPI_COPROC_CONTROL_LIMITED_CYCLES coprocData       = {0};
    coprocData.version                                   = LWAPI_COPROC_CONTROL_LIMITED_CYCLES_VER1;
    if(bResetCycles)
    {
        coprocData.cmd                                   = LWAPI_CMD_SET;
        coprocData.powerTransitionCount                  = cycleCount;
    }
    status = LwAPI_Coproc_ControlLimitedCycles(hPhyGPU ,&coprocData);
    if(status == LWAPI_OK )
    {
        if(!printStats)
        {
            return (status == LWAPI_OK);
        }

        print(true, true, outFile, L"cycles\n");
        print(true, true, outFile, L"\tPower Trasition count:\t%d\n",coprocData.powerTransitionCount);
        print(true, true, outFile, L"\tBudget:\t\t\t%d\n",coprocData.budget);
        print(true, true, outFile, L"\tTime Since Boot(ms):\t%d\n",coprocData.timeSinceBootMs);
    }
    else
    {
        print(true, true, outFile, L"Getting CYCLES has been failed\n");
    }
    return (status == LWAPI_OK);
}

//*****************************************************************************
// Function:  fetchGpuList()
//
// Routine Description:
//
//      Display the coproc cycle related information using LwAPI_Coproc_ControlLimitedCycles
//
// Arguments:
//
//      [IN]  hPhyGPU          -   Handle of GPU.
//      [IN]  bResetCycles     -   Reset the cycle to cycleCount.
//      [IN]  cycleCount       -   reset the cycle to cycleCount.
//      [IN]  printStats       -   if True: All GC6 and GCOFF stats will be reset to default values.
//                                 if False : No change happen.
//      [IN]  outFile          -   Output File pointer.
//
// Return Value:
//
//      None
//
//*****************************************************************************
bool fetchGpuList(vector<GPU> &gpuList)
{
    LwAPI_Status status                                     = LWAPI_ERROR;
	LwPhysicalGpuHandle hPhyGpu[LWAPI_MAX_PHYSICAL_GPUS]    = {0};
    LwU32 maxGpuCount = 0;

    status = LwAPI_EnumPhysicalGPUs( hPhyGpu, &maxGpuCount );

	if ( status == LWAPI_OK )
	{
		for( UINT gpuIndex = 0; gpuIndex < maxGpuCount; gpuIndex++ )
		{
			GPU gpu;
            gpu.hPhyGPU = hPhyGpu[gpuIndex];
            LwAPI_GPU_GetFullName(gpu.hPhyGPU, gpu.gpuName);
            gpuList.push_back(gpu);
		}
	}
    if(gpuList.size() == 0) // No GPU has been added in list
    {
        return false;
    }
    return true;
}

bool clearCoprocStats(LwPhysicalGpuHandle hPhyGPU)
{
    LW_COPROC_GOLD_STATISTICS goldStats  = {0};
    goldStats.version                    = LW_COPROC_GOLD_STATISTICS_VER2;
    LwAPI_Status status = LwAPI_Coproc_GetGoldStatisticsEx(hPhyGPU, true, &goldStats);
    
    return (status==LWAPI_OK);
}
