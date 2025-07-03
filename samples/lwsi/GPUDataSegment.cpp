#include "stdafx.h"
#include "GPUDataSegment.h"
#include "LwsiVer.h"
#include <iostream> // For protobuf serialization...
#include <fstream>  // ...to file

CGPUDataSegment::CGPUDataSegment(const char *szFilename):CDataSegment(szFilename)
{
    m_ppb = new lwsi_pb::GpuDataSegment();

    ZeroMemory(&m_osInfo, sizeof(OSVERSIONINFOEX));
    m_osInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
    if(!GetVersionEx((OSVERSIONINFO *)&m_osInfo)) {
        // TBD: What does one do?
    }

}

CGPUDataSegment::~CGPUDataSegment(void)
{
    delete m_ppb;
}

void CGPUDataSegment::Capture(void)
{
}

void CGPUDataSegment::SaveCatpuredData(CString &tempFilename)
{
    char *szTempFilename =  _tempnam(NULL,"lwsi"); // Get a temp filename with 'lwsi' prefix created in TMP directory

    if (szTempFilename==NULL) {
        throw CLwsiException("Error: failed to create temp file");
    }
    tempFilename = szTempFilename;

    ASSERT(m_ppb);

    std::fstream pb_output(tempFilename, std::ios::out | std::ios::trunc | std::ios::binary);
    if (pb_output.fail())
    {
        throw CLwsiException("Error: failed to serialize CGPUDataSegment PB");
    }
    else
    {
        if (!m_ppb->SerializeToOstream(&pb_output))
        {
            CString msg;
            msg.Format("Error: failed to serialize CGPUDataSegment PB to file %s",m_filename);
            throw CLwsiException(msg);
        }
        pb_output.close();
    }
    free(szTempFilename);
}


void CGPUDataSegment::LoadCatpuredData(void)
{
#if 0
bool pb_parse_from_file(swakpb::Swak *pbMessage, const char *fname)

    ASSERT(pbMessage);
    std::fstream pb_input(fname, std::ios::in | std::ios::binary);
    if (pb_input.fail())
    {
        TRACE("pb_parse_from_file: Failed to open file %s for protobuf message parsing\n",fname);        
        return false;
    }
    else
    {
        if (!pbMessage->ParseFromIstream(&pb_input))
        {
            TRACE("pb_parse_from_file: Failed to parse protobuf message to file %s\n",fname);
            return false;
        }
        pb_input.close();
    }
    TRACE("Swak: Parsed protocol buffer from file %s\n",fname);
    return true;
#endif
}
void CGPUDataSegment::RawViewData(void)
{
    LwsiErrorMsg(m_ppb->DebugString().c_str());
} 
void CGPUDataSegment::FormattedViewData(void)
{
}

void CGPUDataSegment::RunCmds(const char *szCmds)
{
    printf("Running commands: %s\n",szCmds);

    m_ppb->set_lwsi_internalversion(LWSI_PRODUCT_VERSION_STR);

    // Note we can't do hard-ordering into groups here... because clients have in the past wanted to run
    // one command before others (like optimus) because some of the queries alter the system state (by waking up our GPU)
    //
    // So query order IS important in some cases and needs to be controlled.

    CapWin32GetNativeSystemInfo();
    CapWin32GetVersionEx();
    CapWin32EnumDisplayDevices();
    CapWin32QuickFixes();

    CapLwapiData();
}

void CGPUDataSegment::CapLwapiGpuData()
{
    LwU32 i;
    LwAPI_ShortString  szShortGpuName;
    LwAPI_ShortString  szLongGpuName;
    LwAPI_Status lwapiStatus;

    for (i=0; i<m_physicalGpuCount; i++)
    {
        lwsi_pb::LwapiGpu *pbGpu = m_ppb->add_gpu();

        lwapiStatus = LwAPI_GPU_GetFullName(m_hPhysicalGpu_a[i],szLongGpuName);
        if (lwapiStatus != LWAPI_OK)
        {
            printf("Failed LwAPI_GPU_GetFullName with: %d",lwapiStatus);
            strcpy(szLongGpuName,"???");
        }
        else
        {
            pbGpu->set_fullname(szLongGpuName);
        }

        lwapiStatus = LwAPI_GPU_GetShortName(m_hPhysicalGpu_a[i],szShortGpuName);
        if (lwapiStatus != LWAPI_OK)
        {
            printf("Failed LwAPI_GPU_GetShortName with: %d",lwapiStatus);
            strcpy(szShortGpuName,"???");
        }
        else
        {
            pbGpu->set_shortname(szShortGpuName);
        }
    } // End loop over all physical GPUs
}

void CGPUDataSegment::CapLwapiData()
{
    LwAPI_Status lwapiStatus;
        
    // Initialize LwAPI if possible and enumerate the physical GPUs
    lwapiStatus = LwAPI_Initialize();

    if (lwapiStatus == LWAPI_OK) 
    {
        lwapiStatus = LwAPI_EnumPhysicalGPUs(m_hPhysicalGpu_a, &m_physicalGpuCount);

        if (lwapiStatus != LWAPI_OK)
        {
            printf("Error: LwAPI_EnumPhysicalGPUs(): %d",lwapiStatus);
            m_physicalGpuCount = 0;
        }
    }
    else
    {
        printf("Error: LwAPI_Initialize() failed.  No LwAPI based info available.");
    }

    CapLwapiGpuData();
    CapLwApiDisplayEdids();
}

void CGPUDataSegment::CapWin32GetNativeSystemInfo()
{
    SYSTEM_INFO info;

    GetSystemInfo(&info);

    lwsi_pb::Win32GetNativeSystemInfo *pData = m_ppb->mutable_win32_getnativesysteminfo();

    pData->set_wprocessorarchitecture(info.wProcessorArchitecture);
    pData->set_dwpagesize(info.dwPageSize);
    pData->set_lpminimumapplicationaddress((unsigned __int64)info.lpMinimumApplicationAddress);
    pData->set_lpmaximumapplicationaddress((unsigned __int64)info.lpMaximumApplicationAddress);
    pData->set_dwactiveprocessormask(info.dwActiveProcessorMask);
    pData->set_dwnumberofprocessors(info.dwNumberOfProcessors);
    pData->set_dwprocessortype(info.dwProcessorType);
    pData->set_dwallocationgranularity(info.dwAllocationGranularity);
    pData->set_wprocessorlevel(info.wProcessorLevel);
    pData->set_wprocessorrevision(info.wProcessorRevision);
}

void CGPUDataSegment::CapWin32GetVersionEx()
{
    lwsi_pb::Win32GetVersionEx *pData = m_ppb->mutable_win32_getversionex();

    // We've already queried OSVERSIONINFOEX when this class started (so we knew which OS we were)
    pData->set_dwmajorversion(m_osInfo.dwMajorVersion);
    pData->set_dwminorversion(m_osInfo.dwMinorVersion);
    pData->set_dwbuildnumber(m_osInfo.dwBuildNumber);
    pData->set_dwplatformid(m_osInfo.dwPlatformId);
    pData->set_szcsdversion(m_osInfo.szCSDVersion);
    pData->set_wservicepackmajor(m_osInfo.wServicePackMajor);
    pData->set_wservicepackminor(m_osInfo.wServicePackMinor);
    pData->set_wsuitemask(m_osInfo.wSuiteMask);
    pData->set_wproducttype(m_osInfo.wProductType);
}

void CGPUDataSegment::CapWin32EnumDisplayDevices()
{
    LwU32          i;
    DISPLAY_DEVICE dd = {0};

    dd.cb = sizeof(dd);

    for(i=0; EnumDisplayDevices(NULL, i, &dd, 0); i++)
    {
        if (dd.StateFlags & DISPLAY_DEVICE_MIRRORING_DRIVER) continue; // skip fake devices out-right

        lwsi_pb::Win32EnumDisplayDevices *pDisplayDevice = m_ppb->add_win32_displaydevices();

        pDisplayDevice->set_devicename(dd.DeviceName);
        pDisplayDevice->set_devicestring(dd.DeviceString);
        pDisplayDevice->set_stateflags(dd.StateFlags);
        pDisplayDevice->set_deviceid(dd.DeviceID);
        pDisplayDevice->set_devicekey(dd.DeviceKey);

        if (dd.StateFlags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP) 
        {
            DEVMODE       devMode = {0};
            devMode.dmSize = sizeof(devMode);

            lwsi_pb::Win32DevMode *pData = pDisplayDevice->mutable_devmode();

            if (EnumDisplaySettings(dd.DeviceName,  ENUM_LWRRENT_SETTINGS, &devMode)==FALSE)
            {
                printf("Failed to get display mode using EnumDisplaySettings() on display %s",dd.DeviceName);
            }
            pData->set_dmbitsperpel(devMode.dmBitsPerPel);
            pData->set_dmpelswidth(devMode.dmPelsWidth);
            pData->set_dmpelsheight(devMode.dmPelsHeight);
            pData->set_dmdisplayfrequency(devMode.dmDisplayFrequency);
        }
        else
        {
            printf("  %s - Not Attached",dd.DeviceName);
        }
    } // Loop over all display devices
}

void CGPUDataSegment::CapWin32QuickFixes()
{
    CWMIInterface            wmiTool;

    // Check the Windows Quick Fix updates installed using WMI
    // On XP and 2k3, the updates are also around HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Updates\\Windows XP
    // Another way of getting this information is by running "wmic qfe list"

    if (!wmiTool.OpenConnection("\\\\.\\root\\cimv2"))
    {
        // TBD: record failure code?
        printf("  Failed to open WMI connection.");
        return;
    }

    wchar_t wcQueryString[256] = L"select * from Win32_QuickFixEngineering ";

    // Each WQL query can take 4secs to execute disregarding the number of records returned, 
    // so instead of using multiple WQL queries to filter out patches with "WHERE" clause, 
    // we get the full recordset and filter the records manually
    if (!wmiTool.ExelwteWQLQuery(wcQueryString)) 
    {
        // TBD: record failure code?
        printf("  Failed query for Win32_QuickFixEngineering ");
        return;
    }

    while (wmiTool.GetNextObject())
    {
        static char szHotFixDesc[1024];
        static char szHotFixId[1024];

        if (!wmiTool.GetAttributeValue(L"HotFixID", szHotFixId) ||
            !wmiTool.GetAttributeValue(L"Description", szHotFixDesc))
        {
            printf("  Failed to get attribute values for HotFixID and Description, aborting query");
            continue;
        }
        if (szHotFixId && strstr("File 1:",szHotFixId)!=NULL) {
            // Skip this old XP fixes that don't follow WMI standards
            continue;
        }
        lwsi_pb::Win32HotFix *pData = m_ppb->add_win32_hotfixes();

        pData->set_szhotfixid(szHotFixId);
        pData->set_szhotfixdesc(szHotFixDesc);
    }

    wmiTool.CloseConnection();
}


void CGPUDataSegment::CapLwApiDisplayEdids()
{
    LwU32 i;
    LwAPI_Status lwapiStatus;
    LwAPI_ShortString szErrDesc;

    for(i=0; i<m_physicalGpuCount; i++)
    {
        LwU32 connectedDisplays;

        lwapiStatus = LwAPI_GPU_GetConnectedOutputs(m_hPhysicalGpu_a[i],&connectedDisplays);
        if (lwapiStatus != LWAPI_OK)
        {
            printf("LwAPI_GPU_GetConnectedOutputs failed.");
            continue;
        }
        if (connectedDisplays == 0)
        {
            printf("  No displays connected to GPU 0x%08x",m_hPhysicalGpu_a[i]);
            continue;
        }
        for(LwU32 bit=1; bit!=0; bit<<=1)
        {
            if ((bit & connectedDisplays)==0) continue;

            lwsi_pb::LwApiEdid *pData = m_ppb->add_lwapi_edid();

            pData->set_displaybit(bit);

            LW_EDID edid = {0};
            edid.version = LW_EDID_VER;
            lwapiStatus = LwAPI_GPU_GetEDID(m_hPhysicalGpu_a[i],bit,&edid);
            if (lwapiStatus != LWAPI_OK)
            {
                // The EDID struct itself is backward compatible; try earlier versions to see if we can match
                // the version the driver is expecting
                if (lwapiStatus == LWAPI_INCOMPATIBLE_STRUCT_VERSION)
                {
                    edid.version = LW_EDID_VER2;
                    lwapiStatus = LwAPI_GPU_GetEDID(m_hPhysicalGpu_a[i],bit,&edid);
                    if (lwapiStatus == LWAPI_INCOMPATIBLE_STRUCT_VERSION)
                    {
                        edid.version = LW_EDID_VER1;
                        lwapiStatus = LwAPI_GPU_GetEDID(m_hPhysicalGpu_a[i],bit,&edid);
                    }
                }
                if (lwapiStatus != LWAPI_OK)
                {
                    LwAPI_GetErrorMessage(lwapiStatus,szErrDesc);
                    printf("LwAPI_GPU_GetEDID(gpu=%08x,display=%08x) failed with: %s",m_hPhysicalGpu_a[i],bit,szErrDesc);
                    continue;
                }
            }
            pData->set_sizeofedid(edid.sizeofEDID);
            pData->set_data(edid.EDID_Data,edid.sizeofEDID);

        } // Connected Display Loop
    } // Physical GPU Loop
}
#if 0
typedef struct
{
    LwU32   version;        //structure version
    LwU8    EDID_Data[LW_EDID_DATA_SIZE];
} LW_EDID_V1;

typedef struct
{
    LwU32   version;        //structure version
    LwU8    EDID_Data[LW_EDID_DATA_SIZE];
    LwU32   sizeofEDID;
} LW_EDID_V2;

typedef struct
{
    LwU32   version;        //structure version
    LwU8    EDID_Data[LW_EDID_DATA_SIZE];
    LwU32   sizeofEDID;
    LwU32   edidId;     // edidId is an ID which always returned in a monotonically increasing counter.
                       // Across a split-edid read we need to verify that all calls returned the same edidId.
                       // This counter is incremented if we get the updated EDID.
    LwU32   offset;    // which 256byte page of the EDID we want to read. Start at 0.
                       // If the read succeeds with edidSize > LW_EDID_DATA_SIZE
                       // call back again with offset+256 until we have read the entire buffer
} LW_EDID_V3;
#endif