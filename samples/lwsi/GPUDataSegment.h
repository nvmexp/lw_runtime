#pragma once
#include "DataSegment.h"
#include "lwsi.pb.h" 
#include "WMIInterface.h"

class CGPUDataSegment : public CDataSegment
{
    lwsi_pb::GpuDataSegment *m_ppb;
    LwPhysicalGpuHandle      m_hPhysicalGpu_a[LWAPI_MAX_PHYSICAL_GPUS];
    LwU32                    m_physicalGpuCount;
    OSVERSIONINFOEX          m_osInfo;

public:
    CGPUDataSegment(const char *szFilename);
    ~CGPUDataSegment(void);

    void Capture(void);
    void SaveCatpuredData(CString &tempFilename);
    void LoadCatpuredData(void);
    void RawViewData(void);
    void FormattedViewData(void);
    void RunCmds(const char *szCmds);

    void CapWin32GetNativeSystemInfo();
    void CapWin32GetVersionEx();
    void CapWin32EnumDisplayDevices();
    void CapWin32QuickFixes();


    void CapLwapiData();
    void CapLwapiGpuData();
    void CapLwApiDisplayEdids();
};
