#pragma once

/* Generic facility. Indicates that error oclwred on API DLL side. */
#define LWTELEMETRY_HRESULT_FACILITY_API      1
/* Indicates that error oclwred in underlying RPC mechanism. */
#define LWTELEMETRY_HRESULT_FACILITY_RPC      2
/* Indicates that error oclwred in LwTelemetry process. */
#define LWTELEMETRY_HRESULT_FACILITY_MAIN     3

/* Creates HRESULT with customer flag set */
#define LWTELEMETRY_MAKE_HRESULT(sev,fac,code) ((HRESULT) ( ((unsigned long)(sev)<<31) | ((unsigned long)(fac)<<16) | ((unsigned long)(code) | (1<<29) )) )

/* API DLL failure */
#define LWTELEMETRY_E_API_GENERIC LWTELEMETRY_MAKE_HRESULT(SEVERITY_ERROR, LWTELEMETRY_HRESULT_FACILITY_API, 0)

/* RPC failure */
#define LWTELEMETRY_E_RPC_GENERIC LWTELEMETRY_MAKE_HRESULT(SEVERITY_ERROR, LWTELEMETRY_HRESULT_FACILITY_RPC, 0)

HRESULT EnsureThatTelemetryPluginIsRunning(DWORD timeout);

HRESULT LwTelemetrySendEvent(const char* jsonString);