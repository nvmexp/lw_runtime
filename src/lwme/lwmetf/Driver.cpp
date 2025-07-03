/*++

Copyright (C) LWPU Corporation, 2020

Module Name:

    Driver.cpp

Abstract:

    This file implements lwmetf.sys --
    kernel mode filter driver for using LWMe queue reservation code in storlwme.sys for LWAPI direct storage implementation.

--*/
#define ENABLE_WPP_RECORDER 1
#if defined(LW_LDDM) && (LW_LDDM >= 22499)
// Disable warning C4996 due to ExAllocatePoolWithTag deprecation
#pragma warning(push)
#pragma warning( disable : 4996 )
#endif
#include <Ntifs.h>
#if defined(LW_LDDM) && (LW_LDDM >= 22499)
#pragma warning(pop)
#endif
#include <wdf.h>

#include <stdarg.h>
#include <ntddk.h>
#include "driver.h"

#include "SQueue.h"
#include "CQueue.h"

#include <storswtr.h>
#include <Ntddvol.h>
#include <ndis.h>

#include <ntstrsafe.h>

// GLOBAL VARIABLE
BOOL g_bLwidiaGPUPresent = false;

DRIVER_GLOBAL g_GlobalData;

#ifdef USE_ETW_LOGGING
#include "etw\lwmetfETW.h"
#endif

//******************************************************************************

// DRIVER ENTRY
#pragma alloc_text(INIT_CODE, DriverEntry)
NTSTATUS
DriverEntry(
    _In_ PDRIVER_OBJECT  DriverObject,
    _In_ PUNICODE_STRING RegistryPath
)
{
    CHECK_IRQL(PASSIVE_LEVEL);

    WDF_DRIVER_CONFIG config;
    NTSTATUS status = STATUS_SUCCESS;
    WDF_OBJECT_ATTRIBUTES attributes;

    ExInitializeDriverRuntime(DrvRtPoolNxOptIn);

    WPP_INIT_TRACING(DriverObject, RegistryPath);

    //#ifdef DBG
    //    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER, "Enter (DEBUG)\n");
    //#endif

    WDF_OBJECT_ATTRIBUTES_INIT(&attributes);
    attributes.EvtCleanupCallback = lwmetfEvtDriverContextCleanup;

    WDF_DRIVER_CONFIG_INIT(&config, lwmetfEvtDeviceAdd);

    // Initialize the global driver data
    RtlZeroMemory(&g_GlobalData, sizeof(g_GlobalData));
    // Save the driver object
    g_GlobalData.pDriverObject = DriverObject;

    // Catch if due to any issue in build procecss we might get uninitialized WdfFunctions.
    ASSERT(WdfFunctions != NULL);

    __try
    {
        status = WdfDriverCreate(DriverObject,
            RegistryPath,
            &attributes,
            &config,
            WDF_NO_HANDLE
        );

        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "WdfDriverCreate passed. \n");

        KeInitializeSpinLock(&g_PinnedAllocListLock);
        InitializeListHead(&g_PinnedAllocLockedList);

        PsSetCreateProcessNotifyRoutine(ProcessCallback, FALSE);
    }
    __except (MyExceptionFilter(GetExceptionInformation()))
    {
        unsigned long exceptionCode = exception_code();
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, " %s()::%d - WdfDriverCreate exception EXELWTE_HANDLER 0x%x.\n", __FUNCTION__, __LINE__, exceptionCode);
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "RegistryPath = '%wZ'\n", RegistryPath);
        logSystemEvent(status, " % s():: % d - WdfDriverCreate exception EXELWTE_HANDLER 0x % x.\n", __FUNCTION__, __LINE__, exceptionCode);
    }

    if (!NT_SUCCESS(status)) {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "WdfDriverCreate failed %!STATUS!\n", status);
        logSystemEvent(status, "WdfDriverCreate failed %!STATUS!\n", status);
        WPP_CLEANUP(DriverObject);

        return status;
    }

#ifdef USE_ETW_LOGGING
    EventRegisterlwmetfETW();
#endif

    return status;
}
#pragma code_seg()  // Section "INIT" ends here

PUSH_SEGMENTS
// CUSTOM LOGGING ////////////////////////////////////////////////

// To view debug messages in windbg
// ed nt!Kd_IHVDRIVER_Mask 0xff 
CODE_SEGMENT(NONPAGE_CODE)
void TraceEvents(int DebugPrintLevel, int driverOrQueue, PCCHAR DebugMessage, ...)
{
    CHECK_IRQL(DISPATCH_LEVEL);
#ifdef LW_LOG_ENABLE
    va_list ap;

    const char* driverOrQueueStr[] = { "[lwmetf-Driver] ", "[lwmetf-Queue] ", "" };

    va_start(ap, DebugMessage);
#if DBG
    vDbgPrintExWithPrefix(driverOrQueueStr[driverOrQueue], DPFLTR_IHVDRIVER_ID /*not sure where to get component id?*/, DebugPrintLevel, DebugMessage, ap);
#else
    KdPrint((DebugMessage, ap));
#endif
    va_end(ap);
#else

    UNREFERENCED_PARAMETER(DebugPrintLevel);
    UNREFERENCED_PARAMETER(driverOrQueue);
    UNREFERENCED_PARAMETER(DebugMessage);
#endif // DBG
}
// CUSTOM LOGGING ////////////////////////////////////////////////
CODE_SEGMENT(NONPAGE_CODE)
LONG
MyExceptionFilter(_In_ PEXCEPTION_POINTERS ExceptionPointer )
{
    CHECK_IRQL(DISPATCH_LEVEL);

    TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "ExceptionCode = '%x'\n", ExceptionPointer->ExceptionRecord->ExceptionCode);
    TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "ExceptionFlags = '%x'\n", ExceptionPointer->ExceptionRecord->ExceptionFlags);
    TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "NumberParameters = '%x'\n", ExceptionPointer->ExceptionRecord->NumberParameters);
    for (ULONG i = 0; i < EXCEPTION_MAXIMUM_PARAMETERS && i < ExceptionPointer->ExceptionRecord->NumberParameters; i++)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "ExceptionInformation[%d] = '%lu'\n", i, ExceptionPointer->ExceptionRecord->ExceptionInformation);
    }
    return EXCEPTION_EXELWTE_HANDLER;
}

CODE_SEGMENT(PAGE_CODE)
// Every object in the returned list must be explicitly dereferenced with ObDereferenceObject()
PDEVICE_OBJECT*
LwEnumeratePciDeviceObjectList(PULONG ActualNumberDeviceObjects)
{
    CHECK_IRQL(PASSIVE_LEVEL);

    ULONG NumObjects = 0;
    NTSTATUS Status;
    WCHAR buf[21]; // enough to hold "\\Device\\NTPNP_PCI0000"
    UNICODE_STRING deviceName = { sizeof(buf), sizeof(buf), buf };
    PFILE_OBJECT fileObj = 0;
    PDEVICE_OBJECT deviceObj = 0;
    PDEVICE_OBJECT lowestDO = 0;
    PDRIVER_OBJECT pciDriver = NULL;
    PDEVICE_OBJECT* DeviceObjectList = NULL;

    for (ULONG devIndex = 0; devIndex < 0xFFFF; ++devIndex)
    {
        Status = RtlUnicodeStringPrintf(&deviceName, L"%S%04u", "\\Device\\NTPNP_PCI", devIndex);
        if (NT_SUCCESS(Status))
        {
            Status = IoGetDeviceObjectPointer(&deviceName, GENERIC_READ, &fileObj, &deviceObj);
            if (NT_SUCCESS(Status))
            {
                lowestDO = IoGetDeviceAttachmentBaseRef(deviceObj);

                ObDereferenceObject(fileObj);
                ObDereferenceObject(lowestDO);

                pciDriver = lowestDO->DriverObject; // a pointer to PCI driver object
                break;
            }
        }
    }

    if (pciDriver)
    {
        Status = IoEnumerateDeviceObjectList(pciDriver, NULL, 0, &NumObjects);

        if ((!NT_SUCCESS(Status) && Status != STATUS_BUFFER_TOO_SMALL) ||
            (NumObjects == 0))
        {
            return NULL;
        }

        DeviceObjectList = reinterpret_cast<PDEVICE_OBJECT*>(lwAllocatePoolWithTag(NonPagedPool, sizeof(PDEVICE_OBJECT) * NumObjects, LW_LWME_TAG));
        if (DeviceObjectList == NULL)
        {
            return NULL;
        }

        Status = IoEnumerateDeviceObjectList(pciDriver, DeviceObjectList, sizeof(PDEVICE_OBJECT) * NumObjects, &NumObjects);
        if (!NT_SUCCESS(Status))
        {
            ExFreePoolWithTag(DeviceObjectList, LW_LWME_TAG);
            return NULL;
        }

        *ActualNumberDeviceObjects = NumObjects;
    }

    return DeviceObjectList;
}

CODE_SEGMENT(PAGE_CODE)
PBUS_INTERFACE_STANDARD
getPciInterface(PDEVICE_OBJECT PhysicalDeviceObject)
{
    CHECK_IRQL(PASSIVE_LEVEL);
    PDEVICE_OBJECT targetObject;
    PBUS_INTERFACE_STANDARD pciInterface = NULL;
    KEVENT pciEvent;
    NTSTATUS status = STATUS_UNSUCCESSFUL;
    PIRP irp = NULL;
    PIO_STACK_LOCATION irpSp = NULL;
    PIO_STATUS_BLOCK iosb = NULL;

    if (PhysicalDeviceObject == NULL)
        return NULL;

    iosb = reinterpret_cast<PIO_STATUS_BLOCK>(lwAllocatePoolWithTag(PagedPool, sizeof(IO_STATUS_BLOCK), LW_LWME_TAG));
    if (!iosb)
        return NULL;

    // Should be allocated from the PagedPool according to MSDN
    pciInterface = reinterpret_cast<PBUS_INTERFACE_STANDARD>(lwAllocatePoolWithTag(PagedPool, sizeof(BUS_INTERFACE_STANDARD), LW_LWME_TAG));
    if (!pciInterface)
    {
        ExFreePoolWithTag(iosb, LW_LWME_TAG);
        return NULL;
    }

    RtlZeroMemory(pciInterface, sizeof(BUS_INTERFACE_STANDARD));
    KeInitializeEvent(&pciEvent, SynchronizationEvent, FALSE);

    targetObject = IoGetAttachedDeviceReference(PhysicalDeviceObject); // get the highest object in the stack

    irp = IoBuildSynchronousFsdRequest(IRP_MJ_PNP, targetObject, NULL, 0, 0, &pciEvent, iosb);
    if (!irp)
    {
        ExFreePoolWithTag(iosb, LW_LWME_TAG);
        ExFreePoolWithTag(pciInterface, LW_LWME_TAG);
        ObDereferenceObject(targetObject);
        return NULL;
    }

    irp->IoStatus.Status = STATUS_NOT_SUPPORTED;    // Verifier's requirement
    irpSp = IoGetNextIrpStackLocation(irp);
    irpSp->MajorFunction = IRP_MJ_PNP;
    irpSp->MinorFunction = IRP_MN_QUERY_INTERFACE;
    irpSp->Parameters.QueryInterface.InterfaceType = (LPGUID)&GUID_BUS_INTERFACE_STANDARD;
    irpSp->Parameters.QueryInterface.Version = 1; // With any other value the PCI returns nothing
    irpSp->Parameters.QueryInterface.Interface = reinterpret_cast<PINTERFACE>(pciInterface);
    irpSp->Parameters.QueryInterface.Size = sizeof(BUS_INTERFACE_STANDARD);

    // In spite of the fact the request was synchronous, pci may choose to treat it as an asynchronous one. (Verifier enforced)
    status = IoCallDriver(targetObject, irp);

    if (status == STATUS_PENDING)
    {
        KeWaitForSingleObject(&pciEvent, Exelwtive, KernelMode, FALSE, NULL);
        status = iosb->Status;
    }

    ExFreePoolWithTag(iosb, LW_LWME_TAG);
    ObDereferenceObject(targetObject);
    return pciInterface;
}

CODE_SEGMENT(NONPAGE_CODE)
void
freePciInterface(PBUS_INTERFACE_STANDARD PciInterface)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    if (PciInterface)
    {
        PciInterface->InterfaceDereference(PciInterface->Context);
        ExFreePoolWithTag(PciInterface, LW_LWME_TAG);
    }
}

#if DBG
CODE_SEGMENT(PAGE_CODE)
void 
DBGPrintDevice(
    _In_ PDEVICE_OBJECT DeviceObject
)
{
    CHECK_IRQL(PASSIVE_LEVEL);
    // Print it for debugging!
    ULONG busNumber, deviceAddress, resultLength;
    if (NT_SUCCESS(IoGetDeviceProperty(DeviceObject, DevicePropertyBusNumber, sizeof(ULONG), &busNumber, &resultLength)))
    {
        TraceEvents(DPFLTR_VERBOSE_LEVEL,
            TRACE_QUEUE,
            "BusNumber %d.\n", busNumber);
    }

    // Try to get the address for this device (Device & Function)
    if (NT_SUCCESS(IoGetDeviceProperty(DeviceObject, DevicePropertyAddress, sizeof(PCI_SLOT_NUMBER), &deviceAddress, &resultLength)))
    {
        ULONG deviceNumber = (deviceAddress >> 16) & 0xffff;
        TraceEvents(DPFLTR_VERBOSE_LEVEL,
            TRACE_QUEUE,
            "device nunber = %d.\n", deviceNumber);
    }

    {
        ULONG nameLength = 1024;
        PWCHAR deviceName = (PWCHAR)lwAllocatePoolWithTag(NonPagedPoolNx, nameLength, LW_LWME_TAG);
        ASSERT(deviceName);
        if (NT_SUCCESS(IoGetDeviceProperty(DeviceObject, DevicePropertyPhysicalDeviceObjectName, nameLength, deviceName, &nameLength)))
        {
            TraceEvents(DPFLTR_VERBOSE_LEVEL,
                TRACE_QUEUE,
                "PhysicalDeviceObjectName = (%ws).\n", deviceName);

            TraceEvents(DPFLTR_VERBOSE_LEVEL,
                TRACE_QUEUE,
                "DriverObject->DriverName = (%s).\n", DeviceObject->DriverObject->DriverName.Buffer);
        }

        ExFreePoolWithTag(deviceName, LW_LWME_TAG);
    }
    {
        ULONG nameLength = 1024;
        PWCHAR str = (PWCHAR)lwAllocatePoolWithTag(NonPagedPoolNx, nameLength, LW_LWME_TAG);
        ASSERT(str);
        if (NT_SUCCESS(IoGetDeviceProperty(DeviceObject, DevicePropertyDriverKeyName, nameLength, str, &nameLength)))
        {
            TraceEvents(DPFLTR_VERBOSE_LEVEL,
                TRACE_QUEUE,
                "DevicePropertyDriverKeyName = (%ws).\n", str);
        }

        ExFreePoolWithTag(str, LW_LWME_TAG);
    }

    {
        ULONG nameLength = 1024;
        PWCHAR str = (PWCHAR)lwAllocatePoolWithTag(NonPagedPoolNx, nameLength, LW_LWME_TAG);
        ASSERT(str);
        if (NT_SUCCESS(IoGetDeviceProperty(DeviceObject, DevicePropertyLocationInformation, nameLength, str, &nameLength)))
        {
            TraceEvents(DPFLTR_VERBOSE_LEVEL,
                TRACE_QUEUE,
                "DevicePropertyLocationInformation = (%ws).\n", str);
        }

        ExFreePoolWithTag(str, LW_LWME_TAG);
    }
    {
        ULONG nameLength = 1024;
        PWCHAR str = (PWCHAR)lwAllocatePoolWithTag(NonPagedPoolNx, nameLength, LW_LWME_TAG);
        ASSERT(str);
        if (NT_SUCCESS(IoGetDeviceProperty(DeviceObject, DevicePropertyFriendlyName, nameLength, str, &nameLength)))
        {
            TraceEvents(DPFLTR_VERBOSE_LEVEL,
                TRACE_QUEUE,
                "DevicePropertyFriendlyName = (%ws).\n", str);
        }

        ExFreePoolWithTag(str, LW_LWME_TAG);
    }
}
#endif // DBG

CODE_SEGMENT(PAGE_CODE)
USHORT
GetLWMeQueueDepthFromBar0Register(
    _In_ WDFDEVICE Device
)
{
    CHECK_IRQL(PASSIVE_LEVEL);

    PDEVICE_CONTEXT deviceContext = GetDeviceContext(Device);

    USHORT QueueDepth = LW_DEFAULT_QUEUE_DEPTH;
    bool QueueDepthFound = false;

    PDEVICE_OBJECT* DeviceObjectList = NULL;
    ULONG NumObjects = 0, i = 0;

    DeviceObjectList = LwEnumeratePciDeviceObjectList(&NumObjects);
    if (!DeviceObjectList)
    {
        return QueueDepth;
    }

    for (; i < NumObjects; ++i)
    {
        // PCI holds many "fake" devices (filters, etc) which cannot be enumerated
        if (FlagOn(DeviceObjectList[i]->Flags, DO_BUS_ENUMERATED_DEVICE))
        {
            PCI_COMMON_CONFIG pciConfig;
            PBUS_INTERFACE_STANDARD pciInterface = getPciInterface(DeviceObjectList[i]);

            // Try to get Vendor, Device and BaseClass
            if (pciInterface && pciInterface->GetBusData(pciInterface->Context, PCI_WHICHSPACE_CONFIG, &pciConfig, 0, sizeof(pciConfig)) == sizeof(PCI_COMMON_CONFIG))
            {
                TraceEvents(DPFLTR_VERBOSE_LEVEL,
                    TRACE_QUEUE,
                    "PCI Found device number = %d. (VenderID 0x%x, DeviceID 0x%x, RevisionID 0x%x)\n", pciConfig.VendorID, pciConfig.DeviceID, pciConfig.RevisionID);

                if (pciConfig.VendorID == LW_VENDOR_ID)
                {
                    //ASSERT(pciConfig.BaseClass == 0x03); // 0x03 - Display Controller
                    g_bLwidiaGPUPresent = TRUE;
                }

                if (pciConfig.BaseClass == 0x01 &&  // 0x01 - Mass Storage Controller
                    pciConfig.SubClass == 0x08 &&   // 0x08 - Non - Volatile Memory Controller
                    pciConfig.ProgIf == 0x02)       // 0x02 - LWM Express
                {
#if DBG
                    DBGPrintDevice(DeviceObjectList[i]);
#endif
                    // We need to use same LWMe Bar0 of filter driver object ~stack. Since we don't have exact mapping between for them, WAR is using vendor id to match it.
                    // Also, in case we have multiple LWMe on system with same vendor ID, we do min(QueueDepth) so we are still correct and hopefully using better Qdepth than default.

                    // Only use if PCI vendor ID is matched with identity VID pulled from driver stack.
                    if (deviceContext->m_pIdentifyControllerData->VID != pciConfig.VendorID) // So we are matching Bar0 of kind of correct device.)
                    {
                        TraceEvents(DPFLTR_VERBOSE_LEVEL,
                            TRACE_QUEUE,
                            "Skipping LWMe as vendor id didnt match. (VenderID from identify 0x%x, VenderID from pci config 0x%x)\n"
                            , deviceContext->m_pIdentifyControllerData->VID, pciConfig.VendorID);
                    }
                    else
                    {
                        // MAP Bar0 addresses to read LWMe registers
                        PHYSICAL_ADDRESS PhysicalAddressBar0;
                        PhysicalAddressBar0.QuadPart = 0;

                        if (pciConfig.HeaderType == 0x00)
                        {
                            PhysicalAddressBar0.LowPart = pciConfig.u.type0.BaseAddresses[0];
                            PhysicalAddressBar0.HighPart = pciConfig.u.type0.BaseAddresses[1];
                        }
                        else if (pciConfig.HeaderType == 0x01)
                        {
                            PhysicalAddressBar0.LowPart = pciConfig.u.type1.BaseAddresses[0];
                            PhysicalAddressBar0.HighPart = pciConfig.u.type0.BaseAddresses[1];
                        }
                        else
                        {
                            // Check if we get other header type values?
                            ASSERT(pciConfig.HeaderType == 0x02);

                            // TODO: HeaderType=0x02 not implemented, need to check if this is possible for LWMe?
                            ASSERT(0);
                        }

                        PLWME_CONTROLLER_REGISTERS bar0_LWMeControllerRegister = NULL;
                        if (PhysicalAddressBar0.QuadPart != 0)
                        {
                            // Memory Space BAR Layout "2 - 1" bits Type where 0 indicates address is 32bit physical address otherwise 64bit address.
                            if (PhysicalAddressBar0.LowPart & 1)
                            {
                                // truncate last 2 bits of 32bit address
                                PhysicalAddressBar0.LowPart &= 0xFFFFFFFC;
                                // Remove upper32 since address is 32bit always (No Type present in this layout)
                                PhysicalAddressBar0.HighPart = 0;
                            }
                            else
                            {
                                // truncate last 4 bits of 32bit address
                                PhysicalAddressBar0.LowPart &= 0xFFFFFFF0;

                                // BARType (Bits 2-1): it has a value of 0x00 then the base register is 32-bits wide and can be mapped anywhere in the 32-bit Memory Space. 
                                // A value of 0x02 means the base register is 64-bits wide and can be mapped anywhere in the 64-bit Memory Space (A 64-bit base address register 
                                // consumes 2 of the base address registers available). A value of 0x01 is reserved as of revision 3.0 of the PCI Local Bus Specification. 
                                USHORT BARType = ((PhysicalAddressBar0.LowPart & 0x6) >> 1);

                                if (BARType == 0x00)
                                {
                                    // Bar0 is 32 bit address and need to cleanup upper 32bits
                                    PhysicalAddressBar0.HighPart = 0;
                                }
                                else
                                {
                                    // So, for 0x02 we alreay have upper32 bits set from BAR1.
                                    // Make sure we are not hitting 0x01 which is reserved for something else
                                    ASSERT(BARType == 0x02);
                                }
                            }

                            bar0_LWMeControllerRegister = (PLWME_CONTROLLER_REGISTERS)MmMapIoSpace(PhysicalAddressBar0, sizeof(PLWME_CONTROLLER_REGISTERS), MmNonCached);
                        }

                        if (bar0_LWMeControllerRegister != NULL)
                        {
                            // Found first matching LWMe disk, overide QueueDepth default value with LWMe Bar0 caps MQES.
                            // In case found >1 matching LWMe disk maintain QueueDepth as minimum of all LWMe caps MQES found.
                            if (QueueDepthFound)
                            {
                                TraceEvents(DPFLTR_INFO_LEVEL,
                                    TRACE_QUEUE,
                                    "QueueDepth = min(MQES %d, old QueueDepth %d) = %d .\n", bar0_LWMeControllerRegister->CAP.MQES, QueueDepth, min(bar0_LWMeControllerRegister->CAP.MQES, QueueDepth));

                                QueueDepth = min((USHORT)bar0_LWMeControllerRegister->CAP.MQES, QueueDepth);
                            }
                            else
                            {
                                TraceEvents(DPFLTR_INFO_LEVEL,
                                    TRACE_QUEUE,
                                    "QueueDepth = (MQES) %d .\n", bar0_LWMeControllerRegister->CAP.MQES);

                                QueueDepth = (USHORT)bar0_LWMeControllerRegister->CAP.MQES;
                            }

                            QueueDepthFound = true;
                        }

                        if (bar0_LWMeControllerRegister != NULL)
                        {
                            MmUnmapIoSpace(bar0_LWMeControllerRegister, sizeof(PLWME_CONTROLLER_REGISTERS));
                        }
                    }
                }
                freePciInterface(pciInterface);
            }
        }
        ObDereferenceObject(DeviceObjectList[i]);
    }
    ExFreePoolWithTag(DeviceObjectList, LW_LWME_TAG);

    return QueueDepth;
}

CODE_SEGMENT(PAGE_CODE)
NTSTATUS
GetIdentifyController(
    _In_ WDFDEVICE Device
)
{
    CHECK_IRQL(APC_LEVEL);
    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "GetIdentifyController is called.\n");

    NTSTATUS            status = STATUS_SUCCESS;
    PDEVICE_OBJECT      pDeviceObject = WdfDeviceWdmGetDeviceObject(Device);
    PDEVICE_CONTEXT     deviceContext = GetDeviceContext(Device);

    KEVENT              event = {};
    KeInitializeEvent(&event, NotificationEvent, FALSE);

    IO_STATUS_BLOCK     ioStatus = {};
    PIRP                irp;

    ULONG               bufferLength = sizeof(STORAGE_PROPERTY_QUERY)
        + sizeof(STORAGE_PROTOCOL_SPECIFIC_DATA)
        + sizeof(LWME_IDENTIFY_CONTROLLER_DATA);

    PUCHAR buffer = (PUCHAR)MmAllocateNonCachedMemory(bufferLength);
    if (!buffer)
    {
        status = STATUS_INSUFFICIENT_RESOURCES;
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Failed to allocate buffer %d size!\n", bufferLength);
        logSystemEvent(status, "Failed to allocate buffer %d size!\n", bufferLength);
        goto Exit;
    }

    PDEVICE_OBJECT      plowestDeviceObject = IoGetDeviceAttachmentBaseRef(pDeviceObject);
    UNICODE_STRING      targetDriverName;
    RtlInitUnicodeString(&targetDriverName, L"\\Driver\\storlwme");
    if (!RtlEqualUnicodeString((PLWNICODE_STRING)&targetDriverName, (PLWNICODE_STRING)&plowestDeviceObject->DriverObject->DriverName, true))
    {
        status = STATUS_ILWALID_DEVICE_STATE;
        TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_QUEUE, "Driver target is not matched.\n");
        logSystemEvent(status, "Driver target is not matched.\n");
        goto Exit;
    }

    RtlZeroMemory(buffer, bufferLength);

    PSTORAGE_PROPERTY_QUERY query = (PSTORAGE_PROPERTY_QUERY)buffer;
    query->PropertyId = StorageAdapterProtocolSpecificProperty;
    query->QueryType = PropertyStandardQuery;

    PSTORAGE_PROTOCOL_SPECIFIC_DATA protocolData = (PSTORAGE_PROTOCOL_SPECIFIC_DATA)query->AdditionalParameters;
    protocolData->ProtocolType = ProtocolTypeLwme;
    protocolData->DataType = LWMeDataTypeIdentify;
    protocolData->ProtocolDataRequestValue = LWME_IDENTIFY_CNS_CONTROLLER;
    protocolData->ProtocolDataRequestSubValue = 0;
    protocolData->ProtocolDataOffset = sizeof(STORAGE_PROTOCOL_SPECIFIC_DATA);
    protocolData->ProtocolDataLength = sizeof(LWME_IDENTIFY_CONTROLLER_DATA);

    irp = IoBuildDeviceIoControlRequest(IOCTL_STORAGE_QUERY_PROPERTY,
        plowestDeviceObject,
        buffer,
        bufferLength,
        buffer,
        bufferLength,
        false,
        &event,
        &ioStatus);

    if (irp == NULL)
    {
        status = STATUS_INSUFFICIENT_RESOURCES;
        goto Exit;
    }
    else
    {
        // Send the request
        status = IoCallDriver(plowestDeviceObject, irp);

        if (status == STATUS_PENDING)
        {
            if ((status = KeWaitForSingleObject(&event, Exelwtive, KernelMode, FALSE, NULL)) == STATUS_SUCCESS)
            {
                status = ioStatus.Status;
            }
        }
    }

    if (!NT_SUCCESS(status))
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Failed to query lwme identity!\n");
        logSystemEvent(status, "Failed to query lwme identity!\n");
        goto Exit;
    }

    PSTORAGE_PROTOCOL_DATA_DESCRIPTOR protocolDataDescr = (PSTORAGE_PROTOCOL_DATA_DESCRIPTOR)buffer;

    // Validate the returned data.
    if ((protocolDataDescr->Version != sizeof(STORAGE_PROTOCOL_DATA_DESCRIPTOR)) ||
        (protocolDataDescr->Size != sizeof(STORAGE_PROTOCOL_DATA_DESCRIPTOR)))
    {
        status = STATUS_INTERNAL_ERROR;
        TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_QUEUE, "LWMeDevice: Data Descriptor Header is not valid, stop.\n");
        logSystemEvent(status, "LWMeDevice: Data Descriptor Header is not valid, stop.\n");
        goto Exit;
    }

    protocolData = &protocolDataDescr->ProtocolSpecificData;

    if ((protocolData->ProtocolDataOffset > sizeof(STORAGE_PROTOCOL_SPECIFIC_DATA)) ||
        (protocolData->ProtocolDataLength < sizeof(LWME_IDENTIFY_CONTROLLER_DATA)))
    {
        status = STATUS_INTERNAL_ERROR;
        TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_QUEUE, "LWMeDevice: ProtocolData Offset/Length is not valid, stop.\n");

        goto Exit;
    }

    PLWME_IDENTIFY_CONTROLLER_DATA pIdentityControllerData = (PLWME_IDENTIFY_CONTROLLER_DATA)((PCHAR)protocolData + protocolData->ProtocolDataOffset);

    if (!deviceContext->m_pIdentifyControllerData)
    {
        deviceContext->m_pIdentifyControllerData = (PLWME_IDENTIFY_CONTROLLER_DATA)MmAllocateNonCachedMemory(sizeof(LWME_IDENTIFY_CONTROLLER_DATA));
    }
    *deviceContext->m_pIdentifyControllerData = *pIdentityControllerData;

Exit:

    if (buffer)
    {
        MmFreeNonCachedMemory(buffer, bufferLength);
    }
    return status;
}

CODE_SEGMENT(NONPAGE_CODE)
VOID
AddLockedMDLToList(
    PMDL pMDL
)
{
    CHECK_IRQL(DISPATCH_LEVEL);
    PLOCKED_MDL nleNew = (PLOCKED_MDL)lwAllocatePoolWithTag(
        NonPagedPoolNx, sizeof(LOCKED_MDL), LW_LWME_TAG);

    if (NULL == nleNew)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "AddLockedMDLToList: failed (%p)\n", nleNew);
        logSystemEvent(STATUS_INSUFFICIENT_RESOURCES, "AddLockedMDLToList: failed (%p)\n", nleNew);
        ASSERT(0);
    }

    HANDLE ProcessId = PsGetProcessId(PsGetLwrrentProcess());

    nleNew->ProcessId = ProcessId;
    nleNew->pMdl = pMDL;

    KIRQL  oldIrql;
    KeAcquireSpinLock(&g_PinnedAllocListLock, &oldIrql);
    InsertTailList(&g_PinnedAllocLockedList, &(nleNew->ListEntry));
    KeReleaseSpinLock(&g_PinnedAllocListLock, oldIrql);
}

CODE_SEGMENT(NONPAGE_CODE)
VOID
RemoveLockedMDLFromList(
    HANDLE PoricessId
)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    ASSERT(g_PinnedAllocLockedList.Flink != NULL);
    ASSERT(g_PinnedAllocLockedList.Blink != NULL);

    // If called before init list
    if (g_PinnedAllocLockedList.Flink == NULL || g_PinnedAllocLockedList.Blink == NULL)
    {
        return;
    }

    KIRQL  oldIrql;
    KeAcquireSpinLock(&g_PinnedAllocListLock, &oldIrql);

    ULONG count = 0;

    if (!IsListEmpty(&g_PinnedAllocLockedList))
    {
        PLIST_ENTRY leLwrrent = g_PinnedAllocLockedList.Flink;
        while (leLwrrent != &g_PinnedAllocLockedList)
        {
            PLOCKED_MDL nleLwrrent = CONTAINING_RECORD(leLwrrent, LOCKED_MDL, ListEntry);
            if (nleLwrrent->ProcessId == PoricessId)
            {
                TraceEvents(DPFLTR_INFO_LEVEL,
                    TRACE_QUEUE,
                    "Pinned pMDL cleared for process id = %d.\n", nleLwrrent->ProcessId);

                PLIST_ENTRY previous = leLwrrent->Blink;

                __try
                {
                    PMDL mdl = nleLwrrent->pMdl;
                    MmUnlockPages(mdl);
                    IoFreeMdl(mdl);
                    mdl = NULL;

                    count++;
                }
                __except (MyExceptionFilter(GetExceptionInformation()))
                {
                    unsigned long exceptionCode = exception_code();
                    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_DRIVER, " %s()::%d - MmUnlockPages/IoFreeMdl exception EXELWTE_HANDLER 0x%x.\n", __FUNCTION__, __LINE__, exceptionCode);
                }

                RemoveEntryList(leLwrrent);
                ExFreePoolWithTag(nleLwrrent, LW_LWME_TAG);

                leLwrrent = previous;
            }

            leLwrrent = leLwrrent->Flink;
        }
    }

    KeReleaseSpinLock(&g_PinnedAllocListLock, oldIrql);

    if (count > 0)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_DRIVER,
            "Process %lld terminated, MDL unlocked/cleaned count = 0x%x.\n", PoricessId, count);
    }
}

CODE_SEGMENT(NONPAGE_CODE)
BOOLEAN
RemoveLockedMDLFromList(
    PMDL pMDL
)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    BOOLEAN found = FALSE;

    KIRQL  oldIrql;
    KeAcquireSpinLock(&g_PinnedAllocListLock, &oldIrql);

    if (!IsListEmpty(&g_PinnedAllocLockedList))
    {
        PLIST_ENTRY leLwrrent = g_PinnedAllocLockedList.Flink;
        while (leLwrrent != &g_PinnedAllocLockedList)
        {
            PLOCKED_MDL nleLwrrent = CONTAINING_RECORD(leLwrrent, LOCKED_MDL, ListEntry);
            if (nleLwrrent->pMdl == pMDL)
            {
                TraceEvents(DPFLTR_INFO_LEVEL,
                    TRACE_QUEUE,
                    "Pinned pMDL cleared for process id = %d.\n", nleLwrrent->ProcessId);

                __try
                {
                    PMDL mdl = nleLwrrent->pMdl;
                    MmUnlockPages(mdl);
                    IoFreeMdl(mdl);
                    mdl = NULL;
                }
                __except (MyExceptionFilter(GetExceptionInformation()))
                {
                    unsigned long exceptionCode = exception_code();
                    TraceEvents(DPFLTR_VERBOSE_LEVEL, TRACE_DRIVER, " %s()::%d - MmUnlockPages/IoFreeMdl exception EXELWTE_HANDLER 0x%x.\n", __FUNCTION__, __LINE__, exceptionCode);
                }

                RemoveEntryList(leLwrrent);
                ExFreePoolWithTag(nleLwrrent, LW_LWME_TAG);

                found = TRUE;
                break;
            }

            leLwrrent = leLwrrent->Flink;
        }
    }

    KeReleaseSpinLock(&g_PinnedAllocListLock, oldIrql);

    return found;
}

CODE_SEGMENT(PAGE_CODE)
VOID 
ProcessCallback(
    HANDLE hParentId,
    HANDLE hProcessId,
    BOOLEAN bCreate)
{
    CHECK_IRQL(PASSIVE_LEVEL);

    UNREFERENCED_PARAMETER(hParentId);

    if (!bCreate) {
        RemoveLockedMDLFromList(hProcessId);
    }
}

CODE_SEGMENT(PAGE_CODE)
BOOLEAN
GetRegistryKeyValue(
    IN WDFDEVICE  WdfDevice,
    _In_ PCWSTR   Name,
    OUT PULONG    Value
)
/*++
Routine Description:
    Can be used to read any REG_DWORD registry value stored   >> HLM\System\LwrrentControlSet\Services\lwmetf\Parameters\...
Arguments:
    Name - Name of the registry value 
    Value -
Return Value:
   TRUE if successful
   FALSE if not present/error in reading registry
--*/
{
    CHECK_IRQL(PASSIVE_LEVEL);

    WDFKEY      hKey = NULL;
    NTSTATUS    status;
    BOOLEAN     retValue = FALSE;
    UNICODE_STRING valueName;

    *Value = 0;

    status = WdfDriverOpenParametersRegistryKey(
        WdfDeviceGetDriver(WdfDevice), 
        STANDARD_RIGHTS_READ, 
        WDF_NO_OBJECT_ATTRIBUTES, 
        &hKey);

    if (NT_SUCCESS(status)) {

        RtlInitUnicodeString(&valueName, Name);

        status = WdfRegistryQueryULong(hKey,
            &valueName,
            Value);

        if (NT_SUCCESS(status))
        {
            retValue = TRUE;
        }
        else
        {
            switch (status)
            {
            case STATUS_ILWALID_DEVICE_REQUEST:
                TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_DRIVER, "WdfRegistryQueryULong failed - STATUS_ILWALID_DEVICE_REQUEST \n");
                break;
            case STATUS_ILWALID_PARAMETER:
                TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_DRIVER, "WdfRegistryQueryULong failed - STATUS_ILWALID_PARAMETER \n");
                break;
            case STATUS_INSUFFICIENT_RESOURCES:
                TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_DRIVER, "WdfRegistryQueryULong failed - STATUS_INSUFFICIENT_RESOURCES \n");
                break;
            case STATUS_OBJECT_NAME_NOT_FOUND:
                TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_DRIVER, "WdfRegistryQueryULong failed - STATUS_OBJECT_NAME_NOT_FOUND \n");
                break;
            default:
                TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_DRIVER, "WdfRegistryQueryULong failed status %d \n", status);
                break;
            };
        }

        WdfRegistryClose(hKey);
    }

    return retValue;
}

#if DBG
extern const WDFFUNC* WdfFunctions;
#endif

CODE_SEGMENT(NONPAGE_CODE)
ULONG
GetIoctlStorageSetPropertyControlCode()
{
    CHECK_IRQL(HIGH_LEVEL);
    RTL_OSVERSIONINFOW osInfo;
    RtlZeroMemory(&osInfo, sizeof(RTL_OSVERSIONINFOW));
    osInfo.dwOSVersionInfoSize = sizeof(RTL_OSVERSIONINFOW);
    RtlGetVersion(&osInfo);

    ULONG controlCode;

    if (osInfo.dwBuildNumber <= 21032) 
    {
        controlCode = IOCTL_STORAGE_SET_PROPERTY_VIBRANIUM;
    }
    else 
    {
        controlCode = IOCTL_STORAGE_SET_PROPERTY_COLBAT;
    }

    static_assert( (IOCTL_STORAGE_SET_PROPERTY == IOCTL_STORAGE_SET_PROPERTY_VIBRANIUM ||
                    IOCTL_STORAGE_SET_PROPERTY == IOCTL_STORAGE_SET_PROPERTY_COLBAT), 
                   "OS is changing the control code with latest ddk. We need to update the filter driver" );

    return controlCode;
};

CODE_SEGMENT(NONPAGE_CODE)
VOID
UpdatePollingDPCState(
    _In_ PDEVICE_CONTEXT DeviceContext,
    _In_ BOOL calledFromPolling
)
{
    CHECK_IRQL(DISPATCH_LEVEL);
    // Anyone touching submit/complete/notification is required to update polling state after operation which will might have updated data.
    // So, emulating critical section via automic to restrict only single thread process function.

    LONG entryCounter = InterlockedIncrement(&DeviceContext->pollingStatustEntryCounter);
    if (entryCounter != 1)
    {
        if (calledFromPolling)
        {
            InterlockedDecrement(&DeviceContext->pollingStatustEntryCounter);
            // In case simultanious called from polling we can skip this call altogather. So, polling could not cancel SetTimerCancel which it can do in next poll..
            return;
        }

        // Spin loop to wait till we get entry count=1 to enter funtion.
        // This should happen for parallel submissions while updating polling state. (Required to handle corner cases).
        while (entryCounter != 1)
        {
            InterlockedDecrement(&DeviceContext->pollingStatustEntryCounter);
            entryCounter = InterlockedIncrement(&DeviceContext->pollingStatustEntryCounter);
        }
    }

    BOOLEAN PendingEntries = FALSE;
    BOOLEAN PendingNotifications = FALSE;
    for (ULONG index = 0; index < LW_MAX_SQS; index++)
    {
        PendingEntries = (PendingEntries || !DeviceContext->SubmissionQueues[index]->IsEmpty());
        PendingNotifications = (PendingNotifications || DeviceContext->SubmissionQueues[index]->HasPendingNotification());
    }

    POLLING_MODE expectedMode = POLLING_MODE_IDLE;
    BOOL updateTimer = FALSE;
    if (PendingNotifications || PendingEntries)
    {
        expectedMode = POLLING_MODE_ENABLE;
    }
    else
    {
        expectedMode = POLLING_MODE_IDLE;
    }

    if (expectedMode != DeviceContext->pollingMode)
    {
        DeviceContext->pollingMode = expectedMode;
        updateTimer = true;
    }

    DeviceContext->pollingMode = expectedMode;

    if (updateTimer)
    {
        switch (DeviceContext->pollingMode)
        {
            case POLLING_MODE_IDLE:
            {
                ExSetTimerResolution(DeviceContext->PollingIntervalMS, FALSE);
                KeCancelTimer(&DeviceContext->PollingTimer);
#ifdef USE_ETW_LOGGING
                EventWritePollingDisabled_AssumeEnabled(L"Polling Disabled");
#endif
                break;
            }
            case POLLING_MODE_ENABLE:
            {
                // Periodic polling timer is required when Usermode thread is registered event that it might be waiting for.
                ASSERT(DeviceContext->PollingIntervalMS > 0);

                ExSetTimerResolution((ULONG)WDF_ABS_TIMEOUT_IN_MS(DeviceContext->PollingIntervalMS), TRUE);
                LARGE_INTEGER timeout;
                timeout.QuadPart = WDF_REL_TIMEOUT_IN_MS(DeviceContext->PollingIntervalMS);
                KeSetTimerEx(&DeviceContext->PollingTimer, timeout, DeviceContext->PollingIntervalMS, &DeviceContext->PollingDpc);
#ifdef USE_ETW_LOGGING
                EventWritePollingEnabled_AssumeEnabled(L"Polling Enabled");
#endif
                break;
            }
        };
    }

    InterlockedDecrement(&DeviceContext->pollingStatustEntryCounter);

#if DBG
    if (updateTimer)
    {
        if (expectedMode == POLLING_MODE_ENABLE)
        {
            TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER, "Timer DPC Enabled.\n");
        }
        else
        {
            TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER, "Timer Canceled.\n");
        }
    }
#endif
}

CODE_SEGMENT(NONPAGE_CODE)
VOID
DoPollingTimerDpc(
    PKDPC Dpc,
    PVOID Context,
    PVOID SystemArgument1,
    PVOID SystemArgument2
)
{
    CHECK_IRQL(DISPATCH_LEVEL);
    PDEVICE_OBJECT deviceObject;

    UNREFERENCED_PARAMETER(Dpc);
    UNREFERENCED_PARAMETER(SystemArgument1);
    UNREFERENCED_PARAMETER(SystemArgument2);

    deviceObject = (PDEVICE_OBJECT)Context;
    PDEVICE_CONTEXT DeviceContext = (PDEVICE_CONTEXT)deviceObject->DeviceExtension;

    ASSERT(DeviceContext->CompletionQueue);

    BOOLEAN PendingEntries = FALSE;
    BOOLEAN PendingNotifications = FALSE;
    DeviceContext->CompletionQueue->CheckForCompletion(TRUE);
    for (ULONG index = 0; index < LW_MAX_SQS; index++)
    {
        DeviceContext->SubmissionQueues[index]->ProcessNotifications();

        PendingEntries = (PendingEntries || !DeviceContext->SubmissionQueues[index]->IsEmpty());
        PendingNotifications = (PendingNotifications || DeviceContext->SubmissionQueues[index]->HasPendingNotification());
    }

    // Polling function can change state to cancel state. Enable states can only be set by user mode call to submit work or register wait event.
    // So, it is okay to ignore calling update polling state when something pending (work/notification) or called simultaneously.
    if (PendingEntries == 0 && PendingNotifications == 0)
    {
        UpdatePollingDPCState(DeviceContext, TRUE);
    }

#ifdef USE_ETW_LOGGING
    EventWritePollingCalled_AssumeEnabled(L"DoPollingTimerDpc called");
#endif

#if DBG
    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, " Polling called Sub / conp refCounts: %llu / %llu,\t %llu / %llu,\t %llu / %llu,\t %llu / %llu,\t\n",
        DeviceContext->SubmissionQueues[0]->GetSubmissionCounter(), DeviceContext->SubmissionQueues[0]->GetCompletionCounter(),
        DeviceContext->SubmissionQueues[1]->GetSubmissionCounter(), DeviceContext->SubmissionQueues[1]->GetCompletionCounter(),
        DeviceContext->SubmissionQueues[2]->GetSubmissionCounter(), DeviceContext->SubmissionQueues[2]->GetCompletionCounter(),
        DeviceContext->SubmissionQueues[3]->GetSubmissionCounter(), DeviceContext->SubmissionQueues[3]->GetCompletionCounter());
#endif
}

CODE_SEGMENT(PAGE_CODE)
NTSTATUS
lwmetfEvtDeviceAdd(
    _In_    WDFDRIVER       Driver,
    _Inout_ PWDFDEVICE_INIT DeviceInit
)
{
    CHECK_IRQL(PASSIVE_LEVEL);

    NTSTATUS                status = STATUS_SUCCESS;
    WDF_OBJECT_ATTRIBUTES   oa;
    WDFDEVICE               device;
    PDEVICE_CONTEXT         DeviceContext;
    WDF_IO_QUEUE_CONFIG     qc;

    UNREFERENCED_PARAMETER(Driver);

    WdfFdoInitSetFilter(DeviceInit);
    WdfDeviceInitSetDeviceType(DeviceInit, FILE_DEVICE_UNKNOWN);

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&oa, DEVICE_CONTEXT);

    __try
    {
        status = WdfDeviceCreate(&DeviceInit, &oa, &device);
        if (!NT_SUCCESS(status)) {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "Failed to create device: %!STATUS!\n", status);
            logSystemEvent(status, "Failed to create device: %!STATUS!\n", status);
            return status;
        }

        DeviceContext = GetDeviceContext(device);

        WDF_IO_QUEUE_CONFIG_INIT_DEFAULT_QUEUE(&qc, WdfIoQueueDispatchParallel); // TODO - lwrrently code does not handle command index increment in submission thread protected.
        qc.EvtIoDeviceControl = lwmetfEvtIoDeviceControl;

        status = WdfIoQueueCreate(device, &qc, WDF_NO_OBJECT_ATTRIBUTES, WDF_NO_HANDLE);
        if (!NT_SUCCESS(status)) {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "Failed to create queue: %!STATUS!\n", status);
            logSystemEvent(status, "Failed to create queue: %!STATUS!\n", status);
            return status;
        }

        PDEVICE_OBJECT deviceObject = WdfDeviceWdmGetDeviceObject(device);
        KeInitializeDpc(&DeviceContext->PollingDpc, DoPollingTimerDpc, (PVOID)deviceObject);
        KeInitializeTimer(&DeviceContext->PollingTimer);

        DeviceContext->PollingIntervalMS = POLLING_RETRY_INTERVAL_MS;

        DeviceContext->pollingMode = POLLING_MODE_IDLE;
        DeviceContext->pollingStatustEntryCounter = 0;

        DeviceContext->CompletionQueue = NULL;
        RtlZeroMemory(DeviceContext->SubmissionQueues, sizeof(DeviceContext->SubmissionQueues));

        DeviceContext->CompletionQueue = new LWMeCQueue();
        // Configure Pair of Submission & completion queues
        for (ULONG index = 0; index < LW_MAX_SQS; index++)
        {
            DeviceContext->SubmissionQueues[index] = new LWMeSQueue(DeviceContext);
            DeviceContext->CompletionQueue->SetSubmissionQueue(DeviceContext->SubmissionQueues[index], index);
        }

        // After CreateReservedQueues queues will be available to use...
        DeviceContext->IsQueuesAvailable = FALSE;

        DeviceContext->m_ReadSizePerCmd = LW_MIN_READ_PER_COMMAND; // This will be updated on first Q create/handshake from caps m_pIdentifyControllerData
        DeviceContext->m_pIdentifyControllerData = NULL;

        // Refresh polling interval override at handshake with UMD. 
        ULONG pollingIntervalInMS = POLLING_RETRY_INTERVAL_MS;
        if (GetRegistryKeyValue(device, L"PollingIntervalMS", &pollingIntervalInMS))
        {
            DeviceContext->PollingIntervalMS = ((LONGLONG)pollingIntervalInMS);
            TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER, "Override PollingIntervalMS regkey: 0x%llx Millisecond, default 0x%llx\n", pollingIntervalInMS, POLLING_RETRY_INTERVAL_MS);
        }

        PDEVICE_OBJECT      plowestDeviceObject = IoGetDeviceAttachmentBaseRef(deviceObject);

        UNICODE_STRING      targetDriverName;
        RtlInitUnicodeString(&targetDriverName, L"\\Driver\\storlwme");
        if (!RtlEqualUnicodeString((PLWNICODE_STRING)&targetDriverName, (PLWNICODE_STRING)&plowestDeviceObject->DriverObject->DriverName, true))
        {
            TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_DRIVER, "Driver target is not matched.\n");
        }
        else
        {
            if (!AllocateMemoryForQueues(device)) 
            {
                TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_QUEUE, "Failed to allocate memory for queues.\n");
            }
        }

        ULONG assertOnFailure;
        if (GetRegistryKeyValue(device, L"AssertOnFailure", &assertOnFailure))
        {
            DeviceContext->m_assertOnFailure = true;
        }
    }
    __except (MyExceptionFilter(GetExceptionInformation()))
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, " %s:%d - WdfDeviceCreate / WdfIoQueueCreate exception EXELWTE_HANDLER.\n", __FUNCTION__, __LINE__);
        logSystemEvent(status, " %s:%d - WdfDeviceCreate / WdfIoQueueCreate exception EXELWTE_HANDLER.\n", __FUNCTION__, __LINE__);
    }

    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER, " %s:%d - was called.\n", __FUNCTION__, __LINE__);

    return status;
}

CODE_SEGMENT(PAGE_CODE)
VOID
lwmetfEvtDriverContextCleanup(
    _In_ WDFOBJECT DriverObject
    )
{
    CHECK_IRQL(PASSIVE_LEVEL);

    UNREFERENCED_PARAMETER(DriverObject);

#ifdef USE_ETW_LOGGING
    EventUnregisterlwmetfETW();
#endif

    PsSetCreateProcessNotifyRoutine(ProcessCallback, TRUE);

    PDEVICE_CONTEXT DriverContext = GetDeviceContext((WDFDRIVER)DriverObject);
    if (DriverContext->m_pIdentifyControllerData)
    {
        MmFreeNonCachedMemory(DriverContext->m_pIdentifyControllerData, sizeof(LWME_IDENTIFY_CONTROLLER_DATA));
        DriverContext->m_pIdentifyControllerData = NULL;
    }

    WPP_CLEANUP(WdfDriverWdmGetDriverObject((WDFDRIVER)DriverObject));

    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER, " %s:%d - was called.\n", __FUNCTION__, __LINE__);
}

CODE_SEGMENT(NONPAGE_CODE)
BOOLEAN
lwmetfForwardRequest(
    _In_ WDFDEVICE Device,
    _In_ WDFREQUEST Request,
    _In_ WDF_REQUEST_SEND_OPTIONS_FLAGS Option)
/*++

Routine Description:

    Send the request to the filter target.
    If failed, complete the request and log the status to debugger.

--*/
{
    CHECK_IRQL(DISPATCH_LEVEL);

    BOOLEAN result = STATUS_SUCCESS;
    WDF_REQUEST_SEND_OPTIONS options;

    WDF_REQUEST_SEND_OPTIONS_INIT(&options, Option);

    WdfRequestFormatRequestUsingLwrrentType(Request);
    
    __try
    {
        result = WdfRequestSend(Request, WdfDeviceGetIoTarget(Device), &options);

        if (!result)
        {
            NTSTATUS status = WdfRequestGetStatus(Request);

            TraceEvents(DPFLTR_ERROR_LEVEL,
                        TRACE_QUEUE,
                        "Failed to forward request: %!STATUS!\n", status);
            logSystemEvent(status, "Failed to forward request: %!STATUS!\n");
            WdfRequestComplete(Request, status);
        }

    }
    __except (MyExceptionFilter(GetExceptionInformation()))
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, " %s:%d WdfRequestSend / WdfRequestGetStatus / WdfRequestComplete exception EXELWTE_HANDLER.\n", __FUNCTION__, __LINE__);
        logSystemEvent(STATUS_UNEXPECTED_IO_ERROR, "%s:%d WdfRequestSend / WdfRequestGetStatus / WdfRequestComplete exception EXELWTE_HANDLER.\n", __FUNCTION__, __LINE__);
    }

    return result;
}

CODE_SEGMENT(NONPAGE_CODE)
VOID
DeleteSQueues(
    PDEVICE_CONTEXT DeviceContext
)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    DeviceContext->NumberOfSqs = 0;
    DeviceContext->CompletionQueue->DoDestroy();
    for (ULONG index = 0; index < LW_MAX_SQS; index++)
    {
        DeviceContext->SubmissionQueues[index]->DoDestroy();
    }
}

CODE_SEGMENT(NONPAGE_CODE)
VOID
FreeMemoryForQueuesAndTransfers(
    PDEVICE_CONTEXT DeviceContext
)
/*++

Routine Description:

    Free allocated queues and reset physical addresses of the queues in
    the IOCTL request buffer.

--*/
{
    CHECK_IRQL(DISPATCH_LEVEL);

    DeviceContext->NumberOfSqs = 0;
    DeviceContext->CompletionQueue->DoFreeMemory();
    for (ULONG index = 0; index < LW_MAX_SQS; index++)
    {
        DeviceContext->SubmissionQueues[index]->DoFreeMemory();
    }
}

CODE_SEGMENT(NONPAGE_CODE)
BOOLEAN
AllocateMemoryForQueues(
    _In_ WDFDEVICE Device
)
/*++
Routine Description:

    Allocate non-paged memory which is preserved for the queues usage.
    Save the queue informations into the device context.
--*/
{
    CHECK_IRQL(DISPATCH_LEVEL);

    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "AllocateMemoryForQueues is called.\n");
    BOOLEAN success = FALSE;
    PDEVICE_CONTEXT deviceContext = GetDeviceContext(Device);

    NTSTATUS status = GetIdentifyController(Device);
    USHORT queueDepthToUse = LW_DEFAULT_QUEUE_DEPTH;

    if (!NT_SUCCESS(status))
    {
        deviceContext->m_ReadSizePerCmd = LW_MIN_READ_PER_COMMAND;

        TraceEvents(DPFLTR_ERROR_LEVEL,
                    TRACE_QUEUE,
                    "AllocateMemoryForQueues Set deviceContext->m_ReadSizePerCmd as default min value %d.\n Set deviceContext->m_queueDepth as default min value %d.\n", deviceContext->m_ReadSizePerCmd, queueDepthToUse);
        logSystemEvent(status, "AllocateMemoryForQueues Set deviceContext->m_ReadSizePerCmd as default min value %d.\n Set deviceContext->m_queueDepth as default min value %d.\n");
    }
    else
    {
        deviceContext->m_ReadSizePerCmd = LWBIT(deviceContext->m_pIdentifyControllerData->MDTS) * PAGE_SIZE;
        if (deviceContext->m_ReadSizePerCmd == 0)
        {
            TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_DRIVER, "LWMe read size per (MDTS) command from IdentifyController is 0. So using default min value %d.\n", LW_MIN_READ_PER_COMMAND);
            deviceContext->m_ReadSizePerCmd = LW_MIN_READ_PER_COMMAND;
        }
        ASSERT(deviceContext->m_ReadSizePerCmd >= LW_MIN_READ_PER_COMMAND);
        ASSERT(deviceContext->m_ReadSizePerCmd % LW_CLUSTER_SIZE == 0);

        TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER, "LWMe read size: 0x%x.\n", deviceContext->m_ReadSizePerCmd);

        queueDepthToUse = GetLWMeQueueDepthFromBar0Register(Device);
    }

    ULONG overrideQueueDepth = 0;
    if (GetRegistryKeyValue(Device, L"QueueDepth", &overrideQueueDepth))
    {
        TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_DRIVER, "regkey override QueueDepth from %d to %d.\n", queueDepthToUse, (USHORT)overrideQueueDepth);
        queueDepthToUse = (USHORT)overrideQueueDepth;
    }
    deviceContext->m_queueDepth = queueDepthToUse;
    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER, "LWMe QueueDepth: 0x%x.\n", queueDepthToUse);

    if (!AllocateMemoryForQueuesAndTransfers(deviceContext, LW_MAX_SQS, queueDepthToUse))
    {
        // We failed, or the memory was already allocated.
        // In either case we logged the problem already.
        // Free the allocated memory.
        FreeMemoryForQueuesAndTransfers(deviceContext);
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Failed to allocate memory for queues!\n");
        logSystemEvent(STATUS_INSUFFICIENT_RESOURCES, "Failed to allocate memory for queues!\n");
        goto Exit;
    }

    success = TRUE;

Exit:
    return success;
}

CODE_SEGMENT(NONPAGE_CODE)
BOOLEAN
AllocateMemoryForQueuesAndTransfers(
    _In_ PDEVICE_CONTEXT DeviceContext,
    _In_ ULONG           NumberOfSQs,
    _In_ USHORT          QueueDepth
)
/*++

Routine Description:
    Allocate non-paged memory for the specified number of queues,
    save the queue informations into the device context, so we could free the queues later.
--*/
{
    CHECK_IRQL(DISPATCH_LEVEL);

    BOOLEAN success = FALSE;

    if (DeviceContext->NumberOfSqs > 0)
    {
        // We already have queues, don't overwrite them to prevent from leak memory.
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "AllocateMemoryForQueuesAndTransfers - Already allocated memory for queues and transfers!\n");
        logSystemEvent(STATUS_INSUFFICIENT_RESOURCES, "AllocateMemoryForQueuesAndTransfers - Already allocated memory for queues and transfers!\n");

        if (DeviceContext->m_assertOnFailure)
        {
            DbgBreakPoint();
        }

        goto Exit;
    }

    DeviceContext->NumberOfSqs = NumberOfSQs;

    ASSERT(DeviceContext->NumberOfSqs == LW_MAX_SQS);

    if (DeviceContext->NumberOfSqs != LW_MAX_SQS)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "CreateQp->SubmissionQueueCount is %d supported count is %d, failed!\n", NumberOfSQs, LW_MAX_SQS);
        logSystemEvent(STATUS_INSUFFICIENT_RESOURCES, "CreateQp->SubmissionQueueCount is %d supported count is %d, failed!\n", NumberOfSQs, LW_MAX_SQS);
        if (DeviceContext->m_assertOnFailure)
        {
            DbgBreakPoint();
        }

        goto Exit;
    }

    if (!DeviceContext->CompletionQueue->DoAlloc(QueueDepth))
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "CompletionQueues->DoAlloc failed!\n");
        logSystemEvent(STATUS_INSUFFICIENT_RESOURCES, "CompletionQueues->DoAlloc failed!\n");
        if (DeviceContext->m_assertOnFailure)
        {
            DbgBreakPoint();
        }

        goto Exit;
    }

    ASSERT(DeviceContext->CompletionQueue->GetPhysicalAddress());

    for (ULONG index = 0; index < LW_MAX_SQS; index++)
    {
        if (!DeviceContext->SubmissionQueues[index]->DoAlloc(QueueDepth))
        {
            TraceEvents(DPFLTR_ERROR_LEVEL,
                TRACE_QUEUE,
                "pSubmissionQueues[%d]->DoAlloc failed! \n", index);
            logSystemEvent(STATUS_INSUFFICIENT_RESOURCES, "pSubmissionQueues[%d]->DoAlloc failed! \n", index);

            if (DeviceContext->m_assertOnFailure)
            {
                DbgBreakPoint();
            }

            goto Exit;
        }

        ASSERT(DeviceContext->SubmissionQueues[index]->GetPhysicalAddress());
    }
    success = TRUE;

Exit:
    return success;
}

CODE_SEGMENT(NONPAGE_CODE)
NTSTATUS
ReservedQueueQuery(
    _In_ WDFDEVICE Device,
    _In_ PSRB_IO_CONTROL SrbIoControl
)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "ReservedQueueQuery is called.\n");
    PDEVICE_CONTEXT        deviceContext    = GetDeviceContext(Device);
    NTSTATUS               status           = STATUS_SUCCESS;
    PLWME_DRIVER_HANDSHAKE pDriverHandShake = (PLWME_DRIVER_HANDSHAKE)((PCHAR)SrbIoControl + sizeof(SRB_IO_CONTROL));

    if (deviceContext->NumberOfSqs)
    {
        pDriverHandShake->CompletionQueue = deviceContext->CompletionQueue->GetReserveCQueueInfo();

        for (ULONG sq = 0; sq < deviceContext->NumberOfSqs; sq++)
        {
            pDriverHandShake->SubmissionQueue[sq] = deviceContext->SubmissionQueues[sq]->GetReserveSQueueInfo();
        }
    }
    else
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "No Queues are created but land here somehow. Failed status = 0x%x.\n", status);
        logSystemEvent(STATUS_INSUFFICIENT_RESOURCES, "No Queues are created but land here somehow. Failed status = 0x%x.\n", status);
        if (deviceContext->m_assertOnFailure)
        {
            DbgBreakPoint();
        }

        status = STATUS_DRIVER_INTERNAL_ERROR;
    }

    return status;
}

CODE_SEGMENT(PAGE_CODE)
NTSTATUS CreateReservedQueue(
    _In_ WDFDEVICE Device,
    _In_ PSRB_IO_CONTROL SrbIoControl
)
{
    CHECK_IRQL(APC_LEVEL);

    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "CreateReservedQueue is called.\n");

    NTSTATUS            status = STATUS_SUCCESS;
    PDEVICE_OBJECT      pDeviceObject = WdfDeviceWdmGetDeviceObject(Device);
    PDEVICE_CONTEXT     deviceContext = GetDeviceContext(Device);
    PDEVICE_OBJECT      plowestDeviceObject = IoGetDeviceAttachmentBaseRef(pDeviceObject);

    KEVENT              event = {};
    KeInitializeEvent(&event, NotificationEvent, FALSE);

    IO_STATUS_BLOCK     ioStatus = {};
    PIRP                irp;

    ULONG               queueCount = LW_MAX_SQS;
    ULONG               bufferLength = sizeof(SRB_IO_CONTROL) +
                                       sizeof(LWME_RESERVED_QUEUES_CREATE_REQUEST) +
                                       sizeof(LWME_RESERVED_QUEUES_CREATE_RESPONSE);

    CREATE_QUEUES_BUFFER createQueueBuffer;
    ULONG                createQueueBufferLength = sizeof(CREATE_QUEUES_BUFFER);
    RtlZeroMemory(&createQueueBuffer, createQueueBufferLength);
    createQueueBuffer.SrbIoCtl.HeaderLength = sizeof(SRB_IO_CONTROL);
    createQueueBuffer.SrbIoCtl.ControlCode = GetIoctlStorageSetPropertyControlCode();
    RtlMoveMemory(createQueueBuffer.SrbIoCtl.Signature, IOCTL_LWME_SIGNATURE_CREATE_RESERVED_QUEUE_PAIR, 8);
    createQueueBuffer.SrbIoCtl.Timeout = 30;
    createQueueBuffer.SrbIoCtl.Length = bufferLength - sizeof(SRB_IO_CONTROL);

    // Framing create request and setting create response offset
    createQueueBuffer.CreateQueueRequest.ResponseDataBufferOffset =
        FIELD_OFFSET(LWME_RESERVED_QUEUES_CREATE_REQUEST, SubmissionQueue) + (queueCount * sizeof(LWME_RESERVED_SQ_CREATE_REQUEST));
    createQueueBuffer.CreateQueueRequest.ResponseDataBufferLength =
        FIELD_OFFSET(LWME_RESERVED_QUEUES_CREATE_RESPONSE, SubmissionQueue) + (queueCount * sizeof(LWME_RESERVED_SQ_INFO));

    // Get completion queue information from device content
    createQueueBuffer.CreateQueueRequest.CompletionQueue.PhysicalAddress = deviceContext->CompletionQueue->GetPhysicalAddress();
    createQueueBuffer.CreateQueueRequest.CompletionQueue.InterruptVector = 1;
    createQueueBuffer.CreateQueueRequest.CompletionQueue.QueueDepth = deviceContext->m_queueDepth;
    createQueueBuffer.CreateQueueRequest.CompletionQueue.InterruptEnabled = 0;
    createQueueBuffer.CreateQueueRequest.CompletionQueue.PhysicalContiguous = 1;

    createQueueBuffer.CreateQueueRequest.SubmissionQueueCount = LW_MAX_SQS;

    // Get submission queues information from device content (4 queues for 1 completion queue)
    for (ULONG i = 0; i < deviceContext->NumberOfSqs; i++)
    {
        createQueueBuffer.CreateQueueRequest.SubmissionQueue[i].PhysicalAddress = deviceContext->SubmissionQueues[i]->GetPhysicalAddress();
        createQueueBuffer.CreateQueueRequest.SubmissionQueue[i].QueuePriority = i % LW_MAX_SQS;
        createQueueBuffer.CreateQueueRequest.SubmissionQueue[i].QueueDepth = deviceContext->m_queueDepth;
        createQueueBuffer.CreateQueueRequest.SubmissionQueue[i].PhysicalContiguous = 1;
    }

    PUCHAR createQueueBuff = (PUCHAR)MmAllocateNonCachedMemory(bufferLength);
    if (!createQueueBuff)
    {
        status = STATUS_INSUFFICIENT_RESOURCES;
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Failed to allocate buffer %d size!\n", bufferLength);
        logSystemEvent(status, "Failed to allocate buffer %d size!\n", bufferLength);

        if (deviceContext->m_assertOnFailure)
        {
            DbgBreakPointWithStatus(status);
        }

        goto Exit;
    }

    RtlZeroMemory(createQueueBuff, bufferLength);
    RtlCopyMemory(createQueueBuff, &createQueueBuffer.SrbIoCtl, sizeof(SRB_IO_CONTROL));
    RtlCopyMemory(createQueueBuff + sizeof(SRB_IO_CONTROL),
                  &createQueueBuffer.CreateQueueRequest,
                  sizeof(LWME_RESERVED_QUEUES_CREATE_REQUEST));

    PLWME_RESERVED_QUEUES_CREATE_REQUEST  pSendCQRequest;
    PLWME_RESERVED_QUEUES_CREATE_RESPONSE pRQCResponse;

    pSendCQRequest = (PLWME_RESERVED_QUEUES_CREATE_REQUEST)((PUCHAR)createQueueBuff + sizeof(SRB_IO_CONTROL));

    irp = IoBuildDeviceIoControlRequest(IOCTL_SCSI_MINIPORT,
                                        plowestDeviceObject,
                                        createQueueBuff,
                                        bufferLength,
                                        createQueueBuff,
                                        bufferLength,
                                        false,
                                        &event,
                                        &ioStatus);

    if (irp == NULL)
    {
        status = STATUS_INSUFFICIENT_RESOURCES;
    }
    else
    {
        // Send the request
        status = IoCallDriver(plowestDeviceObject, irp);

        if (status == STATUS_PENDING)
        {
            if ((status = KeWaitForSingleObject(&event, Exelwtive, KernelMode, FALSE, NULL)) == STATUS_SUCCESS)
            {
                status = ioStatus.Status;
            }
        }
    }

    if (!NT_SUCCESS(status))
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Reserved queue creation failure on sending IOCTL_SCSI_MINIPORT control call!\n");
        logSystemEvent(status, "Reserved queue creation failure on sending IOCTL_SCSI_MINIPORT control call!\n");

        if (deviceContext->m_assertOnFailure)
        {
            DbgBreakPoint();
        }

        goto Exit;
    }

    pRQCResponse = (PLWME_RESERVED_QUEUES_CREATE_RESPONSE)((PCHAR)pSendCQRequest + pSendCQRequest->ResponseDataBufferOffset);

    BOOL validResponse = TRUE;
    if (pRQCResponse->CompletionQueue.SubmissionQueueCount != LW_MAX_SQS)
    {
        TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_QUEUE, "submission queue count  %d != %d. Bad response from storlwme.sys.\n", pRQCResponse->CompletionQueue.SubmissionQueueCount, LW_MAX_SQS);
        validResponse = FALSE;
    }
    else if (pRQCResponse->CompletionQueue.QueueDepth == 0)
    {
        TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_QUEUE, "Got 0 queue depth for CQ. Bad response from storlwme.sys.\n");
        validResponse = FALSE;
    }
    else if (pRQCResponse->SubmissionQueue[0].QueueDepth == 0 ||
             pRQCResponse->SubmissionQueue[1].QueueDepth == 0 ||
             pRQCResponse->SubmissionQueue[2].QueueDepth == 0 ||
             pRQCResponse->SubmissionQueue[3].QueueDepth == 0)
    {
        TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_QUEUE, "Get 0 queue depth for SQ. Bad response from storlwme.sys.\n");
        validResponse = FALSE;
    }

    if (validResponse == FALSE)
    {
        status = STATUS_INSUFFICIENT_RESOURCES;
        DeleteSQueues(deviceContext);

        goto Exit;
    }

    deviceContext->CompletionQueue->DoCreate(&pRQCResponse->CompletionQueue);
    ASSERT(pRQCResponse->CompletionQueue.QueueID == deviceContext->CompletionQueue->GetQueueID());
    ASSERT(deviceContext->CompletionQueue->IsValid());

    ASSERT(pRQCResponse->CompletionQueue.QueueDepth > 0);
    for (ULONG sq = 0; sq < pSendCQRequest->SubmissionQueueCount; sq++)
    {
        deviceContext->SubmissionQueues[sq]->DoCreate(&pRQCResponse->SubmissionQueue[sq]);
        ASSERT(pRQCResponse->SubmissionQueue[sq].QueueDepth > 0);
        ASSERT(deviceContext->SubmissionQueues[sq]->IsValid());
        ASSERT(deviceContext->SubmissionQueues[sq]->IsEmpty());
        ASSERT(deviceContext->SubmissionQueues[sq]->GetPriority() == sq);
        ASSERT(deviceContext->SubmissionQueues[sq]->GetCompletionQueueID() == pRQCResponse->CompletionQueue.QueueID);
    }

    deviceContext->IsQueuesAvailable = TRUE;

    PLWME_DRIVER_HANDSHAKE pDriverHandShake = (PLWME_DRIVER_HANDSHAKE)((PCHAR)SrbIoControl + sizeof(SRB_IO_CONTROL));
    pDriverHandShake->CompletionQueue = pRQCResponse->CompletionQueue;
    RtlCopyMemory(&pDriverHandShake->SubmissionQueue, pRQCResponse->SubmissionQueue, LW_MAX_SQS * sizeof(LWME_RESERVED_SQ_INFO));

Exit:

    if (createQueueBuff)
    {
        MmFreeNonCachedMemory(createQueueBuff, bufferLength);
        createQueueBuff = NULL;
    }

    return status;
}

CODE_SEGMENT(NONPAGE_CODE)
VOID HandleLWMetfDriverHandShake(
    _In_ WDFDEVICE Device,
    _In_ WDFREQUEST Request,
    _In_ PSRB_IO_CONTROL SrbIoControl,
    _In_ size_t OutputBufferLength,
    _In_ size_t InputBufferLength
) 
{
    CHECK_IRQL(DISPATCH_LEVEL);

    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "HandleLWMetfDriverHandShake called.\n");
  
    NTSTATUS            status = STATUS_SUCCESS;
    PDEVICE_OBJECT      pDeviceObject = WdfDeviceWdmGetDeviceObject(Device);
    PDEVICE_CONTEXT     deviceContext = GetDeviceContext(Device);
    PDEVICE_OBJECT      plowestDeviceObject = IoGetDeviceAttachmentBaseRef(pDeviceObject);

    UNICODE_STRING      targetDriverName;
    RtlInitUnicodeString(&targetDriverName, L"\\Driver\\storlwme");
    if (!RtlEqualUnicodeString((PLWNICODE_STRING)&targetDriverName, (PLWNICODE_STRING)&plowestDeviceObject->DriverObject->DriverName, true))
    {
        status = STATUS_ILWALID_DEVICE_STATE;
        TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_QUEUE, "Driver target is not matched.\n");
        WdfRequestCompleteWithInformation(Request, status, 0);
        logSystemEvent(status, "Driver target is not matched.\n");
        if (deviceContext->m_assertOnFailure)
        {
            DbgBreakPointWithStatus(status);
        }
        goto Exit;
    }

    // Make sure the system buffer is large enough for the specfied number of queues.
    if (InputBufferLength < sizeof(SRB_IO_CONTROL) + sizeof(PLWME_DRIVER_HANDSHAKE))
    {
        status = STATUS_BUFFER_TOO_SMALL;
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "IOCTL system input buffer size is too big for the driver handshake.\n");
        if (deviceContext->m_assertOnFailure)
        {
            DbgBreakPointWithStatus(status);
        }

        goto Exit;
    }

    // Make sure the response fits in the system buffer.
    if (OutputBufferLength < sizeof(SRB_IO_CONTROL) + sizeof(PLWME_DRIVER_HANDSHAKE))
    {
        status = STATUS_BUFFER_TOO_SMALL;
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "IOCTL system buffer size is too small for the output.\n");
        logSystemEvent(status, "IOCTL system buffer size is too small for the output.\n");
        if (deviceContext->m_assertOnFailure)
        {
            DbgBreakPointWithStatus(status);
        }

        goto Exit;
    }

    //1. Check if Q's are avaliable
    if (deviceContext->IsQueuesAvailable)
    {
        // We already have queues, don't overwrite them to avoid memory leak.
        TraceEvents(DPFLTR_INFO_LEVEL,
                    TRACE_QUEUE,
                    "Already allocated memory for queues and transfers!\n");
        // Query the submitted queue
        status = ReservedQueueQuery(Device, SrbIoControl);
    }
    else //2. If not, create reserved queues
    {
        status = CreateReservedQueue(Device, SrbIoControl);

        if (!NT_SUCCESS(status))
        {
            TraceEvents(DPFLTR_ERROR_LEVEL,
                TRACE_QUEUE,
                "Create reserved queue failed.\n");
            logSystemEvent(status, "IOCTL system buffer size is too small for the output.\n");
            if (deviceContext->m_assertOnFailure)
            {
                DbgBreakPointWithStatus(status);
            }
        }
    }
    
    //3. Map allocation of completion of ref count.
    if (status == STATUS_SUCCESS)
    {
        status = CreateMemoryMapping(Device, SrbIoControl);
    }

#ifdef DBG
    // Refresh polling interval override at handshake with UMD. 
    ULONG pollingIntervalInMS = POLLING_RETRY_INTERVAL_MS;
    if (GetRegistryKeyValue(Device, L"PollingIntervalUS", &pollingIntervalInMS))
    {
        deviceContext->PollingIntervalMS = ((LONGLONG)pollingIntervalInMS);
        TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER, "Override PollingIntervalMS regkey: 0x%llx Millisecond, default 0x%llx\n", pollingIntervalInMS, POLLING_RETRY_INTERVAL_MS);
    }
#endif

Exit:
    WdfRequestCompleteWithInformation(Request, status, OutputBufferLength);
    return;
}

CODE_SEGMENT(NONPAGE_CODE)
NTSTATUS
MappingMemory(__in PVOID Va, 
              __in ULONG size,
              __out PMEMORY_REQUEST_DATA memoryRequest)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    PHYSICAL_ADDRESS Address;
    NTSTATUS status = STATUS_SUCCESS;
    MEMORY_REQUEST_DATA memReq;
    
    __try
    {
        RtlZeroMemory(&memReq, sizeof(MEMORY_REQUEST_DATA));
        RtlZeroMemory(&Address, sizeof(PHYSICAL_ADDRESS));
        Address.QuadPart = MmGetPhysicalAddress(Va).QuadPart;
        memReq.Length = size;
        __try
        {
            memReq.MapIo = MmMapIoSpace(Address,
                memReq.Length,
                MmNonCached);
            if (memReq.MapIo == NULL)
            {
                TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, " %s:%d , failed at memReq.MapIo == NULL\n", __FUNCDNAME__,__LINE__);
                status = STATUS_INSUFFICIENT_RESOURCES;
                logSystemEvent(status, " %s:%d , failed at memReq.MapIo == NULL\n", __FUNCDNAME__, __LINE__);
                __leave;
            };
            //
            memReq.Mdl = IoAllocateMdl(memReq.MapIo,
                memReq.Length,
                FALSE,
                FALSE,
                NULL);
            if (memReq.Mdl == NULL)
            {
                TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, " %s:%d , failed at memReq.Mdl\n", __FUNCDNAME__, __LINE__);
                status = STATUS_INSUFFICIENT_RESOURCES;
                logSystemEvent(status, " %s:%d , failed at memReq.Mdl\n", __FUNCDNAME__, __LINE__);
                __leave;
            };
            //
            MmBuildMdlForNonPagedPool(memReq.Mdl);
            memReq.Buffer =
                (PUCHAR)MmMapLockedPagesSpecifyCache(memReq.Mdl,
                    UserMode,
                    MmNonCached,
                    NULL,
                    FALSE,
                    NormalPagePriority);
            if (memReq.Buffer == NULL)
            {
                TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, " %s:%d ,failed at memReq.Buffer == NULL\n", __FUNCDNAME__, __LINE__);
                status = STATUS_INSUFFICIENT_RESOURCES;
                logSystemEvent(status, " %s:%d ,failed at memReq.Buffer == NULL\n", __FUNCDNAME__, __LINE__);
                __leave;
            };
            //
            RtlCopyMemory(memoryRequest, &memReq, sizeof(MEMORY_REQUEST_DATA));
            status = STATUS_SUCCESS;
        }
        __except(EXCEPTION_EXELWTE_HANDLER)
        {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, " %s:%d ,failed at except\n", __FUNCDNAME__, __LINE__);
            status = STATUS_INSUFFICIENT_RESOURCES;
            logSystemEvent(status, " %s:%d ,failed at except\n", __FUNCDNAME__, __LINE__);
            __leave;
        };
    }
    __finally
    {
    };
    return status;
};

CODE_SEGMENT(NONPAGE_CODE)
NTSTATUS CreateMemoryMapping(
    _In_ WDFDEVICE Device,
    _In_ PSRB_IO_CONTROL SrbIoControl
)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "CreateMemoryMapping called.\n");

    PDEVICE_CONTEXT     deviceContext = GetDeviceContext(Device);
    NTSTATUS status = STATUS_SUCCESS;
    // Map alloc parameter
    if (deviceContext->IsQueuesAvailable == FALSE)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Reserved Queues are not yet created.\n");
        logSystemEvent(STATUS_UNEXPECTED_IO_ERROR, " %s:%d ,Reserved Queues are not yet created.\n", __FUNCDNAME__, __LINE__);
        goto Exit;
    }

    for (USHORT index = 0; index < LW_MAX_SQS; index++)
    {
        PMEMORY_REQUEST_DATA memMapData = &(deviceContext->SQCompleteRefCountMappingData[index]);

        ASSERT(deviceContext->SubmissionQueues[index]);
        ASSERT(deviceContext->SubmissionQueues[index]->IsValid());

        status = MappingMemory(deviceContext->SubmissionQueues[index]->GetCompletionCounterPointer(), sizeof(ULONGLONG), memMapData);
        if (status != STATUS_SUCCESS)
        {
            status = STATUS_UNEXPECTED_IO_ERROR;
            TraceEvents(DPFLTR_ERROR_LEVEL,
                TRACE_QUEUE,
                "MappingMemory failed (DeviceContext->CompletionRefCount) status = 0x%x.\n", status);
            logSystemEvent(status, " %s:%d ,MappingMemory failed (DeviceContext->CompletionRefCount\n", __FUNCDNAME__, __LINE__);
            goto Exit;
        }
        PLWME_DRIVER_HANDSHAKE pDriverHandShake = (PLWME_DRIVER_HANDSHAKE)((PCHAR)SrbIoControl + sizeof(SRB_IO_CONTROL));
        pDriverHandShake->SQCompletionRefCountVA[index] = (ULONGLONG)memMapData->Buffer;

        TraceEvents(DPFLTR_INFO_LEVEL,
            TRACE_QUEUE,
            "Completion ref count user mode VA = 0x%llx.\n", (ULONGLONG)memMapData->Buffer);
    }
Exit:
    return status;
}

CODE_SEGMENT(PAGE_CODE)
VOID
RegisterEventWait(
    _In_ WDFDEVICE Device,
    _In_ WDFREQUEST Request,
    _In_ PSRB_IO_CONTROL SrbIoControl,
    _In_ size_t OutputBufferLength,
    _In_ size_t InputBufferLength
)
{
    CHECK_IRQL(PASSIVE_LEVEL);

    UNREFERENCED_PARAMETER(InputBufferLength);

    if (InputBufferLength < sizeof(LWME_REGISTER_EVENT_WAIT))
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Not enough size of input parameter RegisterEventWait\n");
        logSystemEvent(STATUS_DRIVER_INTERNAL_ERROR, " %s:%d ,Not enough size of input parameter RegisterEventWait\n", __FUNCDNAME__, __LINE__);
        WdfRequestCompleteWithInformation(Request, STATUS_DRIVER_INTERNAL_ERROR, OutputBufferLength);
        goto Exit;
    }


    PLWME_REGISTER_EVENT_WAIT command = (PLWME_REGISTER_EVENT_WAIT)((PUCHAR)SrbIoControl + sizeof(SRB_IO_CONTROL));
    PDEVICE_CONTEXT DeviceContext = GetDeviceContext(Device);

    if (DeviceContext->IsQueuesAvailable == FALSE)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "RegisterEventWait - Reserved Queues are not yet created.\n");
        logSystemEvent(STATUS_DRIVER_INTERNAL_ERROR, " %s:%d ,RegisterEventWait - Reserved Queues are not yet created\n", __FUNCDNAME__, __LINE__);
        WdfRequestCompleteWithInformation(Request, STATUS_ILWALID_DEVICE_STATE, OutputBufferLength);
        goto Exit;
    }

    for (USHORT index = 0; index < command->inCount; index++)
    {
        USHORT queuePriority = command->eventData[index].inQueuePriority;
        ULONGLONG refCount = command->eventData[index].inRefCount;
        HANDLE eventHandle = command->eventData[index].inUserEvent;

        ASSERT(DeviceContext->SubmissionQueues[queuePriority]);
        ASSERT(DeviceContext->SubmissionQueues[queuePriority]->IsValid());

        NTSTATUS status = DeviceContext->SubmissionQueues[queuePriority]->RegisterNotification(eventHandle, refCount);
        if (status != STATUS_SUCCESS)
        {
            TraceEvents(DPFLTR_ERROR_LEVEL,
                TRACE_QUEUE,
                "RegisterEventWait failed status = 0x%x.\n", status);
            logSystemEvent(status, " %s:%d ,RegisterEventWait failed ", __FUNCDNAME__, __LINE__);
            WdfRequestCompleteWithInformation(Request, status, 0);

            goto Exit;
        }

        TraceEvents(DPFLTR_VERBOSE_LEVEL,
            TRACE_QUEUE,
            "Registered user event (%p) priority %d signal on ref count = 0x%llx.\n", eventHandle, queuePriority, refCount);
    }

    // Scan for all completion and so we can avoid timerDPC if everything is complete already.
    DeviceContext->CompletionQueue->CheckForCompletion(FALSE);
    for (ULONG index = 0; index < LW_MAX_SQS; index++)
    {
        DeviceContext->SubmissionQueues[index]->ProcessNotifications();
    }

    UpdatePollingDPCState(DeviceContext, FALSE);

    // Mark test success!
    WdfRequestCompleteWithInformation(Request, STATUS_SUCCESS, OutputBufferLength);

Exit:
    return;
}

CODE_SEGMENT(PAGE_CODE)
NTSTATUS
handleReadBatch2(_In_ WDFDEVICE  Device,
                 _In_ WDFREQUEST Request,
                 _In_ size_t OutputBufferLength,
                 _In_ size_t InputBufferLength)
{
    CHECK_IRQL(PASSIVE_LEVEL);

    UNREFERENCED_PARAMETER(OutputBufferLength);
    NTSTATUS status = STATUS_SUCCESS;
    PLWME_READ_COMMAND_BUFFER2 readBatchReq = NULL;

    status = WdfRequestRetrieveInputBuffer(Request, sizeof(LWME_READ_COMMAND_BUFFER2), (PVOID*)&readBatchReq, NULL);
    if (!NT_SUCCESS(status))
    {
        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "DoReadBatch2 WdfRequestRetrieveInputBuffer failed status=%d.\n", status);
        logSystemEvent(status, " %s:%d ,DoReadBatch2 WdfRequestRetrieveInputBuffer failed ", __FUNCDNAME__, __LINE__);
        goto Exit;
    }

    if (InputBufferLength < sizeof(LWME_READ_COMMAND_BUFFER2))
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "InputBufferLength too small. InputBufferLength = %d Expected = %d \n", InputBufferLength, sizeof(SRB_IO_CONTROL) + sizeof(LWME_READ_COMMAND_BUFFER2));
        logSystemEvent(status, " %s:%d ,InputBufferLength too small. InputBufferLength = %d Expected = %d \n", __FUNCDNAME__, __LINE__, InputBufferLength, sizeof(SRB_IO_CONTROL) + sizeof(LWME_READ_COMMAND_BUFFER2));
        status = STATUS_BUFFER_TOO_SMALL;

        goto Exit;
    }

    PDEVICE_CONTEXT DeviceContext = GetDeviceContext(Device);
    if (DeviceContext->IsQueuesAvailable == FALSE)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "DoReadBatch - Reserved Queues are not yet created.\n");
        logSystemEvent(status, " %s:%d ,DoReadBatch - Reserved Queues are not yet created.\n", __FUNCDNAME__, __LINE__);
        status = STATUS_ILWALID_DEVICE_STATE;

        goto Exit;
    }

    if (!readBatchReq->inKernelVA)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "invalid input - command buffer inKernelVA cannot be null.\n");
        logSystemEvent(status, " %s:%d ,invalid input - command buffer inKernelVA cannot be null.\n", __FUNCDNAME__, __LINE__);
        status = STATUS_DRIVER_INTERNAL_ERROR;

        goto Exit;
    }

    if (readBatchReq->inEntryCount == 0)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "invalid input - inEntryCount cannot be 0.\n");

        status = STATUS_DRIVER_INTERNAL_ERROR;
        logSystemEvent(status, " %s:%d ,invalid input - inEntryCount cannot be 0.\n", __FUNCDNAME__, __LINE__);
        goto Exit;
    }

    PLWME_READ_COMAMND_ENTRY readEntry = (PLWME_READ_COMAMND_ENTRY)readBatchReq->inKernelVA;
    LWME_READ readCommand;
    RtlZeroBytes(&readCommand, sizeof(LWME_READ));

#ifdef USE_ETW_LOGGING 
    EventWriteSubmitStart_AssumeEnabled((UINT32)((UINT64)&readBatchReq), L"handleReadBatch2 start");
#endif

    ULONGLONG submitCounter[LW_MAX_SQS];
    RtlZeroMemory(submitCounter, sizeof(submitCounter));

    for (ULONG index = 0; index < readBatchReq->inEntryCount; index++)
    {
        readCommand.inLBA = readEntry[index].LBA;
        readCommand.inReadSize = readEntry[index].ReadSize;
        readCommand.addr.IsUserVA = readEntry[index].IsAllolwserVA;
        readCommand.addr.inAllocBaseVA = (PVOID)readEntry[index].AllocVA;
        readCommand.addr.inAllocSize = readEntry[index].ReadSize;
        readCommand.inDebugPerCmdReadSize = readBatchReq->inDebugPerCmdReadSize;
        readCommand.outSubmissionRefCount = 0;

        const ULONG QueuePriority = readEntry[index].QueuePriority;
        ASSERT(readCommand.addr.inAllocSize >= readCommand.inReadSize);
        ASSERT(readCommand.inReadSize > 0);
        ASSERT(readCommand.addr.inAllocBaseVA);
        ASSERT(readCommand.inLBA);

        ASSERT(QueuePriority < LW_MAX_SQS);

        if (QueuePriority >= LW_MAX_SQS)
        {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "Submit skipped for entry# %d of %d, invalid queue priority %d.\n", index, readBatchReq->inEntryCount, QueuePriority);
            status = STATUS_DRIVER_INTERNAL_ERROR;
            logSystemEvent(status, " %s:%d Submit skipped for entry#  % d of % d, invalid queue priority % d.\n", __FUNCDNAME__, __LINE__, index, readBatchReq->inEntryCount, QueuePriority);
            goto Exit;
        }

        ASSERT(DeviceContext->SubmissionQueues[QueuePriority]->IsValid());
        status = DeviceContext->SubmissionQueues[QueuePriority]->DoSubmitRead(&readCommand);
        if (!NT_SUCCESS(status))
        {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "DoReadBatch failed for entry# %d of %d, queue priority %d.\n", index, readBatchReq->inEntryCount, QueuePriority);
            logSystemEvent(status, " %s:%d ,DoReadBatch failed for entry# %d of %d, queue priority %d.\n", __FUNCDNAME__, __LINE__ , index, readBatchReq->inEntryCount, QueuePriority);
            goto Exit;
        }

        ASSERT(readCommand.outSubmissionRefCount > 0);

        submitCounter[QueuePriority] = readCommand.outSubmissionRefCount;
        
    }

    for (USHORT index = 0; index < LW_MAX_SQS; index++)
    {
        ULONGLONG refCount = submitCounter[index]; // refCount is zero if submission didnt happen on that queue
        HANDLE hUserEvent = readBatchReq->inUserEvents[index];

        if (hUserEvent && refCount > 0)
        {
            status = DeviceContext->SubmissionQueues[index]->RegisterNotification(hUserEvent, refCount);
            if (status != STATUS_SUCCESS)
            {
                TraceEvents(DPFLTR_ERROR_LEVEL,
                    TRACE_QUEUE,
                    "RegisterEventWait failed status = 0x%x.\n", status);
                logSystemEvent(status, " %s:%d RegisterEventWait failed\n", __FUNCDNAME__, __LINE__);

                goto Exit;
            }
        }
        else if (hUserEvent) // Need to signal event since there was no submission in respective priority queue.
        {
            PKEVENT pKrnlEvent = NULL;

            // Signal event
            status = ObReferenceObjectByHandle(hUserEvent,
                SYNCHRONIZE,
                *ExEventObjectType,
                UserMode,
                reinterpret_cast<PVOID*>(&pKrnlEvent),
                NULL);
            if (!NT_SUCCESS(status))
            {
                TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "RegisterNotification: ObReferenceObjectByHandle failed, handle is not valid (%p), ststus = 0x%x ", hUserEvent, status);
                logSystemEvent(status, " %s:%d RegisterNotification: ObReferenceObjectByHandle failed, handle is not valid (%p)", __FUNCDNAME__, __LINE__, hUserEvent);
                switch (status)
                {
                case STATUS_OBJECT_TYPE_MISMATCH:
                    TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "STATUS_OBJECT_TYPE_MISMATCH\n");
                    break;
                case STATUS_ACCESS_DENIED:
                    TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "STATUS_ACCESS_DENIED\n");
                    break;
                case STATUS_ILWALID_HANDLE:
                    TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "STATUS_ILWALID_HANDLE\n");
                    break;
                default:
                    TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "STATUS_UNKNOWN\n");
                };

                goto Exit;
            }

            if (!pKrnlEvent)
            {
                TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "RegisterNotification: ObReferenceObjectByHandle returned NULL (%p), ststus = 0x%x\n", hUserEvent, status);
                status = STATUS_DRIVER_INTERNAL_ERROR;
                logSystemEvent(status, " %s:%d RegisterNotification: ObReferenceObjectByHandle returned NULL (%p)", __FUNCDNAME__, __LINE__, hUserEvent);
                goto Exit;
            }

            KeSetEvent(pKrnlEvent, 0, FALSE);
            ObDereferenceObject(pKrnlEvent);
        }

        TraceEvents(DPFLTR_VERBOSE_LEVEL,
            TRACE_QUEUE,
            "Registered user event (%p) priority %d signal on ref count = 0x%llx.\n", hUserEvent, index, refCount);
    }

    DeviceContext->CompletionQueue->CheckForCompletion(FALSE);

    for (USHORT index = 0; index < LW_MAX_SQS; index++)
    {
        readBatchReq->outSubmissionRefCount[index] = submitCounter[index];

#ifdef USE_ETW_LOGGING
        if (submitCounter[index])
        {
            // Writing out only max refcouts after submit.
            EventWriteSubmissionRef_AssumeEnabled(index, submitCounter[index]);
        }
#endif

        DeviceContext->SubmissionQueues[index]->ProcessNotifications();
    }

    ASSERT(readBatchReq->outSubmissionRefCount[0] ||
        readBatchReq->outSubmissionRefCount[1] ||
        readBatchReq->outSubmissionRefCount[2] ||
        readBatchReq->outSubmissionRefCount[3]);

    UpdatePollingDPCState(DeviceContext, FALSE);

#ifdef USE_ETW_LOGGING 
    EventWriteSubmitEnd_AssumeEnabled((UINT32)((UINT64)&readBatchReq), L"handleReadBatch2 End");
#endif

    TraceEvents(DPFLTR_TRACE_LEVEL, TRACE_DRIVER, "DoReadBatch with %d entries.\n", readBatchReq->inEntryCount);

Exit:
    WdfRequestCompleteWithInformation(Request, status, OutputBufferLength);

    return status;
}

CODE_SEGMENT(PAGE_CODE)
NTSTATUS
DoReadBatch(
    _In_ WDFDEVICE Device,
    _In_ WDFREQUEST Request,
    _In_ PSRB_IO_CONTROL SrbIoControl,
    _In_ size_t OutputBufferLength,
    _In_ size_t InputBufferLength
)
{
    CHECK_IRQL(PASSIVE_LEVEL);

    UNREFERENCED_PARAMETER(InputBufferLength);
    UNREFERENCED_PARAMETER(OutputBufferLength);
    UNREFERENCED_PARAMETER(Request);

    PDEVICE_CONTEXT DeviceContext = GetDeviceContext(Device);
    NTSTATUS status = STATUS_SUCCESS;

    if (DeviceContext->IsQueuesAvailable == FALSE)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "DoReadBatch - Reserved Queues are not yet created.\n");
        status = STATUS_ILWALID_DEVICE_STATE;
        logSystemEvent(status, " %s:%d DoReadBatch - Reserved Queues are not yet created.", __FUNCDNAME__, __LINE__);
        goto Exit;
    }

    PLWME_READ_COMMAND_BUFFER readBatchReq = (PLWME_READ_COMMAND_BUFFER)((PUCHAR)SrbIoControl + sizeof(SRB_IO_CONTROL));
    PLWME_READ_COMAMND_ENTRY readEntry = (PLWME_READ_COMAMND_ENTRY)readBatchReq->inKernelVA;
    LWME_READ readCommand;
    RtlZeroBytes(&readCommand, sizeof(LWME_READ));

    ULONGLONG submitCounter[LW_MAX_SQS];
    RtlZeroMemory(submitCounter, sizeof(submitCounter));

    for (ULONG index = 0; index < readBatchReq->inEntryCount; index++)
    {
        readCommand.inLBA = readEntry[index].LBA;
        readCommand.inReadSize = readEntry[index].ReadSize;
        readCommand.addr.IsUserVA = readEntry[index].IsAllolwserVA;
        readCommand.addr.inAllocBaseVA = (PVOID)readEntry[index].AllocVA;
        readCommand.addr.inAllocSize = readEntry[index].ReadSize;
        readCommand.inDebugPerCmdReadSize = readBatchReq->inDebugPerCmdReadSize;
        
        ASSERT(readCommand.addr.inAllocSize >= readCommand.inReadSize);
        ASSERT(readCommand.inReadSize > 0);
        ASSERT(readCommand.addr.inAllocBaseVA);
        ASSERT(readCommand.inLBA);
        
        ASSERT(readEntry[index].QueuePriority < LW_MAX_SQS);

        if (readEntry[index].QueuePriority >= LW_MAX_SQS)
        {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "Submit skipped for entry# %d of %d, invalid queue priority %d.\n", index, readBatchReq->inEntryCount, readEntry[index].QueuePriority);
            logSystemEvent(STATUS_ILWALID_DEVICE_STATE, " %s:%d Submit skipped for entry# %d of %d, invalid queue priority %d.\n", __FUNCDNAME__, __LINE__ , index, readBatchReq->inEntryCount, readEntry[index].QueuePriority);
            continue;
        }

        ASSERT(DeviceContext->SubmissionQueues[readEntry[index].QueuePriority]->IsValid());
        status = DeviceContext->SubmissionQueues[readEntry[index].QueuePriority]->DoSubmitRead(&readCommand);
        if (!NT_SUCCESS(status))
        {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "DoReadBatch failed for entry# %d of %d, queue priority %d.\n", index, readBatchReq->inEntryCount, readEntry[index].QueuePriority);
            logSystemEvent(STATUS_ILWALID_DEVICE_STATE, "%s:%d DoReadBatch failed for entry#  % d of % d, queue priority % d.\n", __FUNCDNAME__, __LINE__, index, readBatchReq->inEntryCount, readEntry[index].QueuePriority);
            goto Exit;
        }

        ASSERT(readCommand.outSubmissionRefCount > 0);

        submitCounter[readEntry[index].QueuePriority] = readCommand.outSubmissionRefCount;
    }

    DeviceContext->CompletionQueue->CheckForCompletion(FALSE);

    for (ULONG index = 0; index < LW_MAX_SQS; index++)
    {
        readBatchReq->outSubmissionRefCount[index] = submitCounter[index];

        DeviceContext->SubmissionQueues[index]->ProcessNotifications();
    }

    ASSERT( readBatchReq->outSubmissionRefCount[0] ||
            readBatchReq->outSubmissionRefCount[1] ||
            readBatchReq->outSubmissionRefCount[2] ||
            readBatchReq->outSubmissionRefCount[3]);

    UpdatePollingDPCState(DeviceContext, FALSE);

    TraceEvents(DPFLTR_TRACE_LEVEL, TRACE_DRIVER, "DoReadBatch with %d entries.\n", readBatchReq->inEntryCount);

Exit:
    WdfRequestCompleteWithInformation(Request, status, OutputBufferLength);

    return status;
}

CODE_SEGMENT(NONPAGE_CODE)
NTSTATUS
DoPinUnPinAllocation(
    PDEVICE_CONTEXT DeviceContext,
    PLWME_PIN_UNPIN_ALLOC request
)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    UNREFERENCED_PARAMETER(DeviceContext); // TODO: need to store mdl with DeviceContext/somewhere so we can clean up incase app missed/crashed.

#ifdef DBG
    ULONGLONG DurationMDL = 0;
    ULONGLONG startMDLTickCount = KeQueryInterruptTime();
#endif
    
    PVOID userSpaceAllocKernelVA = NULL;
    NTSTATUS status = STATUS_SUCCESS;

    if (request->IsPinAlloc == TRUE)
    {
        PMDL mdl = NULL;

        if (request->para.pinData.inAllocBaseVA ==0 || request->para.pinData.inAllocSize == 0 || request->para.pinData.inAllocSize >= ULONG_MAX)
        {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "Allocation parameter invalid.\n");
            status = STATUS_ILWALID_PARAMETER_2;
            logSystemEvent(STATUS_ILWALID_DEVICE_STATE, "%s:%d Allocation parameter invalid. \n", __FUNCDNAME__, __LINE__);
            goto Exit;
        }

        ASSERT(request->para.pinData.inAllocBaseVA);
        ASSERT(request->para.pinData.inAllocSize);
        ASSERT(request->para.pinData.inAllocSize <= ULONG_MAX);

        mdl = IoAllocateMdl(request->para.pinData.inAllocBaseVA, (ULONG)request->para.pinData.inAllocSize, FALSE, FALSE, NULL);
        if (!mdl)
        {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "Failed to IoAllocateMdl, userVA ox%llx, size 0x%llx\n", request->para.pinData.inAllocBaseVA, request->para.pinData.inAllocSize);
            status = STATUS_INSUFFICIENT_RESOURCES;
            logSystemEvent(STATUS_ILWALID_DEVICE_STATE,
                "%s:%d Failed to IoAllocateMdl, userVA ox%llx, size 0x%llx\n",
                __FUNCDNAME__, __LINE__, request->para.pinData.inAllocBaseVA, request->para.pinData.inAllocSize);
            goto Exit;
        }

        KIRQL Irql;
        // Callers of MmProbeAndLockPages must be running at IRQL <= APC_LEVEL for pageable addresses, or at IRQL <= DISPATCH_LEVEL for nonpageable addresses.
        KeRaiseIrql(APC_LEVEL, &Irql);
        __try
        {
            MmProbeAndLockPages(mdl, UserMode, IoWriteAccess);

        }
        __except(EXCEPTION_EXELWTE_HANDLER)
        {
            KeLowerIrql(Irql);

            status = GetExceptionCode();
            IoFreeMdl(mdl);
            mdl = NULL;
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "Failed to MmProbeAndLockPages(mdl, UserMode, IoWriteAccess), status 0x%x\n", status);
            logSystemEvent(status, "%s:%d Failed to MmProbeAndLockPages(mdl, UserMode, IoWriteAccess)", __FUNCDNAME__, __LINE__);

            goto Exit;
        }

        KeLowerIrql(Irql);

        userSpaceAllocKernelVA = (PVOID)MmGetSystemAddressForMdlSafe(mdl, NormalPagePriority | MdlMappingNoExelwte);
        if (!userSpaceAllocKernelVA) {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, "Failed to MmGetSystemAddressForMdlSafe(mdl, NormalPagePriority | MdlMappingNoExelwte)\n");
            MmUnlockPages(mdl);
            IoFreeMdl(mdl);
            mdl = NULL;
            status = STATUS_INSUFFICIENT_RESOURCES;
            logSystemEvent(status, "%s:%d Failed to MmGetSystemAddressForMdlSafe(mdl, NormalPagePriority | MdlMappingNoExelwte)", __FUNCDNAME__, __LINE__);
            goto Exit;
        }
        TraceEvents(DPFLTR_INFO_LEVEL, TRACE_QUEUE, "userSpaceAllocKernelVA = 0x%llx, Physical = 0x%llx\n", userSpaceAllocKernelVA, MmGetPhysicalAddress(userSpaceAllocKernelVA));

        ASSERT(userSpaceAllocKernelVA);
        ASSERT(mdl);

        if (request->para.pinData.inAllocSize > MmGetMdlByteCount(mdl))
        {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE,
                "Size in Allocation MmGetMdlByteCount(mdl) 0x%llx not enough for read data size 0x%llx \n",
                MmGetMdlByteCount(mdl), request->para.pinData.inAllocSize);
            logSystemEvent(STATUS_INSUFFICIENT_RESOURCES,
                "%s:%d Size in Allocation MmGetMdlByteCount(mdl) 0x%llx not enough for read data size 0x%llx \n",
                __FUNCDNAME__, __LINE__, MmGetMdlByteCount(mdl), request->para.pinData.inAllocSize);
            MmUnlockPages(mdl);
            IoFreeMdl(mdl);
            mdl = NULL;

            goto Exit;
        }

        request->para.pinData.outHandle = (ULONGLONG)mdl;
        request->para.pinData.outKernelVA = (ULONGLONG)userSpaceAllocKernelVA;

        AddLockedMDLToList(mdl);
    }
    else if (request->para.unpinData.inHandle != NULL)
    {
        ASSERT(request->para.unpinData.inHandle);

        PMDL mdl = (PMDL)request->para.unpinData.inHandle;

        BOOLEAN foundAndRemoved = RemoveLockedMDLFromList(mdl);
        if (!foundAndRemoved)
        {
            // Fallback - incase insert into list was failed due to some reason we try to clean here..
            MmUnlockPages(mdl);
            IoFreeMdl(mdl);
            mdl = NULL;
        }
    }
    else
    {
        // So we have IsPinAlloc=TRUE and request->para.unpinData.inHandle = NULL, this is incorrect parameters.
        status = STATUS_ILWALID_PARAMETER_2;
    }

#ifdef DBG
    DurationMDL += KeQueryInterruptTime() - startMDLTickCount;
#endif

Exit:
    return status;
}

CODE_SEGMENT(NONPAGE_CODE)
NTSTATUS
DoPinUnPinBatch(
    _In_ WDFDEVICE Device,
    _In_ WDFREQUEST Request,
    _In_ PSRB_IO_CONTROL SrbIoControl,
    _In_ size_t OutputBufferLength,
    _In_ size_t InputBufferLength
)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    UNREFERENCED_PARAMETER(InputBufferLength);
    UNREFERENCED_PARAMETER(OutputBufferLength);
    UNREFERENCED_PARAMETER(Request);

    PDEVICE_CONTEXT DeviceContext = GetDeviceContext(Device);
    NTSTATUS status = STATUS_SUCCESS;

    if (DeviceContext->IsQueuesAvailable == FALSE)
    {
        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "DoPinUnPinBatch - Reserved Queues are not yet created.\n");
        status = STATUS_ILWALID_DEVICE_STATE;
        logSystemEvent(status, "%s:%d DoPinUnPinBatch - Reserved Queues are not yet created.", __FUNCDNAME__, __LINE__);
        goto Exit;
    }

    PLWME_PIN_UNPIN_ALLOC_BATCH request = (PLWME_PIN_UNPIN_ALLOC_BATCH)((PUCHAR)SrbIoControl + sizeof(SRB_IO_CONTROL));

    for (ULONG index = 0; index < request->requestCount; index++)
    {
        status = DoPinUnPinAllocation(DeviceContext, &request->requests[index]);

        if (!NT_SUCCESS(status))
        {
            break;
        }
    }

Exit:
    WdfRequestCompleteWithInformation(Request, status, OutputBufferLength);

    return status;
}

CODE_SEGMENT(NONPAGE_CODE)
// BusyPolling - function for testing only and not used by RTXIO LWAPI.
NTSTATUS
WaitForRefCount(
    _In_ WDFDEVICE Device,
    _In_ WDFREQUEST Request,
    _In_ PSRB_IO_CONTROL SrbIoControl,
    _In_ size_t OutputBufferLength,
    _In_ size_t InputBufferLength
)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    UNREFERENCED_PARAMETER(InputBufferLength);

    PDEVICE_CONTEXT DeviceContext = GetDeviceContext(Device);
    PLWME_WAIT_FOR_REF waitCmd = (PLWME_WAIT_FOR_REF)((PUCHAR)SrbIoControl + sizeof(SRB_IO_CONTROL));
    NTSTATUS status = STATUS_SUCCESS;
    ULONG64 startTickCount = KeQueryInterruptTime();
    
#if DBG
    LARGE_INTEGER Freq;
    LARGE_INTEGER enterTimeDBG = KeQueryPerformanceCounter(&Freq);
#endif

    if (waitCmd->TimeOutInMiliSeconds >= (ULONGLONG)WDF_ABS_TIMEOUT_IN_SEC(60))
    {
        status = STATUS_ILWALID_PARAMETER_3;

        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Maximum allowed timeout is 60sec but passed 0x%llx miliseconds\n",
            waitCmd->TimeOutInMiliSeconds);
        logSystemEvent(status, "%s:%d Maximum allowed timeout is 60sec but passed 0x%llx miliseconds.\n", __FUNCDNAME__, __LINE__);
        goto Exit;
    }

    if (waitCmd->QueuePriority >= LW_MAX_SQS)
    {
        status = STATUS_ILWALID_PARAMETER_2;

        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Incorrect QueuePriority parameter %d\n",
            waitCmd->QueuePriority);
        logSystemEvent(status, "%s:%d Incorrect QueuePriority parameter", __FUNCDNAME__, __LINE__);
        goto Exit;
    }

    if (waitCmd->RefCount > DeviceContext->SubmissionQueues[waitCmd->QueuePriority]->GetSubmissionCounter())
    {
        status = STATUS_ILWALID_PARAMETER_1;

        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Incorrect refcount parameter. submission counter 0x%llx < and waitForRef 0x%llx.\n",
            DeviceContext->SubmissionQueues[waitCmd->QueuePriority]->GetSubmissionCounter(),
            waitCmd->RefCount);
        logSystemEvent(status, "%s:%d Incorrect refcount parameter. submission counter 0x%llx < and waitForRef 0x%llx.\n", __FUNCDNAME__, __LINE__);
        goto Exit;
    }

    ULONGLONG attempt = 0;
    ULONGLONG CQRefAtStart = DeviceContext->SubmissionQueues[waitCmd->QueuePriority]->GetCompletionCounter();

    while (*(DeviceContext->SubmissionQueues[waitCmd->QueuePriority]->GetCompletionCounterPointer()) < waitCmd->RefCount)
    {
        // Wait for completion.
        NTSTATUS waitStatus = DeviceContext->CompletionQueue->CheckForCompletion(FALSE);
        attempt++;
        if (waitStatus == STATUS_UNEXPECTED_IO_ERROR)
        {
            status = STATUS_UNEXPECTED_IO_ERROR;
            goto Exit;
        }

        if (waitCmd->TimeOutInMiliSeconds == 0)
        {
            break;
        }

        ULONGLONG durationIn100ns = (KeQueryInterruptTime() - startTickCount);
        BOOLEAN isTimeoutPassed = (durationIn100ns * 10) > waitCmd->TimeOutInMiliSeconds;
        if (!isTimeoutPassed)
        {
            ASSERT(waitCmd->TimeOutInMiliSeconds != 0);

#if USE_SLEEP 
            LARGE_INTEGER preSleep = KeQueryPerformanceCounter(NULL);
            INT64 intervalHalfMiliSec = -500i64 * 1000i64 / 100i64; /// ([milli] * (-1[relative] * 500[milli to micro] * 1000[micro to nano]) / 100[ns]
            {
                KeDelayExelwtionThread(KernelMode, FALSE, (PLARGE_INTEGER)&intervalHalfMiliSec);
            }

#if DBG
            LARGE_INTEGER postSleep = KeQueryPerformanceCounter(NULL);
            LONGLONG sleepTimeUS = ((postSleep.QuadPart - preSleep.QuadPart) * 1000000) / Freq.QuadPart;

            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, " KeDelayExelwtionThread duration: %llu us\n", sleepTimeUS);
            logSystemEvent(status, "%s:%d KeDelayExelwtionThread duration: %llu us\n", __FUNCDNAME__, __LINE__, sleepTimeUS);
#endif // DBG
#endif // USE_SLEEP


            // Lets give it a retry to finish work on completion queue..
            continue;
        }
        else
        {
            break;
        }
    }

    if (waitCmd->RefCount > DeviceContext->SubmissionQueues[waitCmd->QueuePriority]->GetCompletionCounter() && status == STATUS_SUCCESS)
    {
        status = STATUS_TIMEOUT;

        TraceEvents(DPFLTR_ERROR_LEVEL,
            TRACE_QUEUE,
            "Could not reach refCount 0x%llx in last 0x%llx attempts. SQ[%d] completion RefCount before 0x%llx now 0x%llx.\n",
            waitCmd->RefCount, attempt, waitCmd->QueuePriority, CQRefAtStart, DeviceContext->SubmissionQueues[waitCmd->QueuePriority]->GetCompletionCounter());
        logSystemEvent(status,
            "%s:%d Could not reach refCount 0x%llx in last 0x%llx attempts. SQ[%d] completion RefCount before 0x%llx now 0x%llx.\n",
            __FUNCDNAME__, __LINE__,
            waitCmd->RefCount, attempt, waitCmd->QueuePriority, CQRefAtStart, DeviceContext->SubmissionQueues[waitCmd->QueuePriority]->GetCompletionCounter());
    }

#if DBG
    ULONGLONG refCountNow = DeviceContext->SubmissionQueues[waitCmd->QueuePriority]->GetCompletionCounter();
    if (waitCmd->TimeOutInMiliSeconds != 0 &&
        status != STATUS_UNEXPECTED_IO_ERROR &&
        waitCmd->RefCount > refCountNow)
    {
        LARGE_INTEGER lwrrentTime = KeQueryPerformanceCounter(NULL);

        LONGLONG totalTimeUS = ((lwrrentTime.QuadPart - enterTimeDBG.QuadPart) * 1000000) / Freq.QuadPart;

        TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_QUEUE, " Total time in WaitForRefCount funtion: %llu us\n", totalTimeUS);
        logSystemEvent(status,
            "%s:%d  Total time in WaitForRefCount funtion: %llu us\n",
            __FUNCDNAME__, __LINE__, totalTimeUS);
    }
#endif

Exit:
    WdfRequestCompleteWithInformation(Request, status, OutputBufferLength);

    return status;
}

CODE_SEGMENT(NONPAGE_CODE)
BOOLEAN 
handlelwmetfEvtIoctlRequests(_In_ WDFDEVICE     device,
                             _In_ WDFREQUEST    Request,
                             _In_ size_t        OutputBufferLength,
                             _In_ size_t        InputBufferLength)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    PSRB_IO_CONTROL srbControl = NULL;
    BOOLEAN         fwdRequest = FALSE;
    NTSTATUS status = WdfRequestRetrieveInputBuffer(Request, sizeof(SRB_IO_CONTROL), (PVOID*)&srbControl, NULL);

    if (NT_SUCCESS(status) &&
        (srbControl->ControlCode == GetIoctlStorageSetPropertyControlCode()))
    {
        if (RtlCompareMemory(&srbControl->Signature, IOCTL_LWME_SIGNATURE_FILTER_DRIVER_HANDSHAKE, 8) == 8)
        {
            HandleLWMetfDriverHandShake(device, Request, srbControl, OutputBufferLength, InputBufferLength);

            TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_DRIVER,
                " %s:%d -LWMeTF driver handshake complete. IOCTL_LWME_SIGNATURE_FILTER_DRIVER_HANDSHAKE called.\n", __FUNCTION__, __LINE__);
        }
        else if (RtlCompareMemory(&srbControl->Signature, IOCTL_LWME_SIGNATURE_LWME_WAIT_FOR_REF, 8) == 8)
        {
            WaitForRefCount(device, Request, srbControl, OutputBufferLength, InputBufferLength);
        }
        else if (RtlCompareMemory(&srbControl->Signature, IOCTL_LWME_SIGNATURE_REGISTER_EVENT_WAIT, 8) == 8)
        {
            RegisterEventWait(device, Request, srbControl, OutputBufferLength, InputBufferLength);
        }
        else if (RtlCompareMemory(&srbControl->Signature, IOCTL_LWME_SIGNATURE_LWME_DO_READ_BATCH, 8) == 8)
        {
            DoReadBatch(device, Request, srbControl, OutputBufferLength, InputBufferLength);
        }
        else if (RtlCompareMemory(&srbControl->Signature, IOCTL_LWME_SIGNATURE_LWME_PIN_UNPIN_ALLOC, 8) == 8)
        {
            DoPinUnPinBatch(device, Request, srbControl, OutputBufferLength, InputBufferLength);
        }
        else
        {
            fwdRequest = TRUE;
        }
    }
    else
    {
        fwdRequest = TRUE;
    }
    return fwdRequest;
}

CODE_SEGMENT(PAGE_CODE)
NTSTATUS
retrieveDataRunLBA(PDEVICE_CONTEXT  deviceContext,
                   HANDLE           fileHandle,
                   HANDLE           volumeHandle,
                   LONGLONG         vcnToRetrieve,
                   LONGLONG*        lba,
                   LONGLONG*        dataRunLength,
                   LONGLONG*        nextVcn)
{
    CHECK_IRQL(PASSIVE_LEVEL);

    STARTING_VCN_INPUT_BUFFER   inputVCN;
    RETRIEVAL_POINTERS_BUFFER   outputExtent;
    LONGLONG                    clustersLength;
    NTSTATUS                    ntStatus;
    IO_STATUS_BLOCK             ioStatus;

    *nextVcn       = 0;
    clustersLength = 0;
    inputVCN.StartingVcn.QuadPart = vcnToRetrieve;
    ntStatus = ZwFsControlFile(fileHandle,
                               NULL,
                               NULL,
                               NULL,
                               &ioStatus,
                               FSCTL_GET_RETRIEVAL_POINTERS,
                               &inputVCN,
                               sizeof(inputVCN),
                               &outputExtent,
                               sizeof(outputExtent));
    switch (ntStatus)
    {
        case STATUS_BUFFER_OVERFLOW:
        {
            TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER,
                        ": Warning (0x%x) STATUS_BUFFER_OVERFLOW. Caller should update VCN and get retrieval pointer again",
                       ntStatus);
        }
        case STATUS_MORE_ENTRIES:
        case STATUS_SUCCESS:
        {
            *nextVcn = outputExtent.Extents[0].NextVcn.QuadPart;
        }
        case STATUS_NO_MORE_ENTRIES:
        {
            clustersLength = outputExtent.Extents[0].NextVcn.QuadPart - outputExtent.StartingVcn.QuadPart;
            if (ntStatus == STATUS_SUCCESS)
            {
                ntStatus = STATUS_MORE_ENTRIES;
            }
        }
        break;
        case STATUS_END_OF_FILE:
        {
            TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER, ": Reached end of file\n");
        }
        break;
        default:
        {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, ": Error(0x % x) on sending FSCTL_GET_RETRIEVAL_POINTERS\n", ntStatus);
            logSystemEvent(ntStatus, "%s:%d  Error(0x % x) on sending FSCTL_GET_RETRIEVAL_POINTERS\n", __FUNCDNAME__, __LINE__, ntStatus);
        }
    }
    if ((ntStatus == STATUS_MORE_ENTRIES) ||
        (ntStatus == STATUS_NO_MORE_ENTRIES) ||
        (ntStatus == STATUS_BUFFER_OVERFLOW))
    {
        LONGLONG                startSector;
        LONGLONG                noOfSectors;
        VOLUME_LOGICAL_OFFSET   volumeLogicalOffset;
        VOLUME_PHYSICAL_OFFSETS volumePhysicalOffsets;
        NTSTATUS                volumeStatus;

        volumeLogicalOffset.LogicalOffset = outputExtent.Extents[0].Lcn.QuadPart * deviceContext->m_bytesPerCluster;
        volumeStatus = ZwDeviceIoControlFile(volumeHandle,
                                             NULL,
                                             NULL,
                                             NULL,
                                             &ioStatus,
                                             IOCTL_VOLUME_LOGICAL_TO_PHYSICAL,
                                             &volumeLogicalOffset,
                                             sizeof(VOLUME_LOGICAL_OFFSET),
                                             &volumePhysicalOffsets,
                                             sizeof(VOLUME_PHYSICAL_OFFSETS));
        if (volumeStatus != STATUS_SUCCESS)
        {
            TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, ": Error (0x%x) on sending IOCTL_VOLUME_LOGICAL_TO_PHYSICAL\n", ntStatus);
            ntStatus = volumeStatus;
            logSystemEvent(ntStatus, "%s:%d   Error (0x%x) on sending IOCTL_VOLUME_LOGICAL_TO_PHYSICAL\n", __FUNCDNAME__, __LINE__, ntStatus);
        }
        startSector = volumePhysicalOffsets.PhysicalOffset[0].Offset / deviceContext->m_bytesPerSector;
        // startSector += <clusterStart>; Adjust for FAT32
        noOfSectors = (clustersLength * deviceContext->m_bytesPerCluster) / deviceContext->m_bytesPerSector;
        *dataRunLength = noOfSectors * deviceContext->m_bytesPerSector;
        *lba = startSector; // Todo: Do adjustment if required
    }
    return ntStatus;
} // retrieveDataRunLBA

CODE_SEGMENT(PAGE_CODE)
NTSTATUS
retriveLBAMappings(PDEVICE_CONTEXT      deviceContext,
                   HANDLE               fileHandle,
                   HANDLE               volumeHandle,
                   LBAMAPPING_INFO*     LBAMappingInfo)
{
    CHECK_IRQL(PASSIVE_LEVEL);
 
    LONGLONG    vcnToRetrieve = LBAMappingInfo->startVCNtoRetrieve;
    LONGLONG    lba;
    LONGLONG    dataRunLength;
    LONGLONG    nextVcn = 0;
    ULONG       lbaIndex = 0;
    NTSTATUS    lbaRetrieveStatus = STATUS_SUCCESS;

    while (lbaIndex < LBA_MAPPING_LENGTH &&
           lbaRetrieveStatus != STATUS_NO_MORE_ENTRIES)
    {
        lbaRetrieveStatus = retrieveDataRunLBA(deviceContext,
                                               fileHandle,
                                               volumeHandle,
                                               vcnToRetrieve,
                                               &lba,
                                               &dataRunLength,
                                               &nextVcn);
        if ((lbaRetrieveStatus == STATUS_MORE_ENTRIES) ||
            (lbaRetrieveStatus == STATUS_NO_MORE_ENTRIES) ||
            (lbaRetrieveStatus == STATUS_BUFFER_OVERFLOW))
        {
            LBAMappingInfo->LBAMapping[lbaIndex].LBA = lba;
            LBAMappingInfo->LBAMapping[lbaIndex].length = dataRunLength;

            TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER, ": LBA: 0x%llx, Length 0x%llx\n", lba, dataRunLength);
            lbaIndex++;
            // Move to the next Data Run
            vcnToRetrieve = nextVcn;
        }
        else if (lbaRetrieveStatus == STATUS_END_OF_FILE)
        {
            TraceEvents(DPFLTR_INFO_LEVEL, TRACE_DRIVER, ": : Reached end of file\n");
            break;
        }
        else
        {
            TraceEvents(DPFLTR_WARNING_LEVEL, TRACE_DRIVER, ": Error on retrieving LBA mapping\n");
            break;
        }
    }
    
    if (lbaIndex == LBA_MAPPING_LENGTH)
    {
        LBAMappingInfo->bRequireExtraCall = true;
        LBAMappingInfo->nextVCNtoRetrieve = nextVcn;
    }
    LBAMappingInfo->retrivedLength = lbaIndex;

    return STATUS_SUCCESS;
}

CODE_SEGMENT(PAGE_CODE)
BOOLEAN
handlelwmetfEvtIoctlMapLBA
(
    _In_ WDFDEVICE     device,
    _In_ WDFREQUEST    Request,
    _In_ size_t        OutputBufferLength,
    _In_ size_t        InputBufferLength
)
{
    CHECK_IRQL(PASSIVE_LEVEL);

    OBJECT_ATTRIBUTES           obj;
    BOOLEAN                     volumeHandleCreated = FALSE;
    UNICODE_STRING              dosFilePath;
    UNICODE_STRING              completeFilePath;
    UNICODE_STRING              usVolumeLetter;
    UNICODE_STRING              completeVolumePath;
    CHAR                        volumeLetter[3];
    ANSI_STRING                 ansiVolumeLetter;
    UNICODE_STRING              filePath;
    ANSI_STRING                 ansiFilePath;
    IO_STATUS_BLOCK             ioStatus;
    HANDLE                      fileHandle;
    HANDLE                      volumeHandle;
    PDEVICE_CONTEXT             deviceContext = GetDeviceContext(device);
    LBAMAPPING_INFO*            LBAMapping = NULL;

    UNREFERENCED_PARAMETER(OutputBufferLength);
    UNREFERENCED_PARAMETER(InputBufferLength);

    NTSTATUS status = WdfRequestRetrieveOutputBuffer(Request, sizeof(LBAMAPPING_INFO), (PVOID*)&LBAMapping, NULL);
    if (!NT_SUCCESS(status))
    {
        return FALSE;
    }

    RtlInitAnsiString(&ansiFilePath, (LBAMapping->completFilePath));
    RtlAnsiStringToUnicodeString(&filePath, &ansiFilePath, TRUE);
    RtlInitUnicodeString(&dosFilePath, L"\\DosDevices\\\0");

    for (int i = 0; i < MAX_PATH; i++)
    {
        if (LBAMapping->completFilePath[i] == '\0')
        {
            break;
        }

        if ((i >= 1) && (LBAMapping->completFilePath[i] == ':'))
        {
            volumeLetter[0] = LBAMapping->completFilePath[i - 1];
            volumeLetter[1] = LBAMapping->completFilePath[i];
            volumeLetter[2] = '\0';
        }
    }

    RtlInitAnsiString(&ansiVolumeLetter, volumeLetter);
    RtlAnsiStringToUnicodeString(&usVolumeLetter, &ansiVolumeLetter, TRUE);
    completeVolumePath.Length = 0;
    completeVolumePath.MaximumLength = dosFilePath.Length + (ansiVolumeLetter.Length * sizeof(WCHAR)) + sizeof(WCHAR);
    completeVolumePath.Buffer = (PWCH)lwAllocatePoolWithTag(PagedPool, completeVolumePath.MaximumLength, LW_LWME_TAG);
    RtlAppendUnicodeStringToString(&completeVolumePath, &dosFilePath);
    RtlAppendUnicodeStringToString(&completeVolumePath, &usVolumeLetter);

    completeFilePath.Length = 0;
    completeFilePath.MaximumLength = dosFilePath.Length + (ansiFilePath.Length * sizeof(WCHAR)) + sizeof(WCHAR);
    completeFilePath.Buffer = (PWCH)lwAllocatePoolWithTag(PagedPool, completeFilePath.MaximumLength, LW_LWME_TAG);
    RtlAppendUnicodeStringToString(&completeFilePath, &dosFilePath);
    RtlAppendUnicodeStringToString(&completeFilePath, &filePath);

    RtlZeroMemory(&obj, sizeof(obj));
    InitializeObjectAttributes(&obj, &completeFilePath, OBJ_CASE_INSENSITIVE | OBJ_KERNEL_HANDLE, NULL, NULL);

    NTSTATUS ntStatus = ZwCreateFile(&fileHandle,
                                     0/*ACCESS_MASK*/,
                                     &obj,
                                     &ioStatus,
                                     0/*shareAccess*/,
                                     FILE_ATTRIBUTE_NORMAL,
                                     0,
                                     FILE_OPEN_IF,
                                     FILE_NON_DIRECTORY_FILE,
                                     NULL,
                                     0);
    
    if (ntStatus == STATUS_SUCCESS)
        // TODO: Add the consideration of already retrieved information about the volume; 
        //      i.e., no need to retrieve same volume information again
    {
        RtlZeroMemory(&obj, sizeof(obj));
        InitializeObjectAttributes(&obj, &completeVolumePath, OBJ_CASE_INSENSITIVE | OBJ_KERNEL_HANDLE, NULL, NULL);

        ntStatus = ZwCreateFile(&volumeHandle,
                                GENERIC_WRITE | GENERIC_READ,
                                &obj,
                                &ioStatus,
                                0,
                                FILE_ATTRIBUTE_NORMAL,
                                FILE_SHARE_WRITE | FILE_SHARE_READ,
                                FILE_OPEN_IF,
                                FILE_NON_DIRECTORY_FILE,
                                NULL,
                                0);
        if (ntStatus == STATUS_SUCCESS)
        {
            volumeHandleCreated = TRUE;
            FILE_FS_FULL_SIZE_INFORMATION fsFSSizeInformation;
            ntStatus = ZwQueryVolumeInformationFile(volumeHandle,
                                                    &ioStatus,
                                                    &fsFSSizeInformation,
                                                    sizeof(FILE_FS_FULL_SIZE_INFORMATION),
                                                    FileFsFullSizeInformation);
            if (ntStatus == STATUS_SUCCESS)
            {
                deviceContext->m_bytesPerSector     = fsFSSizeInformation.BytesPerSector;
                deviceContext->m_sectorsPerCluster  = fsFSSizeInformation.SectorsPerAllocationUnit;
            }
            else
            {
                TraceEvents(DPFLTR_ERROR_LEVEL, TRACE_DRIVER, "Error 0x%x on quering file volume information\n", ntStatus);
                logSystemEvent(ntStatus, "%s:%d Error 0x%x on quering file volume information\n", __FUNCDNAME__, __LINE__, ntStatus);
                deviceContext->m_bytesPerSector = 8;
                deviceContext->m_sectorsPerCluster = 0x200;
            }
            deviceContext->m_bytesPerCluster = deviceContext->m_bytesPerSector * deviceContext->m_sectorsPerCluster;
            ntStatus = retriveLBAMappings(deviceContext,
                                          fileHandle,
                                          volumeHandle,
                                          LBAMapping);
        }
        if (volumeHandleCreated)
        {
            ZwClose(volumeHandle);
        }
        ZwClose(fileHandle);
        RtlFreeUnicodeString(&filePath);
        ExFreePoolWithTag(completeFilePath.Buffer, LW_LWME_TAG);
    }

    WdfRequestCompleteWithInformation(Request, ntStatus, OutputBufferLength);

    return TRUE;
}

CODE_SEGMENT(NONPAGE_CODE)
VOID
lwmetfEvtIoDeviceControl(
    _In_ WDFQUEUE Queue,
    _In_ WDFREQUEST Request,
    _In_ size_t OutputBufferLength,
    _In_ size_t InputBufferLength,
    _In_ ULONG IoControlCode
)
{
    CHECK_IRQL(DISPATCH_LEVEL);

    WDFDEVICE device = WdfIoQueueGetDevice(Queue);

    BOOLEAN   fwdRequest = FALSE;
    switch (IoControlCode)
    {
    case LW_LWME_IOCTL_DRIVER_REQUEST:
        fwdRequest = handlelwmetfEvtIoctlRequests(device, Request, OutputBufferLength, InputBufferLength);
        break;
    case LW_LWME_IOCTL_RETRIEVE_IOCTL:
        handlelwmetfEvtIoctlMapLBA(device, Request, OutputBufferLength, InputBufferLength);
        fwdRequest = FALSE;
        break;
    case LW_LWME_IOCTL_SUBMIT_READS:
        handleReadBatch2(device, Request, OutputBufferLength, InputBufferLength);
        fwdRequest = FALSE;
        break;
    default:
        fwdRequest = TRUE;
        break;
    }
    if (fwdRequest)
    {
        lwmetfForwardRequest(device, Request, WDF_REQUEST_SEND_OPTION_SEND_AND_FORGET);
    }
    return;
}

#define MAX_ERROR_STRING 256
char TooBigString[] = { "Variable String too Large" };

// Colwert 1 byte to 2 byte unicode string
PWSTR
AsciiToUnicode(PWSTR pDest, char* pSrc)
{
    while ('\0' != *pSrc)
        *pDest++ = *pSrc++;

    *pDest++ = '\0';

    return pDest;
}

void
logSystemEvent(NTSTATUS errorCode, const char* pFormat, ...)
{
    va_list arglist;

    va_start(arglist, pFormat);
    logSystemEventV(errorCode, pFormat, arglist);
    va_end(arglist);
}

void
logSystemEventV(NTSTATUS errorCode, const char* pFormat, va_list arglist)
{
    PIO_ERROR_LOG_PACKET pIoErrorLog;
    PDRIVER_OBJECT pDriverObject;
    PVOID pContext2;
    
    PWSTR pStr;
    UINT uMsgSize;
    UINT numChars;
    UINT maxString;
    char  String[MAX_ERROR_STRING];

    if ((pFormat == NULL) || (*pFormat == '\0'))
    {
        return;
    }

    pDriverObject = g_GlobalData.pDriverObject;
    pContext2 = (PVOID)pDriverObject->DeviceObject;

    // write max MAX_ERROR_STRING-1 characters leaving room for a terminating null
    _snprintf_s(String, sizeof(String), (MAX_ERROR_STRING - 1), pFormat, arglist);

    numChars = (UINT)strlen(String) + 1;

    // Make sure we always output an error message
    maxString = ERROR_LOG_MAXIMUM_SIZE - sizeof(IO_ERROR_LOG_PACKET) - 2;
    maxString >>= 1;

    if (numChars > maxString)
    {
        numChars = (UINT)strlen(TooBigString) + 1;
        RtlCopyBytes(String, TooBigString, numChars);
    }

    // New Size code -- Bound by Format and Type
    uMsgSize = (numChars * sizeof(WCHAR)) + sizeof(IO_ERROR_LOG_PACKET);
    if (uMsgSize < ERROR_LOG_MAXIMUM_SIZE)
    {
        pIoErrorLog = (PIO_ERROR_LOG_PACKET)IoAllocateErrorLogEntry(pContext2, (char)uMsgSize);


        // Fill in Message with all zero's
        if (NULL != pIoErrorLog)
        {
            RtlFillMemory(pIoErrorLog, uMsgSize, 0x0);

            pIoErrorLog->ErrorCode = errorCode;

            pIoErrorLog->NumberOfStrings = 1;

            pIoErrorLog->StringOffset = sizeof(IO_ERROR_LOG_PACKET) + pIoErrorLog->DumpDataSize;
            pStr = (PWSTR)((PUCHAR)pIoErrorLog + pIoErrorLog->StringOffset);
            pStr = AsciiToUnicode(pStr, String);
            IoWriteErrorLogEntry(pIoErrorLog);
        }
    }
}


POP_SEGMENTS