#pragma once

#include <lwme.h>

#define IOCTL_LWME_SIGNATURE_QUERY_QUEUE_INFO           "QUEUEINF"
#define IOCTL_LWME_SIGNATURE_CREATE_RESERVED_QUEUE_PAIR "CREATEQU"
#define IOCTL_LWME_SIGNATURE_DELETE_RESERVED_QUEUE_PAIR "DELETEQU"
#define IOCTL_LWME_SIGNATURE_FILTER_DRIVER_HANDSHAKE    "LWMETFHS"

// -- New custom commands

#define LW_LWME_IOCTL_BASE                              (0x0000008DE) // Vendor range (2048-4095)

#define LW_LWME_IOCTL_DRIVER_REQUEST                    CTL_CODE(LW_LWME_IOCTL_BASE, 0x002, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define LW_LWME_IOCTL_RETRIEVE_IOCTL                    CTL_CODE(LW_LWME_IOCTL_BASE, 0x003, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define LW_LWME_IOCTL_SUBMIT_READS                      CTL_CODE(LW_LWME_IOCTL_BASE, 0x004, METHOD_BUFFERED, FILE_ANY_ACCESS)

#define IOCTL_LWME_SIGNATURE_CREATE_MEMORY_MAPPING      "CRMEMMAP" // Map kernel allocation in user space... (Multiple process not handled ...)
#define IOCTL_LWME_SIGNATURE_LWME_DO_READ_BATCH         "DOREADBT" // Read command are in a separate allocation and VA is passed in command.
#define IOCTL_LWME_SIGNATURE_LWME_PIN_UNPIN_ALLOC       "PINUNPIN"
#define IOCTL_LWME_SIGNATURE_REGISTER_EVENT_WAIT        "EVNTWAIT"
#define IOCTL_LWME_SIGNATURE_LWME_WAIT_FOR_REF          "WAIT4REF"

#define LW_LWME_TAG                                     'LwMe'
#define LBA_MAPPING_LENGTH                               32
#define LW_MAX_SQS                                       4

// Avoiding include windows files here - This file is used by UMD and KMD both which defines them on different paths.
#ifndef MAX_PATH 
    #define MAX_PATH          260
#endif

// OUT
typedef struct _LBA_FILE_MAPPING
{
    ULONGLONG LBA;
    ULONGLONG length;
} LBA_FILE_MAPPING;

typedef struct LBAMAPPING_INFO {
    // IN
    CHAR                completFilePath[MAX_PATH];
    ULONGLONG           startVCNtoRetrieve;                 // The start VCN to retrieve. Set it as 0 if it's the first call  

    // OUT
    LBA_FILE_MAPPING    LBAMapping[LBA_MAPPING_LENGTH];     // We have LBA_MAPPING_LENGTH fragment limitaion on single IO control call, set another run if the file has more than LBA_MAPPING_LENGTH fragments. 
    ULONGLONG           retrivedLength;
    bool                bRequireExtraCall;                  // This flag is used to indicate if we need an extra IO control call 
    ULONGLONG           nextVCNtoRetrieve;                  // The next vcn to retrieve if file has more than LBA_MAPPING_LENGTH fragments 
} LBAMAPPING_INFO, * PLBAMAPPING_INFO;

typedef struct _LWME_READ
{
    ULONGLONG     inLBA;                    // Usermode provides LBA right now, this would change to file path once KMD implements it.
    ULONGLONG     inFileOffestInBytes;      // File Offset (within current LBA for now) to start read at
    ULONGLONG     inReadSize;               // [LBA + inFileOffestInBytes, LBA + inFileOffestInBytes + inReadSize) read into [inUserAllocBaseVA + inUserAllocOffset, inUserAllocBaseVA + inUserAllocOffset + inReadSize)
                                            // Must be in multiple of sectors (otherwise rounded upper which reads some extra.)
    struct
    {
        BOOLEAN       IsUserVA;             // false - kernel VA + allocation is pinned, true - User VA (it would be pinned /unpinned during the submit call), 
        PVOID         inAllocBaseVA;        // Read destination Usermode allocation base VA, (Require MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE )
        ULONGLONG     inAllocSize;          // Alloc size, used in pinning/unpinning mdl
        ULONGLONG     inAllocOffset;        // destination offset to start write at.
    } addr;

    ULONG         inDebugPerCmdReadSize;    // Per LWMe CMD max read size. 
    ULONG         outStatus;                // COMPLETED = 0, PENDING = 1, FAILED =2
    ULONGLONG     outSubmissionRefCount;    // Submitted refCount.
} LWME_READ, *PLWME_READ;

typedef struct _LWME_READ_COMAMND_ENTRY
{
    ULONGLONG                   LBA;                      // LBA this would change to file path once KMD implements it.
    ULONGLONG                   ReadSize;
    BOOLEAN                     IsAllolwserVA;            // false - kernel VA + allocation is pinned, true - User VA (it would be locked /unlocked during the submit call), 
    ULONGLONG                   AllocVA;                  // Allocation VA
    LWME_LWM_QUEUE_PRIORITIES   QueuePriority;            // Queue priority
} LWME_READ_COMAMND_ENTRY, *PLWME_READ_COMAMND_ENTRY;

typedef struct _LWME_READ_COMMAND_BUFFER
{
    PVOID         inKernelVA;                           // Allocation of Read entries - sizeof(LWME_READ_COMAMND_ENTRY) * inEntryCount will be processed
    ULONG         inEntryCount;                         // 
    ULONG         inDebugPerCmdReadSize;                // Per LWMe CMD max read size to override from UMD.
    ULONGLONG     outSubmissionRefCount[LW_MAX_SQS];    // Submitted refCount of all 4 submission queue if anything submited on that queue otherwise 0.
} LWME_READ_COMMAND_BUFFER, *PLWME_READ_COMMAND_BUFFER;

typedef struct _LWME_READ_COMMAND_BUFFER2
{
    PVOID         inKernelVA;                           // Allocation of Read entries - sizeof(LWME_READ_COMAMND_ENTRY) * inEntryCount will be processed
    ULONG         inEntryCount;                         // 
    ULONG         inDebugPerCmdReadSize;                // Per LWMe CMD max read size to override from UMD.
    ULONGLONG     outSubmissionRefCount[LW_MAX_SQS];    // Submitted refCount of all 4 submission queue if anything submited on that queue otherwise 0.
    HANDLE        inUserEvents[LW_MAX_SQS];             // Non null event handles will be added to submission tracking and singled when reached.
} LWME_READ_COMMAND_BUFFER2, * PLWME_READ_COMMAND_BUFFER2;

typedef struct _LWME_REGISTER_EVENT_WAIT
{
    USHORT        inCount;                       // entries count
    struct
    {
        USHORT        inQueuePriority;           // There are 4 reserved submission queues with differnt priority
        HANDLE        inUserEvent;               // Usermode event handle which app wants to wait on
        ULONGLONG     inRefCount;                // Wait till refCount is reached on selected priority queue
    } eventData[ANYSIZE_ARRAY];
} LWME_REGISTER_EVENT_WAIT, * PLWME_REGISTER_EVENT_WAIT;

typedef struct _LWME_PIN_UNPIN_ALLOC
{
    BOOLEAN       IsPinAlloc;               // if 1, it will pin allocation. if 0, it will unpin alloc
    union
    {
        struct // pin data - if IsPinAlloc = true
        {
            PVOID         inAllocBaseVA;    // Usermode allocation base VA (Require MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
            ULONGLONG     inAllocSize;      // Alloc size, used in mapping mdl
            ULONGLONG     outKernelVA;      // KernelVA that app can use in read calls.
            ULONGLONG     outHandle;        // handle which can be later used to free lock / mdl
        } pinData;
        struct // Unpin data - if IsPinAlloc = false
        {
            ULONGLONG     inHandle;         // this is allocation of mdl to be freed
        } unpinData;
    } para;
} LWME_PIN_UNPIN_ALLOC, * PLWME_PIN_UNPIN_ALLOC;

typedef struct _LWME_PIN_UNPIN_ALLOC_BATCH
{
    ULONG                      requestCount;
    LWME_PIN_UNPIN_ALLOC       requests[ANYSIZE_ARRAY];
} LWME_PIN_UNPIN_ALLOC_BATCH, * PLWME_PIN_UNPIN_ALLOC_BATCH;

typedef struct _LWME_MAP_ALLOC
{
    ULONGLONG     SQCompletionRefCountVA[ANYSIZE_ARRAY]; // Output Usermode mapped RefCount VAs for tracking purpose.
} LWME_MAP_ALLOC, *PLWME_MAP_ALLOC;

// -- END

#define IOCTL_MINIPORT_SIGNATURE_STORDIAG    "STORDIAG"
//
//
// Initial set of values corresponds to well known system files in FSCTL, but we don't use internal OS GUIDs. In future
// versions of DSM we will integrate explicit attribute values (when standardized)
//
//

#pragma warning(push)
#pragma warning(disable:4214)   // bit fields other than int to disable this around the struct
#pragma warning(disable:4201)   // nameless struct/union

//
// LWME queue header data structure
//
typedef struct _LWME_RESERVED_QUEUE_HEADER {

    ULONG               Version;                // Version of header; by default 0. Added for version compatibility
    ULONG               Size;                   // Size of the information including header

} LWME_RESERVED_QUEUE_HEADER, * PLWME_RESERVED_QUEUE_HEADER;

//
// Details of Create request for reserved Submission queue, provided by caller
//
typedef struct _LWME_RESERVED_SQ_CREATE_REQUEST {

    ULONG64                 PhysicalAddress;    // SQ base physical Address
    USHORT                  QueuePriority;      // SQ priority
    USHORT                  QueueDepth;         // SQ depth
    struct {
        ULONG               PhysicalContiguous : 1; // Physically Contiguous, 
                                                    // if set PRP1 points to contiguous buffer
                                                    // if not set, PRP1 is PRP
        ULONG               Reserved : 31;
    };
} LWME_RESERVED_SQ_CREATE_REQUEST, * PLWME_RESERVED_SQ_CREATE_REQUEST;

//
// Details of Create request for reserved Completion queue, provided by caller
//
typedef struct _LWME_RESERVED_CQ_CREATE_REQUEST {

    ULONG64             PhysicalAddress;    // CQ base physical Address
    ULONG               InterruptVector;    // CQ Interrupt Vector, corresponds to MSI-X or MSI vector
    USHORT              QueueDepth;         // CQ depth
    struct {
        USHORT          InterruptEnabled : 1;   // Interrupt Enabled for the queue
        USHORT          PhysicalContiguous : 1; // Physically Contiguous, 
                                                // if set PRP1 points to contiguous buffer
                                                // if not set, PRP1 is PRP list pointer
        USHORT          Reserved : 14;
    };
} LWME_RESERVED_CQ_CREATE_REQUEST, * PLWME_RESERVED_CQ_CREATE_REQUEST;

//
// Request to create one completion queue and one or more submission queues
//
typedef struct _LWME_RESERVED_QUEUES_CREATE_REQUEST {

    LWME_RESERVED_QUEUE_HEADER          Header;

    ULONG                               ResponseDataBufferOffset;           // The offset is from the beginning of buffer. e.g. from beginning of SRB_IO_CONTROL. 
                                                                            // The value should be multiple of sizeof(PVOID); Value 0 means that there is no data buffer.
    ULONG                               ResponseDataBufferLength;           // Length of the buffer

    LWME_RESERVED_CQ_CREATE_REQUEST     CompletionQueue;                    // Completion queue information

    USHORT                              SubmissionQueueCount;               // Number of submission queues requested
    LWME_RESERVED_SQ_CREATE_REQUEST     SubmissionQueue[LW_MAX_SQS];        // Submission queue(s) information

} LWME_RESERVED_QUEUES_CREATE_REQUEST, * PLWME_RESERVED_QUEUES_CREATE_REQUEST;

//
// Details of reserved Submission queue, provided as create response and queue query
//
typedef struct _LWME_RESERVED_SQ_INFO {

    ULONG64                 PhysicalAddress;                                // SQ base physical address
    ULONG64                 DoorbellRegisterAddress;                        // SQ tail doorbell register address
    USHORT                  QueueID;                                        // SQ ID
    USHORT                  QueueDepth;                                     // SQ depth
    USHORT                  CompletionQueueID;                              // Completion queue identifier
    struct {
        USHORT              PhysicalContiguous : 1;                         // Physically Contiguous
        USHORT              Reserved : 15;
    };
    USHORT                  QueuePriority;                                  // SQ priority
    USHORT                  Reserved2;

} LWME_RESERVED_SQ_INFO, * PLWME_RESERVED_SQ_INFO;

//
// Details of reserved Completion queue, provided as create response and queue query
//
typedef struct _LWME_RESERVED_CQ_INFO {

    ULONG64                 PhysicalAddress;                                // CQ base physical Address
    ULONG64                 DoorbellRegisterAddress;                        // CQ head doorbell register
    USHORT                  QueueID;                                        // CQ ID
    USHORT                  QueueDepth;                                     // CQ depth
    ULONG                   InterruptVector;                                // Interrupt Vector, corresponds to MSI-X or MSI vector
    struct {
        USHORT              InterruptEnabled : 1;                           // Interrupt Enabled
        USHORT              PhysicalContiguous : 1;                         // Physically Contiguous
        USHORT              Reserved : 14;
    };
    USHORT                  SubmissionQueueCount;                           // Mapped submission queue count
} LWME_RESERVED_CQ_INFO, * PLWME_RESERVED_CQ_INFO;

//
// Response to create request of one completion queue and one or more submission queues
//
typedef struct _LWME_RESERVED_QUEUES_CREATE_RESPONSE {

    LWME_RESERVED_QUEUE_HEADER          Header;
    LWME_RESERVED_CQ_INFO               CompletionQueue;                // Completion queue information

    LWME_RESERVED_SQ_INFO               SubmissionQueue[LW_MAX_SQS];    // Submission queue(s) information

} LWME_RESERVED_QUEUES_CREATE_RESPONSE, * PLWME_RESERVED_QUEUES_CREATE_RESPONSE;

//
// Request to delete queue pair (completion queue and corresponding submission queue)
//
typedef struct _LWME_RESERVED_QUEUES_DELETE_REQUEST {

    LWME_RESERVED_QUEUE_HEADER          Header;

    USHORT                              QueueID;                        // Completion queue ID 
                                                                        // (this would map to corresponding submission queues)

} LWME_RESERVED_QUEUES_DELETE_REQUEST, * PLWME_RESERVED_QUEUES_DELETE_REQUEST;

//
// Reserved queue property mapping
//
typedef struct _LWME_RESERVED_QUEUES_MAPPING {
    LWME_RESERVED_CQ_INFO       CompletionQueue;                    // Completion queue information

    LWME_RESERVED_SQ_INFO       SubmissionQueue[ANYSIZE_ARRAY];     // Submission queue(s) information

} LWME_RESERVED_QUEUES_MAPPING, * PLWME_RESERVED_QUEUES_MAPPING;

//
// Out parameters for IOCTL_SCSI_MINIPORT where the IoControlCode is IOCTL_QUERY_STORAGE_PROPERTY and signature IOCTL_MINIPORT_SIGNATURE_QUERY_QUEUE_INFO
// Reserved Queue properties returned on queue information query
//
typedef struct _LWME_RESERVED_QUEUES_PROPERTIES {

    LWME_RESERVED_QUEUE_HEADER      Header;

    USHORT                          QueuePairCount;                     // Number of reserved queue pair in the controller

    LWME_RESERVED_QUEUES_MAPPING    QueueMapping[ANYSIZE_ARRAY];        // Size is determined by QueuePairCount

} LWME_RESERVED_QUEUES_PROPERTIES, * PLWME_RESERVED_QUEUES_PROPERTIES;

typedef struct _LWME_DRIVER_HANDSHAKE
{
    LWME_RESERVED_CQ_INFO           CompletionQueue;                    // Completion queue information
    LWME_RESERVED_SQ_INFO           SubmissionQueue[LW_MAX_SQS];        // Submission queue(s) information
    ULONGLONG                       SQCompletionRefCountVA[LW_MAX_SQS]; // MAP ALLOC
} LWME_DRIVER_HANDSHAKE, * PLWME_DRIVER_HANDSHAKE;

typedef struct _LWME_WAIT_FOR_REF
{
    ULONGLONG                   RefCount;               // Busy polling till RefCount is reached
    LWME_LWM_QUEUE_PRIORITIES   QueuePriority;          // Queue priority
    ULONGLONG                   TimeOutInMiliSeconds;   // if 0 it returns immediately after polling once.
} LWME_WAIT_FOR_REF, * PLWME_WAIT_FOR_REF;

#pragma warning(pop)


