
#ifndef _NEWDEDEPENDENCE_H
#define _NEWDEDEPENDENCE_H



//TODO: remove these after no compiler errors to see if it did anything
#include <ctrl2080.h>
#include <ctrl0000.h>
#include <ctrla080.h>
#include <ctrl0080.h>
#include <ctrl5070.h>
#include <ctrl9096.h>
#include "lwtypes.h"
#include "g_lwconfig.h"

#include "windows.h"
//#include "winServicesIf.hpp"
//#include <baseObject.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <lwos.h>
#include "lwlConsts.h"


//////////////////////////////Win Services/////////////////////////////////
class __declspec(novtable) CWinServicesIf {
public:
//TODO: get this working as welll...
    unsigned long long getSystemMemorySize() const
    {
        return 0;
    }
//TODO: get this working and whatnot....
    bool isWin8_1_OrLater() {return true;}
    bool isWin8OrLater() {return true;}
    
    void* allocateMemoryWithTag(size_t size, int poolType,unsigned long tag)
    {
        if (size != 0)
            return malloc(size);
        else
            return NULL;
    }
    void freeMemoryWithTag(void* pMem, unsigned long tag)
    {
        free(pMem);
    }
};
//////////////////////////////End Win Services/////////////////////////////

//////////////////////////////Base Object/////////////////////////////////
#define PagedPool -1
#define NonPagedPoolNx -1
#define LW_MEMCONFIG_TAG -1

class CBaseObject
{
public:
    CBaseObject ( CWinServicesIf* pWinSvc ) 
     {
        m_WinServices = pWinSvc;
     }
    void*   operator new(size_t          size,
                         CWinServicesIf* pWinSvc,
                         int       allocType,
                         unsigned long   tag)
    {
        return malloc(size);
    }
    CWinServicesIf* getWinServices() const
    {
        return m_WinServices;
    }
private:
    CWinServicesIf* m_WinServices;
};
//////////////////////////////End Base Object/////////////////////////////////

//TODO: remove these after no compiler errors to see if it did anything
//#include "lwUniversal.h"
//#include "lwlPrivate.h"
typedef LARGE_INTEGER PHYSICAL_ADDRESS;


typedef struct _MDL {
    struct _MDL *Next;
    short Size;
    short MdlFlags;

    struct _EPROCESS *Process;
    void* MappedSystemVa;   /* see creators for field size annotations. */
    void* StartVa;   /* see creators for validity; could be address 0.  */
    unsigned long ByteCount;
    unsigned long ByteOffset;
} MDL, *PMDL;

////////////////////////////////From d3dkmddi.h///////////////////////////////////////////
//This is the WDDM2.0 version, although the test will most likely not build with wddm2.0 for other reasons
typedef struct _DXGK_SEGMENTFLAGS
{
    union
    {
        struct
        {
            UINT    Aperture                          : 1;    // 0x00000001
            UINT    Agp                               : 1;    // 0x00000002
            UINT    CpuVisible                        : 1;    // 0x00000004
            UINT    UseBanking                        : 1;    // 0x00000008
            UINT    CacheCoherent                     : 1;    // 0x00000010
            UINT    PitchAlignment                    : 1;    // 0x00000020
            UINT    PopulatedFromSystemMemory         : 1;    // 0x00000040
            UINT    PreservedDuringStandby            : 1;    // 0x00000080
            UINT    PreservedDuringHibernate          : 1;    // 0x00000100
            UINT    PartiallyPreservedDuringHibernate : 1;    // 0x00000200
#if (DXGKDDI_INTERFACE_VERSION >= DXGKDDI_INTERFACE_VERSION_WIN8)
            UINT    DirectFlip                        : 1;    // 0x00000400
#if (DXGKDDI_INTERFACE_VERSION >= DXGKDDI_INTERFACE_VERSION_WDDM2_0)
            UINT    Use64KBPages                      : 1;    // 0x00000800         // Defines if the segment is using 4GB or 64 KB pages 
            UINT    ReservedSysMem                    : 1;    // 0x00001000         // Reserved for system use
            UINT    SupportsCpuHostAperture           : 1;    // 0x00002000         // True if segment supports a CpuHostAperture
            UINT    Reserved                          :18;    // 0xFFFFC000
#else
            UINT    Reserved                          :21;    // 0xFFFFF800
#endif // (DXGKDDI_INTERFACE_VERSION >= DXGKDDI_INTERFACE_VERSION_WDDM2_0)
#else
            UINT    Reserved                          :22;    // 0xFFFFFC00
#endif
        };
        UINT        Value;
    };
} DXGK_SEGMENTFLAGS;

typedef struct _DXGK_SEGMENTDESCRIPTOR
{
    PHYSICAL_ADDRESS        BaseAddress;            // GPU logical base address for
                                                    // the segment.
    PHYSICAL_ADDRESS        CpuTranslatedAddress;   // CPU translated base address
                                                    // for the segment if CPU visible.
    SIZE_T                  Size;                   // Size of the segment.
    UINT                    NbOfBanks;              // Number of bank in the segment.
    SIZE_T*                 pBankRangeTable;        // Range delimiting each bank.
    SIZE_T                  CommitLimit;            // Maximum number of bytes that can be
                                                    // commited to this segment, apply to
                                                    // aperture segment only.
    DXGK_SEGMENTFLAGS       Flags;                  // Segment bit field flags
} DXGK_SEGMENTDESCRIPTOR;

typedef struct _DXGK_SEGMENTDESCRIPTOR3
{
    DXGK_SEGMENTFLAGS       Flags;                  // Segment bit field flags
    PHYSICAL_ADDRESS        BaseAddress;            // GPU logical base address for
                                                    // the segment.
    PHYSICAL_ADDRESS        CpuTranslatedAddress;   // CPU translated base address
                                                    // for the segment if CPU visible.
    SIZE_T                  Size;                   // Size of the segment.
    UINT                    NbOfBanks;              // Number of bank in the segment.
    SIZE_T*                 pBankRangeTable;        // Range delimiting each bank.
    SIZE_T                  CommitLimit;            // Maximum number of bytes that can be
                                                    // commited to this segment, apply to
                                                    // aperture segment only.
    SIZE_T                  SystemMemoryEndAddress; // For segments that are partially composed
                                                    // of system memory, all allocations ending after
                                                    // this address are purged during hibernate.
    SIZE_T                  Reserved;
} DXGK_SEGMENTDESCRIPTOR3;

typedef struct _DXGK_QUERYSEGMENTOUT
{
    UINT                    NbSegment;              // Number of segment described.
    DXGK_SEGMENTDESCRIPTOR* pSegmentDescriptor;     // Buffer describing the segment.
    UINT                    PagingBufferSegmentId;  // SegmentId the paging buffer
                                                    // should be allocated from.
    UINT                    PagingBufferSize;       // Paging buffer size.
    UINT                    PagingBufferPrivateDataSize;
} DXGK_QUERYSEGMENTOUT;

typedef struct _DXGK_QUERYSEGMENTOUT3
{
    UINT                     NbSegment;              // Number of segment described.
    DXGK_SEGMENTDESCRIPTOR3* pSegmentDescriptor;     // Buffer describing the segment.
    UINT                     PagingBufferSegmentId;  // SegmentId the paging buffer
                                                     // should be allocated from.
    UINT                     PagingBufferSize;       // Paging buffer size.
    UINT                     PagingBufferPrivateDataSize;
} DXGK_QUERYSEGMENTOUT3;

typedef struct _DXGK_QUERYSEGMENTOUT4
{
    UINT    NbSegment;                      // Number of segment described.
    BYTE*   pSegmentDescriptor;             // An array of segment descriptors, where each element
                                            // is of 'SegmentDescriptorStride' in size.
    UINT    PagingBufferSegmentId;          // SegmentId the paging buffer
                                            // should be allocated from.
    UINT    PagingBufferSize;               // Paging buffer size.
    UINT    PagingBufferPrivateDataSize;
    SIZE_T  SegmentDescriptorStride;        // Size of each element in the 
                                            // pSegmentDescriptor array
} DXGK_QUERYSEGMENTOUT4;
////////////////////////////////End of d3dkmddi.h///////////////////////////////////////////

typedef struct _LW_REG_KEYDWORD
{
    PWSTR  lpwRegName;
    void  *lpValue;
    ULONG  size;
} LW_REG_KEYVALUE;


#define FUNCTION_PROLOG();
#define CHECK_IRQL(PASSIVE_LEVEL);
#define PAGE_SIZE 4096

#ifndef BIG_GPU
#define BIG_GPU     (!LWCFG(GLOBAL_FEATURE_SOC))
#endif


//////////////////////////Useful stuff//////////////////////////////////////////////

#define DPF_LEVEL(a,...) printf("WARNING ISSUED \n FILE: %s \n FUNCTION: %s \n LINE: %d\n",__FILE__,__FUNCTION__,__LINE__);
#define lwAssert(s)  (s) ? (void) 0 : printf("ASSERTION FAILED \n FILE: %s \n FUNCTION: %s \n LINE: %d\n",__FILE__,__FUNCTION__,__LINE__);

////////////////////////////////////////////////////////////////////////////////////

//TODO: Simplify alignment stuff; learn how alignment stuff works...
#define isAligned(anything,something) 1

inline BYTE*    lwAlign(BYTE* pAddress, ULONG ulAlignment)
    { return reinterpret_cast<BYTE*>(((reinterpret_cast<SIZE_T>(pAddress) + ulAlignment - 1) / ulAlignment) * ulAlignment); }
inline size_t   lwAlign(size_t offset, ULONG ulAlignment)
    { return (((offset + ulAlignment - 1) / ulAlignment) * ulAlignment); }


inline LwU64    lwAlign64(LwU64 qwOffset, ULONG ulAlignment)
    { return (((qwOffset + ulAlignment - 1) / ulAlignment) * ulAlignment); }

#define GPU_PAGE_SIZE_128K 128*1024


#endif _NEWDEDEPENDENCE_H