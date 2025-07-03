/******************************************************************************
*
*   Module: dcgm_export_table.c
*
*   Description:
*       Provides export tables, allowing users to query for versionable
*       API functions by UUID.
*
******************************************************************************/

// This is the translation unit that defines all UUIDs.  Other translation
// units use extern declarations and link to these definitions.  Defining
// all the UUIDs here has the added benefit of allowing the compiler to
// optimize dcgmGetExportTable since it knows all the UUID values.
#define DCGM_INIT_UUID

// For each export table, you must do the following:
//     1. Include the header which defines the UUID and export table's structure
//     2. Declare the export table as an extern const
//     3. Define a macro for including this table in the initializer list (must
//        have the form:  #define ETBL_ENTRY_... { &LWCM_ETID_..., &g_etbl... },
//     4. Add your ETBL_ENTRY_ macro to the initializer list for g_etblMap.
// If you want to exclude an export table under certain preprocessor conditions,
// wrap the group of lines that do items 1-3 in the appropriate #if, and add an
// #else block to define your ETBL_ENTRY_ macro as whitespace.  This pattern
// prevents the need to maintain the same preprocessor logic in multiple places.

#include "dcgm_client_internal.h"
#include "dcgm_agent_internal.h"
#include "dcgm_module_fm_internal.h"
#include "dcgm_modules_internal.h"
#include "lwtypes.h"
#include <string.h>
#include <lwos_internal.h>
#include <lwos.h>


extern const etblDCGMClientInternal g_etblDCGMClientInternal;
#define ETBL_ENTRY_DCGMClientInternal \
    { &ETID_DCGMClientInternal, &g_etblDCGMClientInternal },

#if 0
extern const etblDCGMClientTestInternal g_etblDCGMClientTestInternal;
#define ETBL_ENTRY_DCGMClientTestInternal \
    { &ETID_DCGMClientTestInternal, &g_etblDCGMClientTestInternal },
#endif

extern const etblDCGMEngineInternal g_etblDCGMEngineInternal;
#define ETBL_ENTRY_DCGMEngineInternal \
    { &ETID_DCGMEngineInternal, &g_etblDCGMEngineInternal },

extern const etblDCGMEngineTestInternal g_etblDCGMEngineTestInternal;
#define ETBL_ENTRY_DCGMEngineTestInternal \
    { &ETID_DCGMEngineTestInternal, &g_etblDCGMEngineTestInternal },

#ifdef DCGM_BUILD_LWSWITCH_MODULE
extern const etblDCGMLwSwitchInternal g_etblDCGMLwSwitchInternal;
#define ETBL_ENTRY_DCGMLwSwitchInternal \
    { &ETID_DCGMLwSwitchInternal, &g_etblDCGMLwSwitchInternal },
#endif

#ifdef DCGM_BUILD_VGPU_MODULE
extern const etblDCGMVgpuInternal g_etblDCGMVgpuInternal;
#define ETBL_ENTRY_DCGMVgpuInternal \
    { &ETID_DCGMVgpuInternal, &g_etblDCGMVgpuInternal },
#endif

extern const etblDCGMModuleFMInternal g_etblDCGMModuleFMInternal;
#define ETBL_ENTRY_DCGMModuleFMInternal \
    { &ETID_DCGMModuleFMInternal, &g_etblDCGMModuleFMInternal },


typedef struct
{
    const dcgmUuid_t *pUuid;
    const void *pTable;
} dcgmEtblMapEntry;

// This structure maps export table UUIDs ("ETIDs") to export table pointers.
// If you want an export table to be provided by dcgmGetExportTable, it must be
// in this list.
//
// Please do not use #if blocks here!  Instead, define ETBL_ENTRY_ macros
// above within the same preprocessor conditions for #includes and extern
// declarations.  When excluding a table with preprocessor conditions, be
// sure to define the ETBL_ENTRY_ macro as empty.  This pattern prevents
// the need to maintain the same preprocessor logic in multiple places.
static const dcgmEtblMapEntry g_etblMap[] =
{
    ETBL_ENTRY_DCGMClientInternal
    //ETBL_ENTRY_DCGMClientTestInternal
    ETBL_ENTRY_DCGMEngineInternal
    ETBL_ENTRY_DCGMEngineTestInternal
#ifdef DCGM_BUILD_LWSWITCH_MODULE
    ETBL_ENTRY_DCGMLwSwitchInternal
#endif
#ifdef DCGM_BUILD_VGPU_MODULE
    ETBL_ENTRY_DCGMVgpuInternal
#endif

    ETBL_ENTRY_DCGMModuleFMInternal

    // Add null entry at the end of the table to guarantee last element
    // has no comma.  This makes conditional inclusion of tables simpler.
    { 0, 0 }
};


// Use an enum instead of a const int because it is guaranteed to be
// treated as a compile-time constant value (whereas const int is not).
// Subtract one since we don't need to include the null entry at the end.
enum { g_etblCount = (sizeof(g_etblMap) / sizeof(g_etblMap[0])) - 1 };

// memcmp, return 0 if equal, <0 if lhs < rhs, and >0 if lhs > rhs.
static LW_INLINE int compareUuid(const dcgmUuid_t *pLhs, const dcgmUuid_t *pRhs)
{
    return memcmp(pLhs->bytes, pRhs->bytes, sizeof(dcgmUuid_t));
}

// This function allows you to query for structs of function pointers.  We call
// these "export tables" since they are to be used as entry points into the
// driver from other modules.  These tables are versionable; tables can become
// unsupported, in which case queries for them will return an error.  Tables
// are guaranteed not to change their ABI without changing the UUID as well.
// Individual tables may support additional versioning, such as the option to
// add more functions to the end of the table, and an initial size field that
// is set to the table size.
//
dcgmReturn_t DECLDIR dcgmInternalGetExportTable(const void **ppExportTable, const dcgmUuid_t *pExportTableId)
{
    LwU32 index = 0;

    if (!ppExportTable || !pExportTableId)
        return DCGM_ST_BADPARAM;
    
    *ppExportTable = 0;

    // Linear search.  This could be made faster by sorting the
    // table in a threadsafe init-once section, and using binary
    // search instead.
    for (index = 0; index < g_etblCount; ++index) 
    {
        const dcgmEtblMapEntry *pEntry = &g_etblMap[index];
        if (pEntry->pUuid) 
        {
            if (0 == compareUuid(pExportTableId, pEntry->pUuid)) 
            {
                *ppExportTable = pEntry->pTable;

                LW_ASSERT(NULL != *ppExportTable);

                return DCGM_ST_OK;
            }
        }
    }

    return DCGM_ST_BADPARAM;
}
