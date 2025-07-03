/* 
 * File:   dcgm_client_internal.h
 *
 */

#ifndef DCGM_CLIENT_INTERNAL_H
#define	DCGM_CLIENT_INTERNAL_H

#ifdef	__cplusplus
extern "C" {
#endif
    
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"
#include "dcgm_uuid.h"

/*****************************************************************************
 *****************************************************************************/
/*****************************************************************************
 * DCGM Client Internal functions to be used by DCGMI or other LW tools 
 *****************************************************************************/
/*****************************************************************************
 *****************************************************************************/

// GUIDS for internal APIs
DCGM_DEFINE_UUID(ETID_DCGMClientInternal,
                 0x2c9eabc4, 0x4dc3, 0x2f5d, 0xb7, 0x45, 0xbb, 0x71, 0x9f, 0x26, 0xcf, 0xb5);

typedef struct etblDCGMClientInternal_st {
    /// This export table supports versioning by adding to the end without changing
    /// the ETID.  The struct_size field will always be set to the size in bytes of
    /// the entire export table structure.
    size_t struct_size;
   
    // 1
    /**
     * This method is used to save the LwcmCacheManager's stats to a file local to the host engine
     *
     * @param pLwcmHandle
     * @param filename
     * @param fileType
     * @return
     */
    dcgmReturn_t(*fpClientSaveCacheManagerStats)(dcgmHandle_t pLwcmHandle, const char *filename, dcgmStatsFileType_t fileType);

    // 2
    /**
     * This method is used to load the LwcmCacheManager's stats from a file local to the host engine
     *
     * @param pLwcmHandle
     * @param filename
     * @param fileType
     * @return
     */
    dcgmReturn_t(*fpClientLoadCacheManagerStats)(dcgmHandle_t pLwcmHandle, const char *filename, dcgmStatsFileType_t fileType);
} etblDCGMClientInternal;


#ifdef	__cplusplus
}
#endif

#endif	/* DCGM_CLIENT_INTERNAL_H */
