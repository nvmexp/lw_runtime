/* 
 * File:   dcgm_structs_internal.h
 */

#ifndef DCGM_STRUCTS_INTERNAL_H
#define DCGM_STRUCTS_INTERNAL_H

/* Make sure that dcgm_structs.h is loaded first. This file depends on it */
#include "dcgm_structs.h"
#include "dcgm_agent.h"
#include "lwml.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Version string for DCGM. Use this value when printing out any DCGM version
 */
#define DCGM_VERSION_STRING "1.7.4"

/*
 * Copied from LWML_CASSERT() in apps/lwml/lwml_internal.h
 *
 * The following is a compile time assertion.  It makes use of the
 * restriction that you cannot have an array with a negative size.
 * If the expression resolves to 0, then the index to the array is
 * defined as -1, and a compile time error is generated.  Note that
 * all three macros are needed because of the way the preprocessor
 * evaluates the directives.  Also note that the line number is
 * embedded in the name of the array so that the array name is unique
 * and we can have multiple calls to the assert with the same msg.
 *
 * Usage would be like this:
 * DCGM_CASSERT(DCGM_VGPU_NAME_BUFFER_SIZE == LWML_VGPU_NAME_BUFFER_SIZE, DCGM_VGPU_NAME_BUFFER_SIZE);
 *
 */
#define _DCGM_CASSERT_SYMBOL_INNER(line, msg) COMPILE_TIME_ASSERT_DETECTED_AT_LINE_ ## line ## __ ## msg
#define _DCGM_CASSERT_SYMBOL(line, msg) _DCGM_CASSERT_SYMBOL_INNER(line, msg)
#define DCGM_CASSERT(expression, msg) typedef char _DCGM_CASSERT_SYMBOL(__LINE__, msg) [((expression) ? 1 : -1)]


/**
 * Max length of the DCGM string field
 */
#define DCGM_MAX_STR_LENGTH     256    
    
typedef struct 
{
    unsigned int gpuId;         /* DCGM GPU ID */
    char uuid[DCGM_MAX_STR_LENGTH];     /* UUID String */
} dcgmGpuInfo_t;

/* Below is a test API simply to make sure versioning is working correctly
 */

typedef struct
{
    // version must always be first
    unsigned int version;

    unsigned int a;
} dcgmVersionTest_v1;

typedef struct
{
    // version must always be first
    unsigned int version;

    unsigned int a;
    unsigned int b;
} dcgmVersionTest_v2;

typedef dcgmVersionTest_v2 dcgmVersionTest_t;
#define dcgmVersionTest_version1 MAKE_DCGM_VERSION(dcgmVersionTest_v1,1)
#define dcgmVersionTest_version2 MAKE_DCGM_VERSION(dcgmVersionTest_v2,2)
#define dcgmVersionTest_version3 MAKE_DCGM_VERSION(dcgmVersionTest_v2,3)
#define dcgmVersionTest_version dcgmVersionTest_version2

/**
 * Represents a command to save or load a JSON file to/from the LwcmCacheManager
 */

typedef enum dcgmStatsFileType_enum
{
    DCGM_STATS_FILE_TYPE_JSON = 0  /* JSON */
} dcgmStatsFileType_t;

typedef struct
{
    // version must always be first
    unsigned int version;

    dcgmStatsFileType_t fileType;   /* File type to save to/load from */
    char filename[256];             /* Filename to save to/load from */
} dcgmCacheManagerSave_v1_t;

#define dcgmCacheManagerSave_version1 MAKE_DCGM_VERSION(dcgmCacheManagerSave_v1_t, 1)
#define dcgmCacheManagerSave_version dcgmCacheManagerSave_version1

typedef dcgmCacheManagerSave_v1_t dcgmCacheManagerSave_t;

/* Same message contents for now */
typedef dcgmCacheManagerSave_v1_t dcgmCacheManagerLoad_v1_t;

typedef dcgmCacheManagerLoad_v1_t dcgmCacheManagerLoad_t;

#define dcgmCacheManagerLoad_version1 MAKE_DCGM_VERSION(dcgmCacheManagerLoad_v1_t, 1)
#define dcgmCacheManagerLoad_version dcgmCacheManagerLoad_version1

/**
 * This structure is used to represent a field value to be injected into
 * the cache manager
 */
typedef dcgmFieldValue_v1 dcgmInjectFieldValue_v1;
typedef dcgmInjectFieldValue_v1 dcgmInjectFieldValue_t;
#define dcgmInjectFieldValue_version1 MAKE_DCGM_VERSION(dcgmInjectFieldValue_v1, 1)
#define dcgmInjectFieldValue_version dcgmInjectFieldValue_version1

#define dcgmWatchFieldValue_version1 1
#define dcgmWatchFieldValue_version dcgmWatchFieldValue_version1

#define dcgmUnwatchFieldValue_version1 1
#define dcgmUnwatchFieldValue_version dcgmUnwatchFieldValue_version1

#define dcgmUpdateAllFields_version1 1
#define dcgmUpdateAllFields_version dcgmUpdateAllFields_version1

#define dcgmGetMultipleValuesForField_version1 1
#define dcgmGetMultipleValuesForField_version dcgmGetMultipleValuesForField_version1

/* Underlying structure for the GET_MULTIPLE_LATEST_VALUES request */
typedef struct
{
    unsigned int version;                    /* Set this to dcgmGetMultipleLatestValues_version1 */
    dcgmGpuGrp_t groupId;                    /* Entity group to retrieve values for. This is only
                                                looked at if entitiesCount is 0 */
    unsigned int entitiesCount;              /* Number of entities provided in entities[]. This
                                                should only be provided if you aren't also setting
                                                entityGroupId */
    dcgmGroupEntityPair_t entities[DCGM_GROUP_MAX_ENTITIES]; /* Entities to retrieve values for. 
                                                Only looked at if entitiesCount > 0 */
    dcgmFieldGrp_t fieldGroupId;             /* Field group to retrive values for. This is onlu looked
                                                at if fieldIdCount is 0 */
    unsigned int fieldIdCount;               /* Number of field IDs in fieldIds[] that are valid. This
                                                should only be set if fieldGroupId is not set */
    unsigned short fieldIds[DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP]; /* Field IDs for which values should
                                                be retrieved. only looked at if fieldIdCount is > 0 */
    unsigned int flags;                      /* Mask of DCGM_FV_FLAG_? #defines that affect this
                                                request */

} dcgmGetMultipleLatestValues_v1, dcgmGetMultipleLatestValues_t;

#define dcgmGetMultipleLatestValues_version1 MAKE_DCGM_VERSION(dcgmGetMultipleLatestValues_v1, 1)
#define dcgmGetMultipleLatestValues_version dcgmGetMultipleLatestValues_version1

/* Represents cached record metadata */

/* Represents a unique watcher of an entity in DCGM */

/* Watcher types. Each watcher type's watches are tracked separately within subsystems */
typedef enum
{ 
    DcgmWatcherTypeClient         = 0, /* Embedded or remote client via external APIs */
    DcgmWatcherTypeHostEngine     = 1, /* Watcher is LwcmHostEngineHandler */
    DcgmWatcherTypeHealthWatch    = 2, /* Watcher is LwcmHealthWatch */
    DcgmWatcherTypePolicyManager  = 3, /* Watcher is LwcmPolicyMgr */
    DcgmWatcherTypeCacheManager   = 4, /* Watcher is DcgmCacheManager */
    DcgmWatcherTypeConfigManager  = 5, /* Watcher is LwcmConfigMgr */
    DcgmWatcherTypeFabricManager  = 6, /* Watcher is FabricManager */

    DcgmWatcherTypeCount               /* Should always be last */
} DcgmWatcherType_t;


/* ID of a remote client connection within the host engine */
typedef unsigned int dcgm_connection_id_t;

/* Special constant for not connected */
#define DCGM_CONNECTION_ID_NONE ((dcgm_connection_id_t)0)

/* Cache Manager Info flags */
#define DCGM_CMI_F_WATCHED 0x00000001 /* Is this field being watched? */

/* This structure mirrors the DcgmWatcher object */
typedef struct dcgm_cm_field_info_watcher_t
{
    DcgmWatcherType_t watcherType;      /* Type of watcher. See DcgmWatcherType_t */
    dcgm_connection_id_t connectionId;  /* Connection ID of the watcher */
    long long monitorFrequencyUsec;     /* How often this field should be sampled */
    long long maxAgeUsec;               /* Maximum time to cache samples of this
                                           field. If 0, the class default is used */
} dcgm_cm_field_info_watcher_t, *dcgm_cm_field_info_watcher_p;

#define DCGM_CM_FIELD_INFO_NUM_WATCHERS 10 //Number of watchers to show for each field

typedef struct dcgmCacheManagerFieldInfo_v3_t
{
    unsigned int version;      /* Version. Check against dcgmCacheManagerInfo_version */
    unsigned int flags;        /* Bitmask of DCGM_CMI_F_? #defines that apply to this field */
    unsigned int gpuId;        /* ID of the GPU for this field */
    unsigned short fieldId;    /* Field ID of this field */
    short lastStatus;          /* Last lwml status returned for this field when taking a sample */
    long long oldestTimestamp; /* Timestamp of the oldest record. 0=no records or single
                                  non-time series record */
    long long newestTimestamp; /* Timestamp of the newest record. 0=no records or
                                  single non-time series record */
    long long monitorFrequencyUsec; /* How often is this field updated in usec */
    long long maxAgeUsec;      /* How often is this field updated */
    long long execTimeUsec;    /* Cumulative time spent updating this
                                  field since the cache manager started */
    long long fetchCount;      /* Number of times that this field has been
                                  fetched from the driver */
    int numSamples;            /* Number of samples lwrrently cached for this field */
    int numWatchers;           /* Number of watchers that are valid in watchers[] */
    dcgm_cm_field_info_watcher_t watchers[DCGM_CM_FIELD_INFO_NUM_WATCHERS]; /* Who are the first 10 
                                                                               watchers of this field? */
} dcgmCacheManagerFieldInfo_v3_t, *dcgmCacheManagerFieldInfo_v3_p;

typedef dcgmCacheManagerFieldInfo_v3_t dcgmCacheManagerFieldInfo_t;
#define dcgmCacheManagerFieldInfo_version3 MAKE_DCGM_VERSION(dcgmCacheManagerFieldInfo_v3_t, 3)
#define dcgmCacheManagerFieldInfo_version dcgmCacheManagerFieldInfo_version3

#define dcgmWatchFields_version1 1
#define dcgmWatchFields_version dcgmWatchFields_version1

/**
 * Sync boost groups
 */
#define DCGM_SYNC_BOOST_MAX_GROUPS 16 
#define DCGM_SYNC_BOOST_MAX_PER_GROUP 8

typedef struct dcgmSyncBoostGroupListItem_t
{
    unsigned int rmGroupId;                                  //!< RM-provided ID of this group
    int numDevices;                                          //!< Number of populated entry in devices[]
    unsigned int lwmlIndex[DCGM_SYNC_BOOST_MAX_PER_GROUP];   //!< LWML Index for the group member (Better to change it to UUID later)
} dcgmSyncBoostGroupListItem_t;

typedef struct dcgmSyncBoostGroupList_t
{
    unsigned int version;   //!< Version. Check against dcgmSyncBoostGroupList_version
    int numGroups;          //!< Number of populated entries in groups[]
    int unused;             //!< Align to 8 bytes
    dcgmSyncBoostGroupListItem_t syncBoostGroups[DCGM_SYNC_BOOST_MAX_GROUPS];
} dcgmSyncBoostGroupList_v1_t;
typedef dcgmSyncBoostGroupList_v1_t dcgmSyncBoostGroupList_v_t;
#define dcgmSyncBoostGroupList_version1 MAKE_DCGM_VERSION(dcgmSyncBoostGroupList_v1_t, 1)
#define dcgmSyncBoostGroupList_version dcgmSyncBoostGroupList_version1

/**
 * The maximum number of topology elements possible given DCGM_MAX_NUM_DEVICES
 * callwlated using arithmetic sequence formula
 * (DCGM_MAX_NUM_DEVICES - 1) * (1 + (DCGM_MAX_NUM_DEVICES-2)/2)
 */
#define DCGM_TOPOLOGY_MAX_ELEMENTS 120

/**
 * Topology element structure
 */
typedef struct
{
    unsigned int dcgmGpuA;                                      //!< GPU A
    unsigned int dcgmGpuB;                                      //!< GPU B
    unsigned int AtoBLwLinkIds;                                 //!< bits representing the links connected from GPU A to GPU B
                                                                //!< e.g. if this field == 3, links 0 and 1 are connected, 
                                                                //!< field is only valid if LWLINKS actually exist between GPUs
    unsigned int BtoALwLinkIds;                                 //!< bits representing the links connected from GPU B to GPU A
                                                                //!< e.g. if this field == 3, links 0 and 1 are connected, 
                                                                //!< field is only valid if LWLINKS actually exist between GPUs
    dcgmGpuTopologyLevel_t path;                                //!< path between A and B
} dcgmTopologyElement_t;

/**
 * Topology results structure
 */
typedef struct
{
    unsigned int version;                                       //!< version number (dcgmTopology_version)
    unsigned int numElements;                                   //!< number of valid dcgmTopologyElement_t elements

    dcgmTopologyElement_t element[DCGM_TOPOLOGY_MAX_ELEMENTS];
} dcgmTopology_v1;

/**
 * Typedef for \ref dcgmTopology_v1
 */
typedef dcgmTopology_v1 dcgmTopology_t;

/**
 * Version 1 for \ref dcgmTopology_v1
 */
#define dcgmTopology_version1 MAKE_DCGM_VERSION(dcgmTopology_v1, 1)

/**
 * Latest version for \ref dcgmTopology_t
 */
#define dcgmTopology_version dcgmTopology_version1

typedef struct
{
    unsigned int numGpus;
    struct {
        unsigned int dcgmGpuId;
        unsigned long bitmask[DCGM_AFFINITY_BITMASK_ARRAY_SIZE];
    } affinityMasks[DCGM_MAX_NUM_DEVICES];
} dcgmAffinity_t;


typedef struct
{
    unsigned int version;                     //!<IN: Version number (dcgmCreateFakeEntities_version)
    unsigned int entityGroupId;               //!<IN: Entity group to create the fake entities in
    unsigned int numToCreate;                 //!<IN: Number of fake entities to create
    unsigned int unused;                      //! Not used. Here for structure alignment
    unsigned int entityId[DCGM_MAX_NUM_DEVICES]; //!<OUT: Array of entityIds of fake entities that were created
} dcgmCreateFakeEntities_v1;

typedef dcgmCreateFakeEntities_v1 dcgmCreateFakeEntities_t;

/**
 * Version 1 for \ref dcgmCreateFakeEntities_t
 */
#define dcgmCreateFakeEntities_version1 MAKE_DCGM_VERSION(dcgmCreateFakeEntities_v1, 1)

/**
 * Latest version for \ref dcgmCreateFakeEntities_t
 */
#define dcgmCreateFakeEntities_version dcgmCreateFakeEntities_version1


/* Field watch predefined groups */
typedef enum
{
    DCGM_WATCH_PREDEF_ILWALID = 0,
    DCGM_WATCH_PREDEF_PID,  /* PID stats */
    DCGM_WATCH_PREDEF_JOB,  /* Job stats */
} dcgmWatchPredefinedType_t;

typedef struct
{
    unsigned int version;
    dcgmWatchPredefinedType_t watchPredefType; /* Which type of predefined watch are we adding? */

    dcgmGpuGrp_t groupId;  /* GPU group to watch fields for */
    long long updateFreq;  /* How often to update the fields in usec */
    double maxKeepAge;     /* How long to keep values for the fields in seconds */
    int maxKeepSamples;    /* Maximum number of samples we should keep at a time */
} dcgmWatchPredefined_v1;

typedef dcgmWatchPredefined_v1 dcgmWatchPredefined_t;

/**
 * Version 1 for \ref dcgmWatchPredefined_t
 */
#define dcgmWatchPredefined_version1 MAKE_DCGM_VERSION(dcgmWatchPredefined_v1, 1)

/**
 * Latest version for \ref dcgmWatchPredefined_t
 */
#define dcgmWatchPredefined_version dcgmWatchPredefined_version1

/**
 * Request to set a LwLink link state for an entity
 */
typedef struct 
{
    unsigned int version;                    /* Version. Should be dcgmSetLwLinkLinkState_version1 */
    dcgm_field_entity_group_t entityGroupId; /* Entity group of the entity to set the link state of */
    dcgm_field_eid_t entityId;               /* ID of the entity to set the link state of */
    unsigned int linkId;                     /* Link (or portId) of the link to set the state of */
    dcgmLwLinkLinkState_t linkState;         /* State to set the link to */
    unsigned int unused;                     /* Not used for now. Set to 0 */ 
} dcgmSetLwLinkLinkState_v1;

#define dcgmSetLwLinkLinkState_version1 MAKE_DCGM_VERSION(dcgmSetLwLinkLinkState_v1, 1)


/**
 * Request to blacklist a given module ID
 */
typedef struct
{
    unsigned int version;            /* Version. Should be dcgmModuleBlacklist_version */
    dcgmModuleId_t moduleId;         /* Module to blacklist */
} dcgmModuleBlacklist_v1;

#define dcgmModuleBlacklist_version1 MAKE_DCGM_VERSION(dcgmModuleBlacklist_v1, 1)


/**
 * Counter to use for LwLink
 */
#define DCGMCM_LWLINK_COUNTER_BYTES 0

/**
 * The Brand of the GPU. These are 1:1 with LWML_BRAND_*. There's a DCGM_CASSERT() below that tests that
 */
typedef enum dcgmGpuBrandType_enum
{
    DCGM_GPU_BRAND_UNKNOWN              = 0, 
    DCGM_GPU_BRAND_QUADRO               = 1,
    DCGM_GPU_BRAND_TESLA                = 2,
    DCGM_GPU_BRAND_LWS                  = 3,
    DCGM_GPU_BRAND_GRID                 = 4,
    DCGM_GPU_BRAND_GEFORCE              = 5,
    DCGM_GPU_BRAND_TITAN                = 6,
/* vGPU specific product brand types - start */
    DCGM_GPU_BRAND_LWIDIA_VAPPS         = 7,
    DCGM_GPU_BRAND_LWIDIA_VPC           = 8,
    DCGM_GPU_BRAND_LWIDIA_VCS           = 9,
    DCGM_GPU_BRAND_LWIDIA_VWS           = 10,
    DCGM_GPU_BRAND_LWIDIA_CLOUD_GAMING  = 11,
/* vGPU specific product brand types - end */
    DCGM_GPU_QUADRO_RTX                 = 12,
    DCGM_GPU_LWIDIA_RTX                 = 13,
    DCGM_GPU_LWIDIA                     = 14,
    DCGM_GPU_GEFORCE_RTX                = 15,
    DCGM_GPU_TITAN_RTX                  = 16,

    // Keep this last
    DCGM_GPU_BRAND_COUNT
} dcgmGpuBrandType_t;

/**
 * Verify that DCGM definitions that are copies of LWML ones match up with their LWML counterparts
 */
DCGM_CASSERT(DCGM_VGPU_NAME_BUFFER_SIZE == LWML_VGPU_NAME_BUFFER_SIZE, LWML_VGPU_NAME_BUFFER_SIZE);
DCGM_CASSERT(DCGM_GRID_LICENSE_BUFFER_SIZE == LWML_GRID_LICENSE_BUFFER_SIZE, LWML_GRID_LICENSE_BUFFER_SIZE);
DCGM_CASSERT(DCGM_DEVICE_UUID_BUFFER_SIZE == LWML_DEVICE_UUID_BUFFER_SIZE, LWML_DEVICE_UUID_BUFFER_SIZE);
DCGM_CASSERT(DCGM_LWLINK_MAX_LINKS_PER_GPU == LWML_LWLINK_MAX_LINKS, LWML_LWLINK_MAX_LINKS);
DCGM_CASSERT((int)DCGM_GPU_BRAND_COUNT == (int)LWML_BRAND_COUNT, LWML_BRAND_COUNT);

/**
 *  Verify correct version of APIs that use a versioned structure
 */

DCGM_CASSERT(dcgmPidInfo_version == (long)16786344, 1);
DCGM_CASSERT(dcgmConfig_version == (long)16777256, 1);
DCGM_CASSERT(dcgmConnectV2Params_version1 == (long)16777224, 1);
DCGM_CASSERT(dcgmConnectV2Params_version == (long)0x02000010, 1);
DCGM_CASSERT(dcgmFieldGroupInfo_version == (long)16777744, 1);
DCGM_CASSERT(dcgmAllFieldGroup_version == (long)16811016, 1);
DCGM_CASSERT(dcgmDeviceAttributes_version == (long)16782628 , 1);
DCGM_CASSERT(dcgmHealthResponse_version1 == (long)16942540, 1);
DCGM_CASSERT(dcgmHealthResponse_version2 == (long)34348044, 1);
DCGM_CASSERT(dcgmHealthResponse_version3 == (long)53499916, 1);
DCGM_CASSERT(dcgmIntrospectContext_version == (long)16777232, 1);
DCGM_CASSERT(dcgmIntrospectMemory_version == (long)16777232, 1);
DCGM_CASSERT(dcgmIntrospectCpuUtil_version == (long)16777248, 1);
DCGM_CASSERT(dcgmIntrospectFieldsExecTime_version == (long)16777248, 1);
DCGM_CASSERT(dcgmIntrospectFullFieldsExecTime_version == (long)16777880, 1);
DCGM_CASSERT(dcgmJobInfo_version == (long)33574568, 1);
DCGM_CASSERT(dcgmPolicy_version == (long)16777360, 1);
DCGM_CASSERT(dcgmPolicyCallbackResponse_version == (long)16777240, 1);
DCGM_CASSERT(dcgmDiagResponse_version3 == (long)50562672, 1);
DCGM_CASSERT(dcgmDiagResponse_version4 == (long)67373708, 1);
DCGM_CASSERT(dcgmDiagResponse_version5 == (long)84151440, 1);
DCGM_CASSERT(dcgmDiagResponse_version == (long)84151440, 1);
DCGM_CASSERT(dcgmRunDiag_version1 == (long)16788456, 1);
DCGM_CASSERT(dcgmRunDiag_version2 == (long)33575672, 1);
DCGM_CASSERT(dcgmRunDiag_version3 == (long)50352936, 1);
DCGM_CASSERT(dcgmRunDiag_version4 == (long)67130424, 1);
DCGM_CASSERT(dcgmRunDiag_version5 == (long)83907736, 1);
DCGM_CASSERT(dcgmVgpuDeviceAttributes_version == (long)16787744, 1);
DCGM_CASSERT(dcgmVgpuInstanceAttributes_version == (long)16777556, 1);
DCGM_CASSERT(dcgmVgpuConfig_version == (long)16777256, 1);
DCGM_CASSERT(dcgmModuleGetStatuses_version == (long)0x01000088, 1);
DCGM_CASSERT(dcgmModuleBlacklist_version1 == (long)0x01000008, 1);

/* Min and Max macros */
#ifndef DCGM_MIN
#define DCGM_MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef DCGM_MAX
#define DCGM_MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef DCGM_ARRAY_CAPACITY
#define DCGM_ARRAY_CAPACITY(a) (sizeof(a) / sizeof(a[0]))
#endif

#ifdef __cplusplus
}
#endif

#endif  /* DCGM_STRUCTS_H */
