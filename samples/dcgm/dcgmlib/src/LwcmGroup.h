/* 
 * File:   LwcmGroup.h
 */

#ifndef DCGMGROUP_H
#define	DCGMGROUP_H

#include <iostream>
#include <vector>
#include <map>
#include "dcgm_structs.h"
#include "lwos.h"
#include "LwcmCacheManager.h"
#include "LwcmConnection.h"

using namespace std;

/******************************************************************************
 *
 * This is a callback to provide LwcmGroupManager to be called when a group is 
 * removed from the group manager 
 * 
 * userData IN: A user-supplied pointer that was passed to 
 * LwcmGroupManager::SubscribeForGroupEvents
 *
 *****************************************************************************/
typedef void (*dcgmOnRemoveGroup_f)(unsigned int groupId, void *userData);

/* Array entry to track each callback that has been registered */
typedef struct 
{
    dcgmOnRemoveGroup_f callback;
    void *userData;
} lwcmGroupRemoveCBEntry_t;

/*****************************************************************************/

class LwcmGroup;

class LwcmGroupManager
{
public:
    LwcmGroupManager(DcgmCacheManager *cacheManager);
    ~LwcmGroupManager();
    
    /*****************************************************************************
     * This method is used to add a group to the group manager. Ensures that the
     * group name is unique within the group manager
     * 
     * @param connectionId  IN  :   ConnectionId
     * @param groupName     IN  :   Group Name to assign
     * @param type          IN  :   Type of group to be created
     * @param groupId       OUT :   Identifier to represent the group
     * 
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     *****************************************************************************/
    dcgmReturn_t AddNewGroup(dcgm_connection_id_t connectionId, string groupName, dcgmGroupType_t type, unsigned int *groupId);

    /*****************************************************************************
     * This method is used to remove a group from the group manager
     *
     * @param connectionId  IN  : ConnectionId
     * @param groupId       IN  : Group ID to be removed
     * 
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     *****************************************************************************/
    dcgmReturn_t RemoveGroup(dcgm_connection_id_t connectionId, unsigned int groupId);
    
    
    /*****************************************************************************
     * Removes all the groups corresponding to a connection
     * @param connectionId
     * @return 
     *****************************************************************************/
    dcgmReturn_t RemoveAllGroupsForConnection(dcgm_connection_id_t connectionId);
    
    /*****************************************************************************
     * This method is used to get all the groups configured on the system
     * @param connectionId  IN  :   Connection ID 
     * @param groupIdList   OUT :   List of all the groups configured on the system
     * @return 
     *****************************************************************************/
    int GetAllGroupIds(dcgm_connection_id_t connectionId, unsigned int groupIdList[DCGM_MAX_NUM_GROUPS + 1], 
            unsigned int *count);

    /*****************************************************************************
     * This method is used to check if a group is within bound,s is null or references the default group
     * The ID refering to mAllGpusGroupId must only be referenced by DCGM_GROUP_ALL_GPUS or it will return
     * an BAD PARAM error
     * @param groupIdIn  IN  :  Group to be verified (DCGM_GROUP_ALL_GPUS possible input)
     * @param groupIdOut OUT :  Updated group id (DCGM_GROUP_ALL_GPUS mapped to its ID)
     * @return
     * DCGM_ST_OK             : Success
     * DCGM_ST_BADPARAM       : group ID is invalid (out of bounds) , Out is unchanged
     * DCGM_ST_NOT_CONFIGURED : group ID references default group, Out is unchanged
     *****************************************************************************/
    dcgmReturn_t verifyAndUpdateGroupId(unsigned int *groupId);

    /*****************************************************************************
     * This method is used to get the default group containing all GPUs on the system
     * @return
     * group ID of mAllGpusGroupId
     *****************************************************************************/
    unsigned int GetAllGpusGroup();

    /*****************************************************************************
     * This method is used to get the default group containing all LwSwitches on 
     * the system
     * 
     * @return
     * group ID of mAllLwSwitchesGroupId
     *****************************************************************************/
    unsigned int GetAllLwSwitchesGroup();

    /*****************************************************************************
     * This method is used to add an entity to a group
     *
     * @param connectionId  IN: Connection ID
     * @param groupId       IN: Group to add a GPU to
     * @param entityGroupId IN: Entity group of the entity to add to this group
     * @param entityId      IN: Entity id of the entity to add to this group
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     *****************************************************************************/
    dcgmReturn_t AddEntityToGroup(dcgm_connection_id_t connectionId, unsigned int groupId, 
                                  dcgm_field_entity_group_t entityGroupId, 
                                  dcgm_field_eid_t entityId);

    /*****************************************************************************
     * This method is used to remove an entity from a group
     *
     * @param connectionId  IN: Connection ID
     * @param groupId       IN: Group to remove a GPU from
     * @param entityGroupId IN: Entity group of the entity to remove from this group
     * @param entityId      IN: Entity id of the entity to remove from this group
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     *
     *****************************************************************************/
    dcgmReturn_t RemoveEntityFromGroup(dcgm_connection_id_t connectionId, unsigned int groupId, 
                                       dcgm_field_entity_group_t entityGroupId, 
                                       dcgm_field_eid_t entityId);

    /*****************************************************************************
     * This method is used to get all of the GPU ids of a group.
     *
     * This saves locking and unlocking the LwcmGroup over and over again for
     * each gpu index
     * 
     * NOTE: Non-GPU entities like Switches are ignored by this method
     *
     * @param connectionId IN: Connection ID
     * @param groupId      IN  Group to get gpuIds of
     * @param gpuIds      OUT: Vector of GPU IDs to populate (passed by reference)
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     */
    dcgmReturn_t GetGroupGpuIds(dcgm_connection_id_t connectionId, unsigned int groupId,
                                std::vector<unsigned int> &gpuIds);

    /*****************************************************************************
     * This method is used to get all of the entities of a group
     *
     * This saves locking and unlocking the LwcmGroup over and over again for
     * each entity index
     *
     * @param connectionId IN: Connection ID
     * @param groupId      IN  Group to get gpuIds of
     * @param entities    OUT: Vector of entities to populate (passed by reference)
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     */
    dcgmReturn_t GetGroupEntities(dcgm_connection_id_t connectionId, unsigned int groupId,
                                  std::vector<dcgmGroupEntityPair_t> &entities);

    /*****************************************************************************
     * Gets Name of the group
     *
     * @param connectionId IN: Connection ID
     * @param groupId      IN  Group to get gpuIds of
     *
     * @return
     * Group Name
     *****************************************************************************/
     string GetGroupName(dcgm_connection_id_t connectionId, unsigned int groupId);

    /*****************************************************************************
     * Are all of the GPUs in this group the same SKU?
     *
     * @param connectionId   IN: Connection ID
     * @param groupId        IN: Group to get gpuIds of
     * @param areAllSameSku OUT: 1 if all of the GPUs of this group are the same
     *                           0 if any of the GPUs of this group are different from each other
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     */
    dcgmReturn_t AreAllTheSameSku(dcgm_connection_id_t connectionId, unsigned int groupId, int *areAllSameSku);

    /*****************************************************************************
     * Do all GPUs in this group support policy management
     *
     * @param connectionId       IN: Connection ID
     * @param groupId            IN: Group to check
     * @param supportPolicyMgmt OUT: 1 if all GPUs in this group support policy management
     *                               0 if any GPUs in this group don't support policy management 
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error 
     */
    dcgmReturn_t AllSupportPolicyManagement(dcgm_connection_id_t connectionId, unsigned int groupId, int *supportPolicyMgmt);

    /*****************************************************************************
     * Handle a client disconnecting
     */
    void OnConnectionRemove(dcgm_connection_id_t connectionId, LwcmServerConnection *pConnection);

    /*****************************************************************************
     * Subscribe to be notified when events occur for a group
     * 
     * onRemoveCB  IN: Callback to ilwoke when a group is removed
     * userData    IN: User data pointer to pass to the callbacks. This can be the
     *                 "this" of your object.
     */
    void SubscribeForGroupEvents(dcgmOnRemoveGroup_f onRemoveCB, void *userData);

private:

    /*****************************************************************************
     * This method is used to create the default groups containing all the GPUs on 
     * the system and all of the LwSwitches on the system.
     * 
     * @param heAllGpusId  IN  :  Group ID to be stored as the HE default group
     * @return
     * DCGM_ST_OK             : Success
     * DCGM_ST_?              : Error
     *****************************************************************************/
    dcgmReturn_t CreateDefaultGroups();

    /*****************************************************************************
     * Helper method to generate next groupId
     * @return
     * Next group ID to be used by the group manager to ensure uniqueness of 
     * group IDs.
     *****************************************************************************/
    unsigned int GetNextGroupId();

    /******************************************************************************
     * Private helper to get a Group pointer by connectionId and groupId
     *
     * NOTE: Assumes group manager has been locked with Lock()
     *
     * @param connectionId IN: Connection ID
     * @param groupId      IN  Group to get gpuIds of
     *
     * @return Group pointer on success.
     *         NULL if not found
     */

    LwcmGroup *GetGroupById(dcgm_connection_id_t connectionId, unsigned int groupId);

    /*****************************************************************************
     * Add every entity of a given entityGroup to this group. 
     *
     * @return: DCGM_ST_OK on success.
     *          Other DCGM_ST_? on error.
     */
    dcgmReturn_t AddAllEntitiesToGroup(LwcmGroup *pLwcmGrp, dcgm_field_entity_group_t entityGroupId);

    /*****************************************************************************
     * Lock/Unlocks methods to protect the maps for group IDs and group Names
     *****************************************************************************/
    int Lock();
    int Unlock();

    LWOSCriticalSection mLock; /* Lock used for accessing table for the groups */
    unsigned int mGroupIdSequence; /* Group ID sequence */
    unsigned int mNumGroups;       /* Number of groups configured on the system */
    unsigned int mAllGpusGroupId; /* This is a cached group ID to a group containing all GPUs */
    unsigned int mAllLwSwitchesGroupId; /* This is a cached group ID to a group containing all LwSwitches */
    
    typedef map<unsigned int, LwcmGroup*> GroupIdMap;
    
    GroupIdMap mGroupIdMap; /* GroupId -> LwcmGroup object map of all groups */

    DcgmCacheManager *mpCacheManager; /* Pointer to the cache manager */

    std::vector<lwcmGroupRemoveCBEntry_t>mOnRemoveCBs; /* Callbacks to ilwoke when a group is removed */
};

class LwcmGroup 
{
public:
    LwcmGroup(dcgm_connection_id_t connectionId, string name, unsigned int groupId, 
              DcgmCacheManager *cacheManager);
    virtual ~LwcmGroup();

    /*****************************************************************************
     * This method is used to add an entity to this group
     * 
     * @param entityGroupId IN: Entity group of the entity to add to this group
     * @param entityId      IN: Entity id of the entity to add to this group
     * 
     * @return 
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     *****************************************************************************/
    dcgmReturn_t AddEntityToGroup(dcgm_field_entity_group_t entityGroupId, 
                                  dcgm_field_eid_t entityId);
    
    /*****************************************************************************
     * This method is used to remove an entity from this group
     * 
     * @param entityGroupId IN: Entity group of the entity to remove from this group
     * @param entityId      IN: Entity id of the entity to remove from this group
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error* 
     *****************************************************************************/
    dcgmReturn_t RemoveEntityFromGroup(dcgm_field_entity_group_t entityGroupId, 
                                       dcgm_field_eid_t entityId);
    
    /*****************************************************************************
     * Gets Name for the group
     * @return
     * Group Name
     *****************************************************************************/
     string GetGroupName();
    
     /*****************************************************************************
      * Get Group Id
      * @return
      * Group ID
     *****************************************************************************/
    unsigned int GetGroupId();

    /*****************************************************************************
     * Get the connection ID that created this group
     * @return
     * Connection ID
    *****************************************************************************/
    dcgm_connection_id_t GetConnectionId();
    
    /*****************************************************************************
     * This method is used to get all of the entities of a group
     *
     * This saves locking and unlocking the LwcmGroup over and over again for
     * each entity
     *
     * @param entities  OUT: Vector of entities to populate (passed by reference)
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error*
     */
    dcgmReturn_t GetEntities(std::vector<dcgmGroupEntityPair_t> &entities);

    /*****************************************************************************
     * Are all of the GPUs in this group the same SKU?
     *
     * Returns 1 if all of the GPUs of this group are the same
     *         0 if any of the GPUs of this group are different from each other
     */
    int AreAllTheSameSku(void);

private:
    unsigned int mGroupId;              /* ID representing GPU group */
    string mName;                       /* Name for the group group */
    vector<dcgmGroupEntityPair_t> mEntityList; /* List of entities */
    dcgm_connection_id_t mConnectionId; /* Connection ID that created this group */
    DcgmCacheManager *mpCacheManager;   /* Pointer to the cache manager */
};

#endif	/* DCGMGROUP_H */
