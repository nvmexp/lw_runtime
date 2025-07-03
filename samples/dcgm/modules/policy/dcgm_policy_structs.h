#ifndef DCGM_POLICY_STRUCTS_H
#define DCGM_POLICY_STRUCTS_H

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Policy Subrequest IDs */
#define DCGM_POLICY_SR_GET_POLICIES            1
#define DCGM_POLICY_SR_SET_POLICY              2
#define DCGM_POLICY_SR_REGISTER                3
#define DCGM_POLICY_SR_UNREGISTER              4
#define DCGM_POLICY_SR_COUNT                   5 /* Keep as last entry and 1 greater */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/

/**
 * Subrequest DCGM_POLICY_SR_GET_POLICIES
 */
typedef struct dcgm_policy_msg_get_policies_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;         /*  IN: Group ID to get the policies of */
    int numPolicies;              /* OUT: Number of entries in polcies[] that are set */
    int unused;                   /* Unused. Here to align next member on an 8-byte boundary */
    dcgmPolicy_t policies[DCGM_MAX_NUM_DEVICES]; /* OUT: policies of the GPUs in the group */
} dcgm_policy_msg_get_policies_v1;

#define dcgm_policy_msg_get_policies_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_get_policies_v1,1)
#define dcgm_policy_msg_get_policies_version dcgm_policy_msg_get_policies_version1

typedef dcgm_policy_msg_get_policies_v1 dcgm_policy_msg_get_policies_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_POLICY_SR_SET_POLICY
 */
typedef struct dcgm_policy_msg_set_policy_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;         /*  IN: Group ID to set the POLICY systems of */
    dcgmPolicy_t policy;          /*  IN: Policy to set for the group */
} dcgm_policy_msg_set_policy_v1;

#define dcgm_policy_msg_set_policy_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_set_policy_v1,1)
#define dcgm_policy_msg_set_policy_version dcgm_policy_msg_set_policy_version1

typedef dcgm_policy_msg_set_policy_v1 dcgm_policy_msg_set_policy_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_POLICY_SR_REGISTER
 */
typedef struct dcgm_policy_msg_register_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;            /*  IN: Group ID to register for policy updates from */
    dcgmPolicyCondition_t condition; /*  IN: Policy condition to register for */
} dcgm_policy_msg_register_v1;

#define dcgm_policy_msg_register_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_register_v1,1)
#define dcgm_policy_msg_register_version dcgm_policy_msg_register_version1

typedef dcgm_policy_msg_register_v1 dcgm_policy_msg_register_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_POLICY_SR_UNREGISTER
 */
typedef struct dcgm_policy_msg_unregister_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;            /*  IN: Group ID to unregister policy updates from.
                                             This parameter is lwrrently ignored, just like
                                             condition. It seems disingenuous to consider one
                                             but not the other */
    dcgmPolicyCondition_t condition; /*  IN: Policy condition to register for. Note that
                                             this parameter is lwrrently ignored, as it was
                                             before DCGM 1.5. It is being left in place in
                                             case it is not ignored in the future. */
} dcgm_policy_msg_unregister_v1;

#define dcgm_policy_msg_unregister_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_unregister_v1,1)
#define dcgm_policy_msg_unregister_version dcgm_policy_msg_unregister_version1

typedef dcgm_policy_msg_unregister_v1 dcgm_policy_msg_unregister_t;

/*****************************************************************************/

#endif //DCGM_POLICY_STRUCTS_H
