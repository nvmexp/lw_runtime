/*
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 */

#pragma once

#ifndef __CLUSTER_PROTOTYPE_H__
#define __CLUSTER_PROTOTYPE_H__

#include <lwca.h>

#include <xmma/hopper/bringup/lwda_uuid.h>

namespace xmma {
namespace bringup {

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#if 0
} // To trick formatter
#endif

//------------------------------------------------------------------
// Prototype cluster APIs for functional testing
//------------------------------------------------------------------

//LW_DEFINE_UUID(LW_ETID_CLUSTER_PROTOTYPE, 
//    0x26dc3417, 0x0d80, 0x4547, 0x87, 0x26, 0xc0, 0xf1, 0xe7, 0xdd, 0x8b, 0xca);
static const LWuuid LW_ETID_CLUSTER_PROTOTYPE = {{((char)(651965463 & 255)), ((char)((651965463 >> 8) & 255)), ((char)((651965463 >> 16) & 255)), ((char)((651965463 >> 24) & 255)), ((char)(3456 & 255)), ((char)((3456 >> 8) & 255)), ((char)(17735 & 255)), ((char)((17735 >> 8) & 255)), ((char)(135 & 255)), ((char)(38 & 255)), ((char)(192 & 255)), ((char)(241 & 255)), ((char)(231 & 255)), ((char)(221 & 255)), ((char)(139 & 255)), ((char)(202 & 255))}}; 

#define LW_ETID_ClusterPrototype LW_ETID_CLUSTER_PROTOTYPE

typedef enum {
    LW_CLUSTER_PROT_SCHEDULING_POLICY_DEFAULT,
    LW_CLUSTER_PROT_SCHEDULING_POLICY_LOAD_BALANCING,
    LW_CLUSTER_PROT_SCHEDULING_POLICY_SPREAD
} LWclusterPrototypeSchedulingPolicy;

typedef struct LWetblClusterPrototype_st {
    size_t struct_size;

    // \brief Set the default cluster dimensions of a function. All
    // subsequent launches will default to use the set cluster size.
    // To reset, set all dimensions to 0.
    //
    LWresult (LWDAAPI *SetFunctionClusterDim)(LWfunction func, unsigned int clusterDimX, unsigned int clusterDimY, unsigned int clusterDimZ);

    // \brief Set the default cluster scheduling policy. All
    // subsequent launches of this kernel function will default to use
    // the set policy.
    // To reset, set the policy to LW_CLUSTER_PROT_SCHEDULING_POLICY_DEFAULT.
    //
    LWresult (LWDAAPI *SetFunctionClusterSchedulingPolicy)(LWfunction func, LWclusterPrototypeSchedulingPolicy policy);

    // \brief Allow the function to be launched with non-portable
    // cluster size.
    //
    LWresult (LWDAAPI *SetFunctionClusterNonPortableSizeSupport)(LWfunction func, unsigned int enable);

} LWetblClusterPrototype;

#if 0
{ // To trick formatter
#endif
#ifdef __cplusplus
}
#endif // __cplusplus


#endif // __CLUSTER_PROTOTYPE_H__

} // end namespace bringup
} // end namespace xmma
