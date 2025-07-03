/***************************************************************************************************
 * Copyright (c) 2011-2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <xmma/xmma.h>

// NOTE : This include file is for bring up purposes only
// Will be modified once public facing APIs are ready / implemented

#define LW_INIT_UUID
#include <xmma/hopper/bringup/cluster_prototype.h>

namespace xmma {

using namespace xmma::bringup;

struct Cga_Launcher {

private :
    dim3 cluster_dims_;
    dim3 grid_dims_;

    XMMA_HOST xmma::Error check_cluster_dims(const dim3& grid, const dim3& cluster) {

        // Check for hardware compatibility
        if( ((cluster.x * cluster.y * cluster.z) <= 32) &&
            (grid.x % cluster.x == 0) && (grid.y % cluster.y == 0) &&
            (grid.z % cluster.z == 0) ) {  
            return xmma::Error::SUCCESS;
        } else {
            return xmma::Error::ERROR_ILWALID_PARAMS;
        }
    }

public :

    // Having only a basic constructor enables us to check validity before assignment
    // Also it forces users to use the initialize method
    XMMA_HOST Cga_Launcher()
    : cluster_dims_(dim3 {1,1,1}),
    grid_dims_(dim3 {1,1,1}) { }

    XMMA_HOST xmma::Error initialize(const void* Kernel_Function, 
                                     const dim3& grid_dims,
                                     const dim3& cluster_dims) {

#if __LWDACC_VER_MAJOR__ >= 11
        if( check_cluster_dims(grid_dims, cluster_dims) == xmma::Error::SUCCESS ) {

            // Will be useful when trying to launch GPC_CGAs
            cluster_dims_ = cluster_dims;
            grid_dims_ = grid_dims;

            // Driver Internal functions to setup CGA launch
            lwdaFunction_t func;
            const LWetblClusterPrototype *clusterProt = NULL;

            XMMA_LWDA_CALL( lwdaGetExportTable((const void**)&clusterProt, 
                                                &LW_ETID_ClusterPrototype) );
            XMMA_LWDA_CALL( lwdaGetFuncBySymbol(&func, Kernel_Function) );

            clusterProt->SetFunctionClusterDim((LWfunction)func, 
                                                cluster_dims.x, cluster_dims.y, cluster_dims.z);
            clusterProt->SetFunctionClusterNonPortableSizeSupport((LWfunction)func, 1);

            return xmma::Error::SUCCESS;
        } else {
            return xmma::Error::ERROR_ILWALID_PARAMS;
        }
#else
        return xmma::Error::ERROR_ILWALID_PARAMS;
#endif

    }


    // To be added later - once there is support from driver / lwca
    XMMA_HOST xmma::Error launch() {
        return xmma::Error::SUCCESS;
    } 
};

XMMA_HOST Cga_Launcher create_cga_launcher(){
    return Cga_Launcher {};
}

} // namespace xmma
