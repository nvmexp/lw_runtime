/*
 * Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <exp/pipeline/ReadLWVMContainer.h>

#include <exp/context/ErrorHandling.h>

#include <corelib/compiler/LLVMUtil.h>

#include <lwvm/ClientInterface/Container.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <lwvm/Support/APIUpgradeUtilities.h>

///////////////////////////////////////////////
// BinaryLwvmIRContainer and associated data copied from
// drivers/compiler/lwvm/common/lib/ClientInterface/Container.cpp
// The format of the LWVM Container is managed by the aforementioned file.  Any
// modifications to the container format should go there, with an accompanying
// update to the LWVM_CONTAINER_VERSION macros.

// Don't modify the code we took from Container.cpp
// clang-format off

/////////////////////////////////////////////
/// Binary Container format definition:
// It will layout space as:
//   BinaryLwvmIRContainer header
//   array of OptionTuple
//   OptionData
//   IR

#define LWVMIRCONTAINER_MAGIC 0x7f4e43ed

using VersionPairU8 = std::pair<uint8_t, uint8_t>;

struct BinaryLwvmIRContainer {
  uint32_t Magic;                 // LWVMIRCONTAINER_MAGIC
  VersionPairU8 ContainerVersion;
  VersionPairU8 LwvmIRVersion;
  VersionPairU8 LwvmDebugVersion;
  VersionPairU8 LlvmVersion;
  uint16_t Options;               // offset from start of struct to *OptionTuple
  uint16_t IRLevel;               // level of IR: IRLEVEL_*
  uint32_t OptionsData;           // offset from start of struct to *OptionData
  uint32_t IR;                    // offset from start of struct to IR
};

// clang-format on

namespace optix_exp {

bool isLWVMContainer( const llvm::StringRef& buffer )
{
    if( buffer.size() < sizeof( BinaryLwvmIRContainer::Magic ) )
        return false;
    const BinaryLwvmIRContainer* header          = reinterpret_cast<const BinaryLwvmIRContainer*>( buffer.data() );
    bool                         isLWVMContainer = header->Magic == LWVMIRCONTAINER_MAGIC;
    return isLWVMContainer;
}

static void* MemAlloc( void*, size_t size )
{
    return malloc( size );
}

static void MemFree( void*, void* addr )
{
    free( addr );
}

static void addSmVersionToModule( LWVMArch arch, llvm::Module* module )
{
    int archVal = 0;
    switch( arch )
    {
        case LWVM_ARCH_UNKNOWN:
            break;
        case LWVM_ARCH_KEPLER_3_0:
            archVal = 30;
            break;
        case LWVM_ARCH_KEPLER_3_2:
            archVal = 32;
            break;
        case LWVM_ARCH_KEPLER_3_5:
            archVal = 35;
            break;
        case LWVM_ARCH_KEPLER_3_7:
            archVal = 37;
            break;
        case LWVM_ARCH_MAXWELL_5_0:
            archVal = 50;
            break;
        case LWVM_ARCH_MAXWELL_5_2:
            archVal = 52;
            break;
        case LWVM_ARCH_MAXWELL_5_3:
            archVal = 53;
            break;
        case LWVM_ARCH_PASCAL_6_0:
            archVal = 60;
            break;
        case LWVM_ARCH_PASCAL_6_1:
            archVal = 61;
            break;
        case LWVM_ARCH_PASCAL_6_2:
            archVal = 62;
            break;
        case LWVM_ARCH_VOLTA_7_0:
            archVal = 70;
            break;
        case LWVM_ARCH_VOLTA_7_2:
            archVal = 72;
            break;
        case LWVM_ARCH_TURING_7_3:
            archVal = 73;
            break;
        case LWVM_ARCH_TURING_7_5:
            archVal = 75;
            break;
        case LWVM_ARCH_AMPERE_8_0:
            archVal = 80;
            break;
        case LWVM_ARCH_AMPERE_8_2:
            archVal = 82;
            break;
        case LWVM_ARCH_AMPERE_8_6:
            archVal = 86;
            break;
        case LWVM_ARCH_AMPERE_8_7:
            archVal = 87;
            break;
        case LWVM_ARCH_AMPERE_8_8:
            archVal = 88;
            break;
        case LWVM_ARCH_ADA_8_9:
            archVal = 89;
            break;
        case LWVM_ARCH_HOPPER_9_0:
            archVal = 90;
            break;
    }
    llvm::NamedMDNode* lwvmannotate = module->getOrInsertNamedMetadata( "lwvm.annotations" );
    llvm::MDString*    str          = llvm::MDString::get( module->getContext(), "targetArch" );
    MetadataValueTy*   av           = corelib::createMetadata( module->getContext(), archVal );

    std::vector<llvm::Metadata*> values = { av, str };
    llvm::MDNode*                mdNode = llvm::MDNode::get( module->getContext(), values );

    lwvmannotate->addOperand( mdNode );
}

static void addFastMathOptionsToModule( LWVMFastMathOptions fmOptions, llvm::Module* module )
{
    llvm::NamedMDNode* lwvmannotate = module->getOrInsertNamedMetadata( "lwvm.annotations" );

    // Lwrrently we are only interested in the Fmad option
    llvm::MDString*              str    = llvm::MDString::get( module->getContext(), "fmad" );
    MetadataValueTy*             av     = corelib::createMetadata( module->getContext(), (int)fmOptions.Fmad );
    std::vector<llvm::Metadata*> values = { av, str };
    llvm::MDNode*                mdNode = llvm::MDNode::get( module->getContext(), values );

    lwvmannotate->addOperand( mdNode );
}

OptixResult getModuleFromLWVMContainer( std::unique_ptr<llvm::MemoryBuffer>& memBuffer, llvm::Module*& module, ErrorDetails& errDetails )
{
    LWVMClientDesc client          = {};
    client.AllocFn                 = MemAlloc;
    client.FreeFn                  = MemFree;
    client.AllocArg                = NULL;
    llvm::LWVMContainer* container = llvm::LWVMContainer::createFromContainer( memBuffer.release(), &client );
    if( container == nullptr )
        return OPTIX_ERROR_ILWALID_PTX;
    LWVMArch arch = container->getOptions()->ArchVariant;
    LWVMFastMathOptions fmOptions = container->getOptions()->FastMathOptions;
    module        = container->releaseModule();
    delete container;
    if( module == nullptr )
        return OPTIX_ERROR_ILWALID_PTX;
    addSmVersionToModule( arch, module );
    addFastMathOptionsToModule( fmOptions, module );
    return OPTIX_SUCCESS;
}
}  // end namespace optix_exp
