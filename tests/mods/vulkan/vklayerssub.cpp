/*
* LWIDIA_COPYRIGHT_BEGIN
*
* Copyright 2017-2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* LWIDIA_COPYRIGHT_END
*/

// Warning: DO NOT INCLUDE "vkmain.h" here as it renames vk* functions to disabledVk*
#include "vulkan/vk_layer.h"
#include "vk_layer_dispatch_table.h"
#include "vklayers.h"

#ifdef __linux__
#include <dlfcn.h>
#endif

namespace
{
#ifdef __linux__
    typedef void* HMODULE;

    HMODULE LoadLibrary(const char* filename)
    {
        return dlopen(filename, RTLD_LAZY | RTLD_LOCAL);
    }

    void* GetProcAddress(HMODULE lib, const char* name)
    {
        return dlsym(lib, name);
    }
#endif

    HMODULE vulkanDll = 0;
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = 0;
    PFN_vkEnumerateInstanceLayerProperties enumerateInstanceLayerProperties = 0;
    PFN_vkEnumerateInstanceExtensionProperties enumerateInstanceExtensionProperties = 0;
    PFN_vkEnumerateInstanceVersion enumerateInstanceVersion = 0;
    PFN_vkCreateDevice createDevice = 0;

    void FillInstanceDispatchTableToDriver(VkLayerInstanceDispatchTable *pIDT)
    {
    #define GET_DRIVER_FUNCTION(name) pIDT->name = reinterpret_cast<PFN_vk##name>(GetProcAddress(vulkanDll, "vk" #name));
        memset(pIDT, 0, sizeof(VkLayerInstanceDispatchTable));
        GET_DRIVER_FUNCTION(CreateInstance);
        GET_DRIVER_FUNCTION(DestroyInstance);
        GET_DRIVER_FUNCTION(EnumeratePhysicalDevices);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceFeatures);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceFormatProperties);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceImageFormatProperties);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceProperties);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceQueueFamilyProperties);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceMemoryProperties);
        GET_DRIVER_FUNCTION(GetInstanceProcAddr);
        GET_DRIVER_FUNCTION(CreateDevice);
        GET_DRIVER_FUNCTION(EnumerateInstanceExtensionProperties);
        GET_DRIVER_FUNCTION(EnumerateDeviceExtensionProperties);
        GET_DRIVER_FUNCTION(EnumerateInstanceLayerProperties);
        GET_DRIVER_FUNCTION(EnumerateDeviceLayerProperties);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceSparseImageFormatProperties);

        GET_DRIVER_FUNCTION(EnumerateInstanceVersion);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceFeatures2);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceProperties2);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceFormatProperties2);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceImageFormatProperties2);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceQueueFamilyProperties2);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceMemoryProperties2);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceSparseImageFormatProperties2);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceExternalBufferProperties);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceExternalSemaphoreProperties);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceExternalFenceProperties);

        GET_DRIVER_FUNCTION(DestroySurfaceKHR);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceSurfaceSupportKHR);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceSurfaceCapabilitiesKHR);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceSurfaceFormatsKHR);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceSurfacePresentModesKHR);

        GET_DRIVER_FUNCTION(GetPhysicalDeviceDisplayPropertiesKHR);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceDisplayPlanePropertiesKHR);
        GET_DRIVER_FUNCTION(GetDisplayPlaneSupportedDisplaysKHR);
        GET_DRIVER_FUNCTION(GetDisplayModePropertiesKHR);
        GET_DRIVER_FUNCTION(CreateDisplayModeKHR);
        GET_DRIVER_FUNCTION(GetDisplayPlaneCapabilitiesKHR);
        GET_DRIVER_FUNCTION(CreateDisplayPlaneSurfaceKHR);

    #ifdef VK_USE_PLATFORM_XLIB_KHR
        GET_DRIVER_FUNCTION(CreateXlibSurfaceKHR);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceXlibPresentationSupportKHR);
    #endif

    #ifdef VK_USE_PLATFORM_XCB_KHR
        GET_DRIVER_FUNCTION(CreateXcbSurfaceKHR);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceXcbPresentationSupportKHR);
    #endif

    #ifdef VK_USE_PLATFORM_WAYLAND_KHR
        GET_DRIVER_FUNCTION(CreateWaylandSurfaceKHR);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceWaylandPresentationSupportKHR);
    #endif

    #ifdef VK_USE_PLATFORM_MIR_KHR
        GET_DRIVER_FUNCTION(CreateMirSurfaceKHR);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceMirPresentationSupportKHR);
    #endif

    #ifdef VK_USE_PLATFORM_ANDROID_KHR
        GET_DRIVER_FUNCTION(CreateAndroidSurfaceKHR);
    #endif

    #ifdef VK_USE_PLATFORM_WIN32_KHR
        GET_DRIVER_FUNCTION(CreateWin32SurfaceKHR);
        GET_DRIVER_FUNCTION(GetPhysicalDeviceWin32PresentationSupportKHR);
    #endif

    #ifdef VK_USE_PLATFORM_VI_NN
        GET_DRIVER_FUNCTION(CreateViSurfaceNN);
    #endif

    #ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
        GET_DRIVER_FUNCTION(AcquireXlibDisplayEXT);
        GET_DRIVER_FUNCTION(GetRandROutputDisplayEXT);
    #endif

    #ifdef VK_USE_PLATFORM_IOS_MVK
        GET_DRIVER_FUNCTION(CreateIOSSurfaceMVK);
    #endif

    #ifdef VK_USE_PLATFORM_MACOS_MVK
        GET_DRIVER_FUNCTION(CreateMacOSSurfaceMVK);
    #endif

    #undef GET_DRIVER_FUNCTION
    }

    void FillDeviceDispatchTableToDriver(VkLayerDispatchTable *pDDT)
    {
    #define GET_DRIVER_FUNCTION(name) pDDT->name = reinterpret_cast<PFN_vk##name>(GetProcAddress(vulkanDll, "vk" #name));
        memset(pDDT, 0, sizeof(VkLayerDispatchTable));
        GET_DRIVER_FUNCTION(GetDeviceProcAddr);
        GET_DRIVER_FUNCTION(DestroyDevice);
        GET_DRIVER_FUNCTION(GetDeviceQueue);
        GET_DRIVER_FUNCTION(QueueSubmit);
        GET_DRIVER_FUNCTION(QueueWaitIdle);
        GET_DRIVER_FUNCTION(DeviceWaitIdle);
        GET_DRIVER_FUNCTION(AllocateMemory);
        GET_DRIVER_FUNCTION(FreeMemory);
        GET_DRIVER_FUNCTION(MapMemory);
        GET_DRIVER_FUNCTION(UnmapMemory);
        GET_DRIVER_FUNCTION(FlushMappedMemoryRanges);
        GET_DRIVER_FUNCTION(IlwalidateMappedMemoryRanges);
        GET_DRIVER_FUNCTION(GetDeviceMemoryCommitment);
        GET_DRIVER_FUNCTION(BindBufferMemory);
        GET_DRIVER_FUNCTION(BindImageMemory);
        GET_DRIVER_FUNCTION(GetBufferMemoryRequirements);
        GET_DRIVER_FUNCTION(GetImageMemoryRequirements);
        GET_DRIVER_FUNCTION(GetImageSparseMemoryRequirements);
        GET_DRIVER_FUNCTION(QueueBindSparse);
        GET_DRIVER_FUNCTION(CreateFence);
        GET_DRIVER_FUNCTION(DestroyFence);
        GET_DRIVER_FUNCTION(ResetFences);
        GET_DRIVER_FUNCTION(GetFenceStatus);
        GET_DRIVER_FUNCTION(WaitForFences);
        GET_DRIVER_FUNCTION(CreateSemaphore);
        GET_DRIVER_FUNCTION(DestroySemaphore);
        GET_DRIVER_FUNCTION(CreateEvent);
        GET_DRIVER_FUNCTION(DestroyEvent);
        GET_DRIVER_FUNCTION(GetEventStatus);
        GET_DRIVER_FUNCTION(SetEvent);
        GET_DRIVER_FUNCTION(ResetEvent);
        GET_DRIVER_FUNCTION(CreateQueryPool);
        GET_DRIVER_FUNCTION(DestroyQueryPool);
        GET_DRIVER_FUNCTION(GetQueryPoolResults);
        GET_DRIVER_FUNCTION(CreateBuffer);
        GET_DRIVER_FUNCTION(DestroyBuffer);
        GET_DRIVER_FUNCTION(CreateBufferView);
        GET_DRIVER_FUNCTION(DestroyBufferView);
        GET_DRIVER_FUNCTION(CreateImage);
        GET_DRIVER_FUNCTION(DestroyImage);
        GET_DRIVER_FUNCTION(GetImageSubresourceLayout);
        GET_DRIVER_FUNCTION(CreateImageView);
        GET_DRIVER_FUNCTION(DestroyImageView);
        GET_DRIVER_FUNCTION(CreateShaderModule);
        GET_DRIVER_FUNCTION(DestroyShaderModule);
        GET_DRIVER_FUNCTION(CreatePipelineCache);
        GET_DRIVER_FUNCTION(DestroyPipelineCache);
        GET_DRIVER_FUNCTION(GetPipelineCacheData);
        GET_DRIVER_FUNCTION(MergePipelineCaches);
        GET_DRIVER_FUNCTION(CreateGraphicsPipelines);
        GET_DRIVER_FUNCTION(CreateComputePipelines);
        GET_DRIVER_FUNCTION(DestroyPipeline);
        GET_DRIVER_FUNCTION(CreatePipelineLayout);
        GET_DRIVER_FUNCTION(DestroyPipelineLayout);
        GET_DRIVER_FUNCTION(CreateSampler);
        GET_DRIVER_FUNCTION(DestroySampler);
        GET_DRIVER_FUNCTION(CreateDescriptorSetLayout);
        GET_DRIVER_FUNCTION(DestroyDescriptorSetLayout);
        GET_DRIVER_FUNCTION(CreateDescriptorPool);
        GET_DRIVER_FUNCTION(DestroyDescriptorPool);
        GET_DRIVER_FUNCTION(ResetDescriptorPool);
        GET_DRIVER_FUNCTION(AllocateDescriptorSets);
        GET_DRIVER_FUNCTION(FreeDescriptorSets);
        GET_DRIVER_FUNCTION(UpdateDescriptorSets);
        GET_DRIVER_FUNCTION(CreateFramebuffer);
        GET_DRIVER_FUNCTION(DestroyFramebuffer);
        GET_DRIVER_FUNCTION(CreateRenderPass);
        GET_DRIVER_FUNCTION(DestroyRenderPass);
        GET_DRIVER_FUNCTION(GetRenderAreaGranularity);
        GET_DRIVER_FUNCTION(CreateCommandPool);
        GET_DRIVER_FUNCTION(DestroyCommandPool);
        GET_DRIVER_FUNCTION(ResetCommandPool);
        GET_DRIVER_FUNCTION(AllocateCommandBuffers);
        GET_DRIVER_FUNCTION(FreeCommandBuffers);
        GET_DRIVER_FUNCTION(BeginCommandBuffer);
        GET_DRIVER_FUNCTION(EndCommandBuffer);
        GET_DRIVER_FUNCTION(ResetCommandBuffer);
        GET_DRIVER_FUNCTION(CmdBindPipeline);
        GET_DRIVER_FUNCTION(CmdSetViewport);
        GET_DRIVER_FUNCTION(CmdSetScissor);
        GET_DRIVER_FUNCTION(CmdSetLineWidth);
        GET_DRIVER_FUNCTION(CmdSetDepthBias);
        GET_DRIVER_FUNCTION(CmdSetBlendConstants);
        GET_DRIVER_FUNCTION(CmdSetDepthBounds);
        GET_DRIVER_FUNCTION(CmdSetStencilCompareMask);
        GET_DRIVER_FUNCTION(CmdSetStencilWriteMask);
        GET_DRIVER_FUNCTION(CmdSetStencilReference);
        GET_DRIVER_FUNCTION(CmdBindDescriptorSets);
        GET_DRIVER_FUNCTION(CmdBindIndexBuffer);
        GET_DRIVER_FUNCTION(CmdBindVertexBuffers);
        GET_DRIVER_FUNCTION(CmdDraw);
        GET_DRIVER_FUNCTION(CmdDrawIndexed);
        GET_DRIVER_FUNCTION(CmdDrawIndirect);
        GET_DRIVER_FUNCTION(CmdDrawIndexedIndirect);
        GET_DRIVER_FUNCTION(CmdDispatch);
        GET_DRIVER_FUNCTION(CmdDispatchIndirect);
        GET_DRIVER_FUNCTION(CmdCopyBuffer);
        GET_DRIVER_FUNCTION(CmdCopyImage);
        GET_DRIVER_FUNCTION(CmdBlitImage);
        GET_DRIVER_FUNCTION(CmdCopyBufferToImage);
        GET_DRIVER_FUNCTION(CmdCopyImageToBuffer);
        GET_DRIVER_FUNCTION(CmdUpdateBuffer);
        GET_DRIVER_FUNCTION(CmdFillBuffer);
        GET_DRIVER_FUNCTION(CmdClearColorImage);
        GET_DRIVER_FUNCTION(CmdClearDepthStencilImage);
        GET_DRIVER_FUNCTION(CmdClearAttachments);
        GET_DRIVER_FUNCTION(CmdResolveImage);
        GET_DRIVER_FUNCTION(CmdSetEvent);
        GET_DRIVER_FUNCTION(CmdResetEvent);
        GET_DRIVER_FUNCTION(CmdWaitEvents);
        GET_DRIVER_FUNCTION(CmdPipelineBarrier);
        GET_DRIVER_FUNCTION(CmdBeginQuery);
        GET_DRIVER_FUNCTION(CmdEndQuery);
        GET_DRIVER_FUNCTION(CmdResetQueryPool);
        GET_DRIVER_FUNCTION(CmdWriteTimestamp);
        GET_DRIVER_FUNCTION(CmdCopyQueryPoolResults);
        GET_DRIVER_FUNCTION(CmdPushConstants);
        GET_DRIVER_FUNCTION(CmdBeginRenderPass);
        GET_DRIVER_FUNCTION(CmdNextSubpass);
        GET_DRIVER_FUNCTION(CmdEndRenderPass);
        GET_DRIVER_FUNCTION(CmdExelwteCommands);
        GET_DRIVER_FUNCTION(CreateSwapchainKHR);
        GET_DRIVER_FUNCTION(DestroySwapchainKHR);
        GET_DRIVER_FUNCTION(GetSwapchainImagesKHR);
        GET_DRIVER_FUNCTION(AcquireNextImageKHR);
        GET_DRIVER_FUNCTION(QueuePresentKHR);
        GET_DRIVER_FUNCTION(CreateSharedSwapchainsKHR);
    #undef GET_DRIVER_FUNCTION
    }

    void FillDeviceDispatchTableFromGetDeviceProcAddr
    (
        VkLayerDispatchTable *pDDT,
        VkDevice device,
        PFN_vkGetDeviceProcAddr pGetDeviceProcAddr
    )
    {
        if (pGetDeviceProcAddr == nullptr)
        {
            return;
        }

    #define GET_DEVICE_FUNCTION(name) pDDT->name = PFN_vk##name(pGetDeviceProcAddr(device, "vk" #name))
        GET_DEVICE_FUNCTION(GetBufferDeviceAddress);
        GET_DEVICE_FUNCTION(DebugMarkerSetObjectTagEXT);
        GET_DEVICE_FUNCTION(DebugMarkerSetObjectNameEXT);
        GET_DEVICE_FUNCTION(CmdDebugMarkerBeginEXT);
        GET_DEVICE_FUNCTION(CmdDebugMarkerEndEXT);
        GET_DEVICE_FUNCTION(CmdDebugMarkerInsertEXT);
        GET_DEVICE_FUNCTION(CmdSetDepthCompareOpEXT);
        GET_DEVICE_FUNCTION(CmdSetStencilOpEXT);
        GET_DEVICE_FUNCTION(CmdBuildAccelerationStructuresKHR);
        GET_DEVICE_FUNCTION(CmdCopyAccelerationStructureKHR);
        GET_DEVICE_FUNCTION(CmdTraceRaysKHR);
        GET_DEVICE_FUNCTION(CmdWriteAccelerationStructuresPropertiesKHR);
        GET_DEVICE_FUNCTION(CreateAccelerationStructureKHR);
        GET_DEVICE_FUNCTION(DestroyAccelerationStructureKHR);
        GET_DEVICE_FUNCTION(GetAccelerationStructureBuildSizesKHR);
        GET_DEVICE_FUNCTION(GetAccelerationStructureDeviceAddressKHR);
        GET_DEVICE_FUNCTION(CreateRayTracingPipelinesKHR);
        GET_DEVICE_FUNCTION(GetRayTracingShaderGroupHandlesKHR);
    #undef GET_DEVICE_FUNCTION
    }
}

VkResult VulkanLayers::Init(bool enableLayersInMODS)
{
    if (createDevice)
    {
        return VK_SUCCESS;
    }

    if (!vulkanDll)
    {
#ifdef _WIN32
        const char* const vulkanLibName = "vulkan-1.dll";
#elif defined(__linux__)
        const char* const vulkanLibName = "libvulkan.so.1";
#endif
        vulkanDll = LoadLibrary(vulkanLibName);
        if (!vulkanDll)
        {
            return VK_NOT_READY;
        }
    }

    getInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        GetProcAddress(vulkanDll, "vkGetInstanceProcAddr"));
    if (!getInstanceProcAddr)
    {
        return VK_NOT_READY;
    }

    enumerateInstanceLayerProperties = reinterpret_cast<
        PFN_vkEnumerateInstanceLayerProperties>(
            GetProcAddress(vulkanDll, "vkEnumerateInstanceLayerProperties"));
    if (!enumerateInstanceLayerProperties)
    {
        return VK_NOT_READY;
    }

    enumerateInstanceExtensionProperties = reinterpret_cast<
        PFN_vkEnumerateInstanceExtensionProperties>(
            GetProcAddress(vulkanDll, "vkEnumerateInstanceExtensionProperties"));
    if (!enumerateInstanceExtensionProperties)
    {
        return VK_NOT_READY;
    }

    enumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            GetProcAddress(vulkanDll, "vkEnumerateInstanceVersion"));

    createDevice = reinterpret_cast<PFN_vkCreateDevice>(
            GetProcAddress(vulkanDll, "vkCreateDevice"));
    if (!createDevice)
    {
        return VK_NOT_READY;
    }

    return VK_SUCCESS;
}

VkResult VulkanLayers::EnumerateInstanceLayerProperties
(
    uint32_t*                                   pPropertyCount,
    VkLayerProperties*                          pProperties
)
{
    return enumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

VkResult VulkanLayers::EnumerateInstanceExtensionProperties
(
    const char*                                 pLayerName,
    uint32_t*                                   pPropertyCount,
    VkExtensionProperties*                      pProperties
)
{
    return enumerateInstanceExtensionProperties(pLayerName, pPropertyCount, pProperties);
}

uint32_t VulkanLayers::EnumerateInstanceVersion()
{
    uint32_t ver = VK_API_VERSION_1_0;

    if (!enumerateInstanceVersion)
        return ver;

    enumerateInstanceVersion(&ver);

    return ver;
}

VkResult VulkanLayers::CreateInstance
(
    const VkInstanceCreateInfo*                 pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkInstance*                                 pInstance,
    VkLayerInstanceDispatchTable*               pvi
)
{
    FillInstanceDispatchTableToDriver(pvi);
    VkResult res = pvi->CreateInstance(pCreateInfo, pAllocator, pInstance);
    if (res == VK_SUCCESS)
    {
        pvi->GetPhysicalDeviceProperties2KHR =
            PFN_vkGetPhysicalDeviceProperties2KHR(pvi->GetInstanceProcAddr(
                *pInstance, "vkGetPhysicalDeviceProperties2KHR"));
    }
    return res;
}

VkResult VulkanLayers::CreateDevice
(
    VkInstance                                  instance,
    VkPhysicalDevice                            physicalDevice,
    const VkDeviceCreateInfo*                   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDevice*                                   pDevice,
    VkLayerDispatchTable*                       pvd
)
{
    FillDeviceDispatchTableToDriver(pvd);
    VkResult res = createDevice(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (res == VK_SUCCESS)
    {
        FillDeviceDispatchTableFromGetDeviceProcAddr(pvd, *pDevice, pvd->GetDeviceProcAddr);
    }
    return res;
}
