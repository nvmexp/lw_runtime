/* Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample demonstrates LWCA Interop with Vulkan using lwMemMap APIs.
 * Allocating device memory and updating values in those allocations are
 * performed by LWCA and the contents of the allocation are visualized by
 * Vulkan.
 */

#include "VulkanBaseApp.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>

#include "MonteCarloPi.h"
#include <helper_lwda.h>
#include <lwca.h>

#include "helper_multiprocess.h"

//#define DEBUG
#ifndef DEBUG
#define ENABLE_VALIDATION (false)
#else
#define ENABLE_VALIDATION (true)
#endif

#define NUM_SIMULATION_POINTS 50000

std::string exelwtion_path;

class VulkanLwdaPi : public VulkanBaseApp {
  typedef struct UniformBufferObject_st { float frame; } UniformBufferObject;

  VkBuffer m_inCircleBuffer, m_xyPositionBuffer;
  VkDeviceMemory m_inCircleMemory, m_xyPositionMemory;
  VkSemaphore m_vkWaitSemaphore, m_vkSignalSemaphore;
  MonteCarloPiSimulation m_sim;
  UniformBufferObject m_ubo;
  lwdaStream_t m_stream;
  lwdaExternalSemaphore_t m_lwdaWaitSemaphore, m_lwdaSignalSemaphore;
  using chrono_tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
  chrono_tp m_lastTime;
  size_t m_lastFrame;

 public:
  VulkanLwdaPi(size_t num_points)
      : VulkanBaseApp("simpleVulkanMMAP", ENABLE_VALIDATION),
        m_inCircleBuffer(VK_NULL_HANDLE),
        m_xyPositionBuffer(VK_NULL_HANDLE),
        m_inCircleMemory(VK_NULL_HANDLE),
        m_xyPositionMemory(VK_NULL_HANDLE),
        m_sim(num_points),
        m_ubo(),
        m_stream(0),
        m_vkWaitSemaphore(VK_NULL_HANDLE),
        m_vkSignalSemaphore(VK_NULL_HANDLE),
        m_lwdaWaitSemaphore(),
        m_lwdaSignalSemaphore(),
        m_lastFrame(0) {
    // Add our compiled vulkan shader files
    char* vertex_shader_path =
        sdkFindFilePath("vert.spv", exelwtion_path.c_str());
    char* fragment_shader_path =
        sdkFindFilePath("frag.spv", exelwtion_path.c_str());
    m_shaderFiles.push_back(
        std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, vertex_shader_path));
    m_shaderFiles.push_back(
        std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_shader_path));
  }

  ~VulkanLwdaPi() {
    if (m_stream) {
      // Make sure there's no pending work before we start tearing down
      checkLwdaErrors(lwdaStreamSynchronize(m_stream));
      checkLwdaErrors(lwdaStreamDestroy(m_stream));
    }

    if (m_vkSignalSemaphore != VK_NULL_HANDLE) {
      checkLwdaErrors(lwdaDestroyExternalSemaphore(m_lwdaSignalSemaphore));
      vkDestroySemaphore(m_device, m_vkSignalSemaphore, nullptr);
    }
    if (m_vkWaitSemaphore != VK_NULL_HANDLE) {
      checkLwdaErrors(lwdaDestroyExternalSemaphore(m_lwdaWaitSemaphore));
      vkDestroySemaphore(m_device, m_vkWaitSemaphore, nullptr);
    }
    if (m_xyPositionBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_xyPositionBuffer, nullptr);
    }
    if (m_xyPositionMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_xyPositionMemory, nullptr);
    }
    if (m_inCircleBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_inCircleBuffer, nullptr);
    }
    if (m_inCircleMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_inCircleMemory, nullptr);
    }
  }

  void fillRenderingCommandBuffer(VkCommandBuffer& commandBuffer) {
    VkBuffer vertexBuffers[] = {m_inCircleBuffer, m_xyPositionBuffer};
    VkDeviceSize offsets[] = {0, 0};
    vkCmdBindVertexBuffers(commandBuffer, 0,
                           sizeof(vertexBuffers) / sizeof(vertexBuffers[0]),
                           vertexBuffers, offsets);
    vkCmdDraw(commandBuffer, (uint32_t)(m_sim.getNumPoints()), 1, 0, 0);
  }

  void getVertexDescriptions(
      std::vector<VkVertexInputBindingDescription>& bindingDesc,
      std::vector<VkVertexInputAttributeDescription>& attribDesc) {
    bindingDesc.resize(2);
    attribDesc.resize(2);

    bindingDesc[0].binding = 0;
    bindingDesc[0].stride = sizeof(float);
    bindingDesc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    bindingDesc[1].binding = 1;
    bindingDesc[1].stride = sizeof(vec2);
    bindingDesc[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    attribDesc[0].binding = 0;
    attribDesc[0].location = 0;
    attribDesc[0].format = VK_FORMAT_R32_SFLOAT;
    attribDesc[0].offset = 0;

    attribDesc[1].binding = 1;
    attribDesc[1].location = 1;
    attribDesc[1].format = VK_FORMAT_R32G32_SFLOAT;
    attribDesc[1].offset = 0;
  }

  void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info) {
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    info.primitiveRestartEnable = VK_FALSE;
  }

  void getWaitFrameSemaphores(
      std::vector<VkSemaphore>& wait,
      std::vector<VkPipelineStageFlags>& waitStages) const {
    if (m_lwrrentFrame != 0) {
      // Have vulkan wait until lwca is done with the vertex buffer before
      // rendering
      // We don't do this on the first frame, as the wait semaphore hasn't been
      // initialized yet
      wait.push_back(m_vkWaitSemaphore);
      // We want to wait until all the pipeline commands are complete before
      // letting lwca work
      waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
  }

  void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const {
    // Add this semaphore for vulkan to signal once the vertex buffer is ready
    // for lwca to modify
    signal.push_back(m_vkSignalSemaphore);
  }

  void initVulkanApp() {
    const size_t lwerts = m_sim.getNumPoints();

    // Obtain lwca device id for the device corresponding to the Vulkan physical
    // device
    int deviceCount;
    int lwdaDevice = lwdaIlwalidDeviceId;
    checkLwdaErrors(lwdaGetDeviceCount(&deviceCount));
    for (int dev = 0; dev < deviceCount; ++dev) {
      lwdaDeviceProp devProp = {};
      checkLwdaErrors(lwdaGetDeviceProperties(&devProp, dev));
      if (isVkPhysicalDeviceUuid(&devProp.uuid)) {
        lwdaDevice = dev;
        break;
      }
    }
    if (lwdaDevice == lwdaIlwalidDeviceId) {
      throw std::runtime_error("No Suitable device found!");
    }

    // On the corresponding lwca device, create the lwca stream we'll using
    checkLwdaErrors(lwdaSetDevice(lwdaDevice));
    checkLwdaErrors(
        lwdaStreamCreateWithFlags(&m_stream, lwdaStreamNonBlocking));
    m_sim.initSimulation(lwdaDevice, m_stream);

    importExternalBuffer(
        (void*)(uintptr_t)m_sim.getPositionShareableHandle(),
        getDefaultMemHandleType(), lwerts * sizeof(vec2),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_xyPositionBuffer,
        m_xyPositionMemory);

    importExternalBuffer(
        (void*)(uintptr_t)m_sim.getInCircleShareableHandle(),
        getDefaultMemHandleType(), lwerts * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_inCircleBuffer,
        m_inCircleMemory);

    // Create the semaphore vulkan will signal when it's done with the vertex
    // buffer
    createExternalSemaphore(m_vkSignalSemaphore,
                            getDefaultSemaphoreHandleType());
    // Create the semaphore vulkan will wait for before using the vertex buffer
    createExternalSemaphore(m_vkWaitSemaphore, getDefaultSemaphoreHandleType());
    // Import the semaphore lwca will use -- vulkan's signal will be lwca's wait
    importLwdaExternalSemaphore(m_lwdaWaitSemaphore, m_vkSignalSemaphore,
                                getDefaultSemaphoreHandleType());
    // Import the semaphore lwca will use -- lwca's signal will be vulkan's wait
    importLwdaExternalSemaphore(m_lwdaSignalSemaphore, m_vkWaitSemaphore,
                                getDefaultSemaphoreHandleType());
  }

  void importLwdaExternalSemaphore(
      lwdaExternalSemaphore_t& lwdaSem, VkSemaphore& vkSem,
      VkExternalSemaphoreHandleTypeFlagBits handleType) {
    lwdaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};

    if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
      externalSemaphoreHandleDesc.type =
          lwdaExternalSemaphoreHandleTypeOpaqueWin32;
    } else if (handleType &
               VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
      externalSemaphoreHandleDesc.type =
          lwdaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
      externalSemaphoreHandleDesc.type =
          lwdaExternalSemaphoreHandleTypeOpaqueFd;
    } else {
      throw std::runtime_error("Unknown handle type requested!");
    }

#ifdef _WIN64
    externalSemaphoreHandleDesc.handle.win32.handle =
        (HANDLE)getSemaphoreHandle(vkSem, handleType);
#else
    externalSemaphoreHandleDesc.handle.fd =
        (int)(uintptr_t)getSemaphoreHandle(vkSem, handleType);
#endif

    externalSemaphoreHandleDesc.flags = 0;

    checkLwdaErrors(
        lwdaImportExternalSemaphore(&lwdaSem, &externalSemaphoreHandleDesc));
  }

  VkDeviceSize getUniformSize() const { return sizeof(UniformBufferObject); }

  void updateUniformBuffer(uint32_t imageIndex, size_t globalFrame) {
    m_ubo.frame = (float)globalFrame;
    void* data;
    vkMapMemory(m_device, m_uniformMemory[imageIndex], 0, getUniformSize(), 0,
                &data);
    memcpy(data, &m_ubo, sizeof(m_ubo));
    vkUnmapMemory(m_device, m_uniformMemory[imageIndex]);
  }

  std::vector<const char*> getRequiredExtensions() const {
    std::vector<const char*> extensions;
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    return extensions;
  }

  std::vector<const char*> getRequiredDeviceExtensions() const {
    std::vector<const char*> extensions;

    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef _WIN64
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif /* _WIN64 */
    return extensions;
  }

  void drawFrame() {
    static chrono_tp startTime = std::chrono::high_resolution_clock::now();

    chrono_tp lwrrentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     lwrrentTime - startTime)
                     .count();

    if (m_lwrrentFrame == 0) {
      m_lastTime = startTime;
    }

    lwdaExternalSemaphoreWaitParams waitParams = {};
    waitParams.flags = 0;
    waitParams.params.fence.value = 0;

    lwdaExternalSemaphoreSignalParams signalParams = {};
    signalParams.flags = 0;
    signalParams.params.fence.value = 0;

    // Have vulkan draw the current frame...
    VulkanBaseApp::drawFrame();
    // Wait for vulkan to complete it's work
    checkLwdaErrors(lwdaWaitExternalSemaphoresAsync(&m_lwdaWaitSemaphore,
                                                    &waitParams, 1, m_stream));
    // Now step the simulation
    m_sim.stepSimulation(time, m_stream);

    // Signal vulkan to continue with the updated buffers
    checkLwdaErrors(lwdaSignalExternalSemaphoresAsync(
        &m_lwdaSignalSemaphore, &signalParams, 1, m_stream));
  }
};

int main(int argc, char** argv) {
  exelwtion_path = argv[0];
  VulkanLwdaPi app(NUM_SIMULATION_POINTS);
  app.init();
  app.mainLoop();
  return 0;
}
