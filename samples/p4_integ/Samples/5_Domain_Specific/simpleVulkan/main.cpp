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

#include "VulkanBaseApp.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include "linmath.h"

#include "SineWaveSimulation.h"

#include <helper_lwda.h>

typedef float vec2[2];
std::string exelwtion_path;

#ifndef NDEBUG
#define ENABLE_VALIDATION (false)
#else
#define ENABLE_VALIDATION (true)
#endif

class VulkanLwdaSineWave : public VulkanBaseApp {
  typedef struct UniformBufferObject_st {
    mat4x4 modelViewProj;
  } UniformBufferObject;

  VkBuffer m_heightBuffer, m_xyBuffer, m_indexBuffer;
  VkDeviceMemory m_heightMemory, m_xyMemory, m_indexMemory;
  UniformBufferObject m_ubo;
  VkSemaphore m_vkWaitSemaphore, m_vkSignalSemaphore;
  SineWaveSimulation m_sim;
  lwdaStream_t m_stream;
  lwdaExternalSemaphore_t m_lwdaWaitSemaphore, m_lwdaSignalSemaphore,
      m_lwdaTimelineSemaphore;
  lwdaExternalMemory_t m_lwdaVertMem;
  float *m_lwdaHeightMap;
  using chrono_tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
  chrono_tp m_lastTime;
  size_t m_lastFrame;

 public:
  VulkanLwdaSineWave(size_t width, size_t height)
      : VulkanBaseApp("vulkanLwdaSineWave", ENABLE_VALIDATION),
        m_heightBuffer(VK_NULL_HANDLE),
        m_xyBuffer(VK_NULL_HANDLE),
        m_indexBuffer(VK_NULL_HANDLE),
        m_heightMemory(VK_NULL_HANDLE),
        m_xyMemory(VK_NULL_HANDLE),
        m_indexMemory(VK_NULL_HANDLE),
        m_ubo(),
        m_sim(width, height),
        m_stream(0),
        m_vkWaitSemaphore(VK_NULL_HANDLE),
        m_vkSignalSemaphore(VK_NULL_HANDLE),
        m_lwdaWaitSemaphore(),
        m_lwdaSignalSemaphore(),
        m_lwdaTimelineSemaphore(),
        m_lwdaVertMem(),
        m_lwdaHeightMap(nullptr),
        m_lastFrame(0) {
    // Our index buffer can only index 32-bits of the vertex buffer
    if ((width * height) > (1ULL << 32ULL)) {
      throw std::runtime_error(
          "Requested height and width is too large for this sample!");
    }
    // Add our compiled vulkan shader files
    char *vertex_shader_path =
        sdkFindFilePath("vert.spv", exelwtion_path.c_str());
    char *fragment_shader_path =
        sdkFindFilePath("frag.spv", exelwtion_path.c_str());
    m_shaderFiles.push_back(
        std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, vertex_shader_path));
    m_shaderFiles.push_back(
        std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_shader_path));
  }
  ~VulkanLwdaSineWave() {
    // Make sure there's no pending work before we start tearing down
    checkLwdaErrors(lwdaStreamSynchronize(m_stream));

#ifdef _VK_TIMELINE_SEMAPHORE
    if (m_vkTimelineSemaphore != VK_NULL_HANDLE) {
      checkLwdaErrors(lwdaDestroyExternalSemaphore(m_lwdaTimelineSemaphore));
      vkDestroySemaphore(m_device, m_vkTimelineSemaphore, nullptr);
    }
#endif /* _VK_TIMELINE_SEMAPHORE */

    if (m_vkSignalSemaphore != VK_NULL_HANDLE) {
      checkLwdaErrors(lwdaDestroyExternalSemaphore(m_lwdaSignalSemaphore));
      vkDestroySemaphore(m_device, m_vkSignalSemaphore, nullptr);
    }
    if (m_vkWaitSemaphore != VK_NULL_HANDLE) {
      checkLwdaErrors(lwdaDestroyExternalSemaphore(m_lwdaWaitSemaphore));
      vkDestroySemaphore(m_device, m_vkWaitSemaphore, nullptr);
    }

    if (m_xyBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_xyBuffer, nullptr);
    }
    if (m_xyMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_xyMemory, nullptr);
    }

    if (m_heightBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_heightBuffer, nullptr);
    }
    if (m_heightMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_heightMemory, nullptr);
    }
    if (m_lwdaHeightMap) {
      checkLwdaErrors(lwdaDestroyExternalMemory(m_lwdaVertMem));
    }

    if (m_indexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_indexBuffer, nullptr);
    }
    if (m_indexMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_indexMemory, nullptr);
    }
  }

  void fillRenderingCommandBuffer(VkCommandBuffer &commandBuffer) {
    VkBuffer vertexBuffers[] = {m_heightBuffer, m_xyBuffer};
    VkDeviceSize offsets[] = {0, 0};
    vkCmdBindVertexBuffers(commandBuffer, 0,
                           sizeof(vertexBuffers) / sizeof(vertexBuffers[0]),
                           vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(commandBuffer, (uint32_t)((m_sim.getWidth() - 1) *
                                               (m_sim.getHeight() - 1) * 6),
                     1, 0, 0, 0);
  }

  void getVertexDescriptions(
      std::vector<VkVertexInputBindingDescription> &bindingDesc,
      std::vector<VkVertexInputAttributeDescription> &attribDesc) {
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

  void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo &info) {
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    info.primitiveRestartEnable = VK_FALSE;
  }

  void getWaitFrameSemaphores(
      std::vector<VkSemaphore> &wait,
      std::vector<VkPipelineStageFlags> &waitStages) const {
    if (m_lwrrentFrame != 0) {
      // Have vulkan wait until lwca is done with the vertex buffer before
      // rendering, We don't do this on the first frame, as the wait semaphore
      // hasn't been initialized yet
      wait.push_back(m_vkWaitSemaphore);
      // We want to wait until all the pipeline commands are complete before
      // letting lwca work
      waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }
  }

  void getSignalFrameSemaphores(std::vector<VkSemaphore> &signal) const {
    // Add this semaphore for vulkan to signal once the vertex buffer is ready
    // for lwca to modify
    signal.push_back(m_vkSignalSemaphore);
  }

  void initVulkanApp() {
    int lwda_device = -1;

    // Select lwca device where vulkan is running.
    lwda_device = m_sim.initLwda(m_vkDeviceUUID, VK_UUID_SIZE);
    if (lwda_device == -1) {
      printf("Error: No LWCA-Vulkan interop capable device found\n");
      exit(EXIT_FAILURE);
    }

    m_sim.initLwdaLaunchConfig(lwda_device);

    // Create the lwca stream we'll be using
    checkLwdaErrors(
        lwdaStreamCreateWithFlags(&m_stream, lwdaStreamNonBlocking));

    const size_t lwerts = m_sim.getWidth() * m_sim.getHeight();
    const size_t nInds = (m_sim.getWidth() - 1) * (m_sim.getHeight() - 1) * 6;

    // Create the height map lwca will write to
    createExternalBuffer(
        lwerts * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, getDefaultMemHandleType(),
        m_heightBuffer, m_heightMemory);

    // Create the vertex buffer that will hold the xy coordinates for the grid
    createBuffer(lwerts * sizeof(vec2), VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_xyBuffer, m_xyMemory);

    // Create the index buffer that references from both buffers above
    createBuffer(
        nInds * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_indexBuffer, m_indexMemory);

    // Import the height map into lwca and retrieve a device pointer to use
    importLwdaExternalMemory((void **)&m_lwdaHeightMap, m_lwdaVertMem,
                             m_heightMemory, lwerts * sizeof(*m_lwdaHeightMap),
                             getDefaultMemHandleType());
    // Set the height map to use in the simulation
    m_sim.initSimulation(m_lwdaHeightMap);

    {
      // Set up the initial values for the vertex buffers with Vulkan
      void *stagingBase;
      VkBuffer stagingBuffer;
      VkDeviceMemory stagingMemory;
      VkDeviceSize stagingSz =
          std::max(lwerts * sizeof(vec2), nInds * sizeof(uint32_t));
      createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   stagingBuffer, stagingMemory);

      vkMapMemory(m_device, stagingMemory, 0, stagingSz, 0, &stagingBase);

      memset(stagingBase, 0, lwerts * sizeof(float));
      copyBuffer(m_heightBuffer, stagingBuffer, lwerts * sizeof(float));

      for (size_t y = 0; y < m_sim.getHeight(); y++) {
        for (size_t x = 0; x < m_sim.getWidth(); x++) {
          vec2 *stagedVert = (vec2 *)stagingBase;
          stagedVert[y * m_sim.getWidth() + x][0] =
              (2.0f * x) / (m_sim.getWidth() - 1) - 1;
          stagedVert[y * m_sim.getWidth() + x][1] =
              (2.0f * y) / (m_sim.getHeight() - 1) - 1;
        }
      }
      copyBuffer(m_xyBuffer, stagingBuffer, lwerts * sizeof(vec2));

      {
        uint32_t *indices = (uint32_t *)stagingBase;
        for (size_t y = 0; y < m_sim.getHeight() - 1; y++) {
          for (size_t x = 0; x < m_sim.getWidth() - 1; x++) {
            indices[0] = (uint32_t)((y + 0) * m_sim.getWidth() + (x + 0));
            indices[1] = (uint32_t)((y + 1) * m_sim.getWidth() + (x + 0));
            indices[2] = (uint32_t)((y + 0) * m_sim.getWidth() + (x + 1));
            indices[3] = (uint32_t)((y + 1) * m_sim.getWidth() + (x + 0));
            indices[4] = (uint32_t)((y + 1) * m_sim.getWidth() + (x + 1));
            indices[5] = (uint32_t)((y + 0) * m_sim.getWidth() + (x + 1));
            indices += 6;
          }
        }
      }
      copyBuffer(m_indexBuffer, stagingBuffer, nInds * sizeof(uint32_t));

      vkUnmapMemory(m_device, stagingMemory);
      vkDestroyBuffer(m_device, stagingBuffer, nullptr);
      vkFreeMemory(m_device, stagingMemory, nullptr);
    }

#ifdef _VK_TIMELINE_SEMAPHORE
    // Create the timeline semaphore to sync lwca and vulkan access to vertex
    // buffer
    createExternalSemaphore(m_vkTimelineSemaphore,
                            getDefaultSemaphoreHandleType());
    // Import the timeline semaphore lwca will use to sync lwca and vulkan
    // access to vertex buffer
    importLwdaExternalSemaphore(m_lwdaTimelineSemaphore, m_vkTimelineSemaphore,
                                getDefaultSemaphoreHandleType());
#else
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
#endif /* _VK_TIMELINE_SEMAPHORE */
  }

  void importLwdaExternalMemory(void **lwdaPtr, lwdaExternalMemory_t &lwdaMem,
                                VkDeviceMemory &vkMem, VkDeviceSize size,
                                VkExternalMemoryHandleTypeFlagBits handleType) {
    lwdaExternalMemoryHandleDesc externalMemoryHandleDesc = {};

    if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
      externalMemoryHandleDesc.type = lwdaExternalMemoryHandleTypeOpaqueWin32;
    } else if (handleType &
               VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
      externalMemoryHandleDesc.type =
          lwdaExternalMemoryHandleTypeOpaqueWin32Kmt;
    } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
      externalMemoryHandleDesc.type = lwdaExternalMemoryHandleTypeOpaqueFd;
    } else {
      throw std::runtime_error("Unknown handle type requested!");
    }

    externalMemoryHandleDesc.size = size;

#ifdef _WIN64
    externalMemoryHandleDesc.handle.win32.handle =
        (HANDLE)getMemHandle(vkMem, handleType);
#else
    externalMemoryHandleDesc.handle.fd =
        (int)(uintptr_t)getMemHandle(vkMem, handleType);
#endif

    checkLwdaErrors(
        lwdaImportExternalMemory(&lwdaMem, &externalMemoryHandleDesc));

    lwdaExternalMemoryBufferDesc externalMemBufferDesc = {};
    externalMemBufferDesc.offset = 0;
    externalMemBufferDesc.size = size;
    externalMemBufferDesc.flags = 0;

    checkLwdaErrors(lwdaExternalMemoryGetMappedBuffer(lwdaPtr, lwdaMem,
                                                      &externalMemBufferDesc));
  }

  void importLwdaExternalSemaphore(
      lwdaExternalSemaphore_t &lwdaSem, VkSemaphore &vkSem,
      VkExternalSemaphoreHandleTypeFlagBits handleType) {
    lwdaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};

#ifdef _VK_TIMELINE_SEMAPHORE
    if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
      externalSemaphoreHandleDesc.type =
          lwdaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    } else if (handleType &
               VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
      externalSemaphoreHandleDesc.type =
          lwdaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    } else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
      externalSemaphoreHandleDesc.type =
          lwdaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    }
#else
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
    }
#endif /* _VK_TIMELINE_SEMAPHORE */
    else {
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

  void updateUniformBuffer(uint32_t imageIndex) {
    {
      mat4x4 view, proj;
      vec3 eye = {1.75f, 1.75f, 1.25f};
      vec3 center = {0.0f, 0.0f, -0.25f};
      vec3 up = {0.0f, 0.0f, 1.0f};

      mat4x4_perspective(
          proj, (float)degreesToRadians(45.0f),
          m_swapChainExtent.width / (float)m_swapChainExtent.height, 0.1f,
          10.0f);
      proj[1][1] *= -1.0f;  // Flip y axis

      mat4x4_look_at(view, eye, center, up);
      mat4x4_mul(m_ubo.modelViewProj, proj, view);
    }

    void *data;
    vkMapMemory(m_device, m_uniformMemory[imageIndex], 0, getUniformSize(), 0,
                &data);
    memcpy(data, &m_ubo, sizeof(m_ubo));
    vkUnmapMemory(m_device, m_uniformMemory[imageIndex]);
  }

  std::vector<const char *> getRequiredExtensions() const {
    std::vector<const char *> extensions;
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
    return extensions;
  }

  std::vector<const char *> getRequiredDeviceExtensions() const {
    std::vector<const char *> extensions;
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
    extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
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

    float frame_time =
        std::chrono::duration<float, std::chrono::seconds::period>(lwrrentTime -
                                                                   m_lastTime)
            .count();

    // Have vulkan draw the current frame...
    VulkanBaseApp::drawFrame();

#ifdef _VK_TIMELINE_SEMAPHORE
    static uint64_t waitValue = 1;
    static uint64_t signalValue = 2;

    lwdaExternalSemaphoreWaitParams waitParams = {};
    waitParams.flags = 0;
    waitParams.params.fence.value = waitValue;

    lwdaExternalSemaphoreSignalParams signalParams = {};
    signalParams.flags = 0;
    signalParams.params.fence.value = signalValue;
    // Wait for vulkan to complete it's work
    checkLwdaErrors(lwdaWaitExternalSemaphoresAsync(&m_lwdaTimelineSemaphore,
                                                    &waitParams, 1, m_stream));
    // Now step the simulation
    m_sim.stepSimulation(time, m_stream);
    // Signal vulkan to continue with the updated buffers
    checkLwdaErrors(lwdaSignalExternalSemaphoresAsync(
        &m_lwdaTimelineSemaphore, &signalParams, 1, m_stream));

    waitValue += 2;
    signalValue += 2;
#else
    lwdaExternalSemaphoreWaitParams waitParams = {};
    waitParams.flags = 0;
    waitParams.params.fence.value = 0;

    lwdaExternalSemaphoreSignalParams signalParams = {};
    signalParams.flags = 0;
    signalParams.params.fence.value = 0;

    // Wait for vulkan to complete it's work
    checkLwdaErrors(lwdaWaitExternalSemaphoresAsync(&m_lwdaWaitSemaphore,
                                                    &waitParams, 1, m_stream));
    // Now step the simulation
    m_sim.stepSimulation(time, m_stream);
    // Signal vulkan to continue with the updated buffers
    checkLwdaErrors(lwdaSignalExternalSemaphoresAsync(
        &m_lwdaSignalSemaphore, &signalParams, 1, m_stream));
#endif /* _VK_TIMELINE_SEMAPHORE */

    // Output a naive measurement of the frames per second every five seconds
    if (frame_time > 5) {
      std::cout << "Average FPS (over " << std::fixed << std::setprecision(2)
                << frame_time << " seconds): " << std::fixed
                << std::setprecision(2)
                << ((m_lwrrentFrame - m_lastFrame) / frame_time) << std::endl;
      m_lastFrame = m_lwrrentFrame;
      m_lastTime = lwrrentTime;
    }
  }
};

int main(int argc, char **argv) {
  exelwtion_path = argv[0];
  VulkanLwdaSineWave app((1ULL << 8ULL), (1ULL << 8ULL));
  app.init();
  app.mainLoop();
  return 0;
}
