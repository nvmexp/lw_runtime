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

#ifndef _RENDERCHECK_GL_H_
#define _RENDERCHECK_GL_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <string>
#include <vector>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <helper_image.h>
#include <lwShaderUtils.h>

using std::map;
using std::string;
using std::vector;

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

#if _DEBUG
#define CHECK_FBO checkStatus(__FILE__, __LINE__, true)
#else
#define CHECK_FBO true
#endif

class CheckRender {
 public:
  CheckRender(unsigned int width, unsigned int height, unsigned int Bpp,
              bool bQAReadback, bool bUseFBO, bool bUsePBO)
      : m_Width(width),
        m_Height(height),
        m_Bpp(Bpp),
        m_bQAReadback(bQAReadback),
        m_bUseFBO(bUseFBO),
        m_bUsePBO(bUsePBO),
        m_PixelFormat(GL_BGRA),
        m_fThresholdCompare(0.0f) {
    allocateMemory(width, height, Bpp, bUseFBO, bUsePBO);
  }

  virtual ~CheckRender() {
    // Release PBO resources
    if (m_bUsePBO) {
      glDeleteBuffers(1, &m_pboReadback);
      m_pboReadback = 0;
    }

    free(m_pImageData);
  }

  virtual void allocateMemory(unsigned int width, unsigned int height,
                              unsigned int Bpp, bool bUseFBO, bool bUsePBO) {
    // Create the PBO for readbacks
    if (bUsePBO) {
      glGenBuffers(1, &m_pboReadback);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_pboReadback);
      glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * Bpp, NULL,
                   GL_STREAM_READ);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    }

    m_pImageData = (unsigned char *)malloc(
        width * height *
        Bpp);  // This is the image data stored in system memory
  }

  virtual void setExecPath(char *path) { m_ExecPath = path; }
  virtual void EnableQAReadback(bool bStatus) { m_bQAReadback = bStatus; }
  virtual bool IsQAReadback() { return m_bQAReadback; }
  virtual bool IsFBO() { return m_bUseFBO; }
  virtual bool IsPBO() { return m_bUsePBO; }
  virtual void *imageData() { return m_pImageData; }

  // Interface to this class functions
  virtual void setPixelFormat(GLenum format) { m_PixelFormat = format; }
  virtual int getPixelFormat() { return m_PixelFormat; }
  virtual bool checkStatus(const char *zfile, int line, bool silent) = 0;
  virtual bool readback(GLuint width, GLuint height) = 0;
  virtual bool readback(GLuint width, GLuint height, GLuint bufObject) = 0;
  virtual bool readback(GLuint width, GLuint height, unsigned char *membuf) = 0;

  virtual void bindReadback() {
    if (!m_bQAReadback) {
      printf("CheckRender::bindReadback() uninitialized!\n");
      return;
    }

    if (m_bUsePBO) {
      glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, m_pboReadback);  // Bind the PBO
    }
  }

  virtual void unbindReadback() {
    if (!m_bQAReadback) {
      printf("CheckRender::unbindReadback() uninitialized!\n");
      return;
    }

    if (m_bUsePBO) {
      glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);  // Release the bind on the PBO
    }
  }

  virtual void savePGM(const char *zfilename, bool bIlwert, void **ppReadBuf) {
    if (zfilename != NULL) {
      if (bIlwert) {
        unsigned char *readBuf;
        unsigned char *writeBuf = (unsigned char *)malloc(m_Width * m_Height);

        for (unsigned int y = 0; y < m_Height; y++) {
          if (ppReadBuf) {
            readBuf = *(unsigned char **)ppReadBuf;
          } else {
            readBuf = (unsigned char *)m_pImageData;
          }

          memcpy(&writeBuf[m_Width * m_Bpp * y],
                 (readBuf + m_Width * (m_Height - 1 - y)), m_Width);
        }

        // we copy the results back to original system buffer
        if (ppReadBuf) {
          memcpy(*ppReadBuf, writeBuf, m_Width * m_Height);
        } else {
          memcpy(m_pImageData, writeBuf, m_Width * m_Height);
        }

        free(writeBuf);
      }

      printf("> Saving PGM: <%s>\n", zfilename);

      if (ppReadBuf) {
        sdkSavePGM<unsigned char>(zfilename, *(unsigned char **)ppReadBuf,
                                  m_Width, m_Height);
      } else {
        sdkSavePGM<unsigned char>(zfilename, (unsigned char *)m_pImageData,
                                  m_Width, m_Height);
      }
    }
  }

  virtual void savePPM(const char *zfilename, bool bIlwert, void **ppReadBuf) {
    if (zfilename != NULL) {
      if (bIlwert) {
        unsigned char *readBuf;
        unsigned char *writeBuf =
            (unsigned char *)malloc(m_Width * m_Height * m_Bpp);

        for (unsigned int y = 0; y < m_Height; y++) {
          if (ppReadBuf) {
            readBuf = *(unsigned char **)ppReadBuf;
          } else {
            readBuf = (unsigned char *)m_pImageData;
          }

          memcpy(&writeBuf[m_Width * m_Bpp * y],
                 (readBuf + m_Width * m_Bpp * (m_Height - 1 - y)),
                 m_Width * m_Bpp);
        }

        // we copy the results back to original system buffer
        if (ppReadBuf) {
          memcpy(*ppReadBuf, writeBuf, m_Width * m_Height * m_Bpp);
        } else {
          memcpy(m_pImageData, writeBuf, m_Width * m_Height * m_Bpp);
        }

        free(writeBuf);
      }

      printf("> Saving PPM: <%s>\n", zfilename);

      if (ppReadBuf) {
        sdkSavePPM4ub(zfilename, *(unsigned char **)ppReadBuf, m_Width,
                      m_Height);
      } else {
        sdkSavePPM4ub(zfilename, (unsigned char *)m_pImageData, m_Width,
                      m_Height);
      }
    }
  }

  virtual bool PGMvsPGM(const char *src_file, const char *ref_file,
                        const float epsilon, const float threshold = 0.0f) {
    unsigned char *src_data = NULL, *ref_data = NULL;
    unsigned long error_count = 0;
    unsigned int width, height;

    char *ref_file_path = sdkFindFilePath(ref_file, m_ExecPath.c_str());

    if (ref_file_path == NULL) {
      printf(
          "CheckRender::PGMvsPGM unable to find <%s> in <%s> Aborting "
          "comparison!\n",
          ref_file, m_ExecPath.c_str());
      printf(">>> Check info.xml and [project//data] folder <%s> <<<\n",
             ref_file);
      printf("Aborting comparison!\n");
      printf("  FAILED\n");
      error_count++;
    } else {
      if (src_file == NULL || ref_file_path == NULL) {
        printf("PGMvsPGM: Aborting comparison\n");
        return false;
      }

      printf("   src_file <%s>\n", src_file);
      printf("   ref_file <%s>\n", ref_file_path);

      if (sdkLoadPPMub(ref_file_path, &ref_data, &width, &height) != true) {
        printf("PGMvsPGM: unable to load ref image file: %s\n", ref_file_path);
        return false;
      }

      if (sdkLoadPPMub(src_file, &src_data, &width, &height) != true) {
        printf("PGMvsPGM: unable to load src image file: %s\n", src_file);
        return false;
      }

      printf(
          "PGMvsPGM: comparing images size (%d,%d) epsilon(%2.4f), "
          "threshold(%4.2f%%)\n",
          m_Height, m_Width, epsilon, threshold * 100);

      if (compareDataAsFloatThreshold<unsigned char, float>(
              ref_data, src_data, m_Height * m_Width, epsilon, threshold) ==
          false) {
        error_count = 1;
      }
    }

    if (error_count == 0) {
      printf("  OK\n");
    } else {
      printf("  FAILURE: %d errors...\n", (unsigned int)error_count);
    }

    return (error_count == 0);  // returns true if all pixels pass
  }

  virtual bool PPMvsPPM(const char *src_file, const char *ref_file,
                        const float epsilon, const float threshold = 0.0f) {
    unsigned long error_count = 0;

    char *ref_file_path = sdkFindFilePath(ref_file, m_ExecPath.c_str());

    if (ref_file_path == NULL) {
      printf(
          "CheckRender::PPMvsPPM unable to find <%s> in <%s> Aborting "
          "comparison!\n",
          ref_file, m_ExecPath.c_str());
      printf(">>> Check info.xml and [project//data] folder <%s> <<<\n",
             ref_file);
      printf("Aborting comparison!\n");
      printf("  FAILED\n");
      error_count++;
    }

    if (src_file == NULL || ref_file_path == NULL) {
      printf("PPMvsPPM: Aborting comparison\n");
      return false;
    }

    printf("   src_file <%s>\n", src_file);
    printf("   ref_file <%s>\n", ref_file_path);
    return (sdkComparePPM(src_file, ref_file_path, epsilon, threshold, true) ==
                    true
                ? true
                : false);
  }

  void setThresholdCompare(float value) { m_fThresholdCompare = value; }

  virtual void dumpBin(void *data, unsigned int bytes, const char *filename) {
    FILE *fp;
    printf("CheckRender::dumpBin: <%s>\n", filename);
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
  }

  virtual bool compareBin2BinUint(const char *src_file, const char *ref_file,
                                  unsigned int nelements, const float epsilon,
                                  const float threshold) {
    unsigned int *src_buffer, *ref_buffer;
    FILE *src_fp = NULL, *ref_fp = NULL;

    unsigned long error_count = 0;
    size_t fsize = 0;

    FOPEN(src_fp, src_file, "rb");

    if (src_fp == NULL) {
      printf("compareBin2Bin <unsigned int> unable to open src_file: %s\n",
             src_file);
      error_count++;
    }

    char *ref_file_path = sdkFindFilePath(ref_file, m_ExecPath.c_str());

    if (ref_file_path == NULL) {
      printf("compareBin2Bin <unsigned int>  unable to find <%s> in <%s>\n",
             ref_file, m_ExecPath.c_str());
      printf(">>> Check info.xml and [project//data] folder <%s> <<<\n",
             ref_file);
      printf("Aborting comparison!\n");
      printf("  FAILED\n");
      error_count++;

      if (src_fp) {
        fclose(src_fp);
      }

      if (ref_fp) {
        fclose(ref_fp);
      }
    } else {
      FOPEN(ref_fp, ref_file_path, "rb");

      if (ref_fp == NULL) {
        printf("compareBin2Bin <unsigned int>  unable to open ref_file: %s\n",
               ref_file_path);
        error_count++;
      }

      if (src_fp && ref_fp) {
        src_buffer = (unsigned int *)malloc(nelements * sizeof(unsigned int));
        ref_buffer = (unsigned int *)malloc(nelements * sizeof(unsigned int));

        fsize = fread(src_buffer, sizeof(unsigned int), nelements, src_fp);

        if (fsize != nelements) {
          printf(
              "compareBin2Bin <unsigned int>  failed to read %u elements from "
              "%s\n",
              nelements, src_file);
          error_count++;
        }

        fsize = fread(ref_buffer, sizeof(unsigned int), nelements, ref_fp);

        if (fsize == 0) {
          printf(
              "compareBin2Bin <unsigned int>  failed to read %u elements from "
              "%s\n",
              nelements, ref_file_path);
          error_count++;
        }

        printf(
            "> compareBin2Bin <unsigned int> nelements=%d, epsilon=%4.2f, "
            "threshold=%4.2f\n",
            nelements, epsilon, threshold);
        printf("   src_file <%s>\n", src_file);
        printf("   ref_file <%s>\n", ref_file_path);

        if (!compareData<unsigned int, float>(ref_buffer, src_buffer, nelements,
                                              epsilon, threshold)) {
          error_count++;
        }

        fclose(src_fp);
        fclose(ref_fp);

        free(src_buffer);
        free(ref_buffer);
      } else {
        if (src_fp) {
          fclose(src_fp);
        }

        if (ref_fp) {
          fclose(ref_fp);
        }
      }
    }

    if (error_count == 0) {
      printf("  OK\n");
    } else {
      printf("  FAILURE: %d errors...\n", (unsigned int)error_count);
    }

    return (error_count == 0);  // returns true if all pixels pass
  }

  virtual bool compareBin2BinFloat(const char *src_file, const char *ref_file,
                                   unsigned int nelements, const float epsilon,
                                   const float threshold) {
    float *src_buffer, *ref_buffer;
    FILE *src_fp = NULL, *ref_fp = NULL;
    size_t fsize = 0;

    unsigned long error_count = 0;

    FOPEN(src_fp, src_file, "rb");

    if (src_fp == NULL) {
      printf("compareBin2Bin <float> unable to open src_file: %s\n", src_file);
      error_count = 1;
    }

    char *ref_file_path = sdkFindFilePath(ref_file, m_ExecPath.c_str());

    if (ref_file_path == NULL) {
      printf("compareBin2Bin <float> unable to find <%s> in <%s>\n", ref_file,
             m_ExecPath.c_str());
      printf(">>> Check info.xml and [project//data] folder <%s> <<<\n",
             m_ExecPath.c_str());
      printf("Aborting comparison!\n");
      printf("  FAILED\n");
      error_count++;

      if (src_fp) {
        fclose(src_fp);
      }

      if (ref_fp) {
        fclose(ref_fp);
      }
    } else {
      FOPEN(ref_fp, ref_file_path, "rb");

      if (ref_fp == NULL) {
        printf("compareBin2Bin <float> unable to open ref_file: %s\n",
               ref_file_path);
        error_count = 1;
      }

      if (src_fp && ref_fp) {
        src_buffer = (float *)malloc(nelements * sizeof(float));
        ref_buffer = (float *)malloc(nelements * sizeof(float));

        fsize = fread(src_buffer, sizeof(float), nelements, src_fp);

        if (fsize != nelements) {
          printf("compareBin2Bin <float>  failed to read %u elements from %s\n",
                 nelements, src_file);
          error_count++;
        }

        fsize = fread(ref_buffer, sizeof(float), nelements, ref_fp);

        if (fsize == 0) {
          printf("compareBin2Bin <float>  failed to read %u elements from %s\n",
                 nelements, ref_file_path);
          error_count++;
        }

        printf(
            "> compareBin2Bin <float> nelements=%d, epsilon=%4.2f, "
            "threshold=%4.2f\n",
            nelements, epsilon, threshold);
        printf("   src_file <%s>\n", src_file);
        printf("   ref_file <%s>\n", ref_file_path);

        if (!compareDataAsFloatThreshold<float, float>(
                ref_buffer, src_buffer, nelements, epsilon, threshold)) {
          error_count++;
        }

        fclose(src_fp);
        fclose(ref_fp);

        free(src_buffer);
        free(ref_buffer);
      } else {
        if (src_fp) {
          fclose(src_fp);
        }

        if (ref_fp) {
          fclose(ref_fp);
        }
      }
    }

    if (error_count == 0) {
      printf("  OK\n");
    } else {
      printf("  FAILURE: %d errors...\n", (unsigned int)error_count);
    }

    return (error_count == 0);  // returns true if all pixels pass
  }

 protected:
  unsigned int m_Width, m_Height, m_Bpp;
  unsigned char
      *m_pImageData;  // This is the image data stored in system memory
  bool m_bQAReadback, m_bUseFBO, m_bUsePBO;
  GLuint m_pboReadback;
  GLenum m_PixelFormat;
  float m_fThresholdCompare;
  string m_ExecPath;
};

class CheckBackBuffer : public CheckRender {
 public:
  CheckBackBuffer(unsigned int width, unsigned int height, unsigned int Bpp,
                  bool bUseOpenGL = true)
      : CheckRender(width, height, Bpp, false, false, bUseOpenGL) {}

  virtual ~CheckBackBuffer() {}

  virtual bool checkStatus(const char *zfile, int line, bool silent) {
    GLenum nErrorCode = glGetError();

    if (nErrorCode != GL_NO_ERROR) {
      if (!silent) {
        printf("Assertion failed(%s,%d): %s\n", zfile, line,
               gluErrorString(nErrorCode));
      }
    }

    return true;
  }

  virtual bool readback(GLuint width, GLuint height) {
    bool ret = false;

    if (m_bUsePBO) {
      // binds the PBO for readback
      bindReadback();

      // Initiate the readback BLT from BackBuffer->PBO->membuf
      glReadPixels(0, 0, width, height, getPixelFormat(), GL_UNSIGNED_BYTE,
                   BUFFER_OFFSET(0));
      ret = checkStatus(__FILE__, __LINE__, true);

      if (!ret) {
        printf("CheckBackBuffer::glReadPixels() checkStatus = %d\n", ret);
      }

      // map - unmap simulates readback without the copy
      void *ioMem = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
      memcpy(m_pImageData, ioMem, width * height * m_Bpp);

      glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);

      // release the PBO
      unbindReadback();
    } else {
      // reading direct from the backbuffer
      glReadBuffer(GL_FRONT);
      glReadPixels(0, 0, width, height, getPixelFormat(), GL_UNSIGNED_BYTE,
                   m_pImageData);
    }

    return ret;
  }

  virtual bool readback(GLuint width, GLuint height, GLuint bufObject) {
    bool ret = false;

    if (m_bUseFBO) {
      if (m_bUsePBO) {
        printf("CheckBackBuffer::readback() FBO->PBO->m_pImageData\n");
        // binds the PBO for readback
        bindReadback();

        // bind FBO buffer (we want to transfer FBO -> PBO)
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, bufObject);

        // Now initiate the readback to PBO
        glReadPixels(0, 0, width, height, getPixelFormat(), GL_UNSIGNED_BYTE,
                     BUFFER_OFFSET(0));
        ret = checkStatus(__FILE__, __LINE__, true);

        if (!ret) {
          printf("CheckBackBuffer::readback() FBO->PBO checkStatus = %d\n",
                 ret);
        }

        // map - unmap simulates readback without the copy
        void *ioMem = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
        memcpy(m_pImageData, ioMem, width * height * m_Bpp);

        glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);

        // release the FBO
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

        // release the PBO
        unbindReadback();
      } else {
        printf("CheckBackBuffer::readback() FBO->m_pImageData\n");
        // Reading direct to FBO using glReadPixels
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, bufObject);
        ret = checkStatus(__FILE__, __LINE__, true);

        if (!ret) {
          printf(
              "CheckBackBuffer::readback::glBindFramebufferEXT() fbo=%d "
              "checkStatus = %d\n",
              bufObject, ret);
        }

        glReadBuffer(static_cast<GLenum>(GL_COLOR_ATTACHMENT0_EXT));
        ret &= checkStatus(__FILE__, __LINE__, true);

        if (!ret) {
          printf(
              "CheckBackBuffer::readback::glReadBuffer() fbo=%d checkStatus = "
              "%d\n",
              bufObject, ret);
        }

        glReadPixels(0, 0, width, height, getPixelFormat(), GL_UNSIGNED_BYTE,
                     m_pImageData);

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
      }
    } else {
      printf("CheckBackBuffer::readback() PBO->m_pImageData\n");
      // read from bufObject (PBO) to system memorys image
      glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, bufObject);  // Bind the PBO

      // map - unmap simulates readback without the copy
      void *ioMem = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);

      // allocate a buffer so we can flip the image
      unsigned char *temp_buf = (unsigned char *)malloc(width * height * m_Bpp);
      memcpy(temp_buf, ioMem, width * height * m_Bpp);

      // let's flip the image as we copy
      for (unsigned int y = 0; y < height; y++) {
        memcpy((void *)&(m_pImageData[(height - y) * width * m_Bpp]),
               (void *)&(temp_buf[y * width * m_Bpp]), width * m_Bpp);
      }

      free(temp_buf);

      glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);

      // read from bufObject (PBO) to system memory image
      glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);  // unBind the PBO
    }

    return CHECK_FBO;
  }

  virtual bool readback(GLuint width, GLuint height, unsigned char *memBuf) {
    // let's flip the image as we copy
    for (unsigned int y = 0; y < height; y++) {
      memcpy((void *)&(m_pImageData[(height - y) * width * m_Bpp]),
             (void *)&(memBuf[y * width * m_Bpp]), width * m_Bpp);
    }

    return true;
  }

 private:
  virtual void bindFragmentProgram(){};
  virtual void bindRenderPath(){};
  virtual void unbindRenderPath(){};

  // bind to the BackBuffer to Texture
  virtual void bindTexture(){};

  // release this bind
  virtual void unbindTexture(){};
};

// structure defining the properties of a single buffer
struct bufferConfig {
  string name;
  GLenum format;
  int bits;
};

// structures defining properties of an FBO
struct fboConfig {
  string name;
  GLenum colorFormat;
  GLenum depthFormat;
  int redbits;
  int depthBits;
  int depthSamples;
  int coverageSamples;
};

struct fboData {
  GLuint colorTex;   // color texture
  GLuint depthTex;   // depth texture
  GLuint fb;         // render framebuffer
  GLuint resolveFB;  // multisample resolve target
  GLuint colorRB;    // color render buffer
  GLuint depthRB;    // depth render buffer
};

class CFrameBufferObject {
 public:
  CFrameBufferObject(unsigned int width, unsigned int height, unsigned int Bpp,
                     bool bUseFloat, GLenum eTarget)
      : m_Width(width),
        m_Height(height),
        m_bUseFloat(bUseFloat),
        m_eGLTarget(eTarget) {
    glGenFramebuffersEXT(1, &m_fboData.fb);

    m_fboData.colorTex =
        createTexture(m_eGLTarget, width, height,
                      (bUseFloat ? GL_RGBA32F_ARB : GL_RGBA8), GL_RGBA);

    m_fboData.depthTex = createTexture(
        m_eGLTarget, width, height, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT);

    attachTexture(m_eGLTarget, m_fboData.colorTex, GL_COLOR_ATTACHMENT0_EXT);
    attachTexture(m_eGLTarget, m_fboData.depthTex, GL_DEPTH_ATTACHMENT_EXT);

    //    bool ret = checkStatus(__FILE__, __LINE__, true);
  }

  virtual ~CFrameBufferObject() {
    //   freeResources();
  }

  GLuint createTexture(GLenum target, int w, int h, GLint internalformat,
                       GLenum format) {
    GLuint texid;
    glGenTextures(1, &texid);
    glBindTexture(target, texid);

    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_FLOAT, 0);
    return texid;
  }

  void attachTexture(GLenum texTarget, GLuint texId,
                     GLenum attachment = GL_COLOR_ATTACHMENT0_EXT,
                     int mipLevel = 0, int zSlice = 0) {
    bindRenderPath();

    switch (texTarget) {
      case GL_TEXTURE_1D:
        glFramebufferTexture1DEXT(GL_FRAMEBUFFER_EXT, attachment, GL_TEXTURE_1D,
                                  texId, mipLevel);
        break;

      case GL_TEXTURE_3D:
        glFramebufferTexture3DEXT(GL_FRAMEBUFFER_EXT, attachment, GL_TEXTURE_3D,
                                  texId, mipLevel, zSlice);
        break;

      default:
        // Default is GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_ARB, or lwbe faces
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, attachment, texTarget,
                                  texId, mipLevel);
        break;
    }

    unbindRenderPath();
  }

  bool initialize(unsigned width, unsigned height, fboConfig &rConfigFBO,
                  fboData &rActiveFBO) {
    // Framebuffer config options
    vector<bufferConfig> colorConfigs;
    vector<bufferConfig> depthConfigs;
    bufferConfig temp;

    // add default color configs
    temp.name = (m_bUseFloat ? "RGBA32F" : "RGBA8");
    temp.bits = (m_bUseFloat ? 32 : 8);
    temp.format = (m_bUseFloat ? GL_RGBA32F_ARB : GL_RGBA8);
    colorConfigs.push_back(temp);

    // add default depth configs
    temp.name = "D24";
    temp.bits = 24;
    temp.format = GL_DEPTH_COMPONENT24;
    depthConfigs.push_back(temp);

    // If the FBO can be created, add it to the list of available configs, and
    // make a menu entry
    string root = colorConfigs[0].name + " " + depthConfigs[0].name;

    rConfigFBO.colorFormat = colorConfigs[0].format;
    rConfigFBO.depthFormat = depthConfigs[0].format;
    rConfigFBO.redbits = colorConfigs[0].bits;
    rConfigFBO.depthBits = depthConfigs[0].bits;

    // single sample
    rConfigFBO.name = root;
    rConfigFBO.coverageSamples = 0;
    rConfigFBO.depthSamples = 0;

    create(width, height, rConfigFBO, rActiveFBO);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    if (m_bUseFloat) {
      // load fragment programs
      const char *strTextureProgram2D =
          "!!ARBfp1.0\n"
          "TEX result.color, fragment.texcoord[0], texture[0], 2D;\n"
          "END\n";

      m_textureProgram =
          lw::CompileASMShader(GL_FRAGMENT_PROGRAM_ARB, strTextureProgram2D);

      const char *strOverlayProgram =
          "!!ARBfp1.0\n"
          "TEMP t;\n"
          "TEX t, fragment.texcoord[0], texture[0], 2D;\n"
          "MOV result.color, t;\n"
          "END\n";

      m_overlayProgram =
          lw::CompileASMShader(GL_FRAGMENT_PROGRAM_ARB, strOverlayProgram);
    }

    return CHECK_FBO;
  }

  bool create(GLuint width, GLuint height, fboConfig &config, fboData &data) {
    bool multisample = config.depthSamples > 0;
    bool ret = true;
    GLint query;

    printf("\nCreating FBO <%s> (%dx%d) Float:%s\n", config.name.c_str(),
           (int)width, (int)height, (m_bUseFloat ? "Y" : "N"));

    glGenFramebuffersEXT(1, &data.fb);
    glGenTextures(1, &data.colorTex);

    // init texture
    glBindTexture(m_eGLTarget, data.colorTex);
    glTexImage2D(m_eGLTarget, 0, config.colorFormat, width, height, 0, GL_RGBA,
                 (m_bUseFloat ? GL_FLOAT : GL_UNSIGNED_BYTE), NULL);

    glGenerateMipmapEXT(m_eGLTarget);

    glTexParameterf(m_eGLTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(m_eGLTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(m_eGLTarget, GL_TEXTURE_MIN_FILTER,
                    GL_NEAREST);  // GL_LINEAR_MIPMAP_LINEAR);
    glTexParameterf(m_eGLTarget, GL_TEXTURE_MAG_FILTER,
                    GL_NEAREST);  // GL_LINEAR);

    {
      glGenTextures(1, &data.depthTex);
      data.depthRB = 0;
      data.colorRB = 0;
      data.resolveFB = 0;

      // non-multisample, so bind things directly to the FBO
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, data.fb);
      glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                                m_eGLTarget, data.colorTex, 0);

      glBindTexture(m_eGLTarget, data.depthTex);
      glTexImage2D(m_eGLTarget, 0, config.depthFormat, width, height, 0,
                   GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

      glTexParameterf(m_eGLTarget, GL_TEXTURE_MIN_FILTER,
                      GL_NEAREST);  // GL_LINEAR);
      glTexParameterf(m_eGLTarget, GL_TEXTURE_MAG_FILTER,
                      GL_NEAREST);  // GL_LINEAR);
      glTexParameterf(m_eGLTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameterf(m_eGLTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameterf(m_eGLTarget, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);

      glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                                m_eGLTarget, data.depthTex, 0);

      ret &= checkStatus(__FILE__, __LINE__, true);
    }

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, data.fb);
    glGetIntegerv(GL_RED_BITS, &query);

    if (query != config.redbits) {
      ret = false;
    }

    glGetIntegerv(GL_DEPTH_BITS, &query);

    if (query != config.depthBits) {
      ret = false;
    }

    if (multisample) {
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, data.resolveFB);
      glGetIntegerv(GL_RED_BITS, &query);

      if (query != config.redbits) {
        ret = false;
      }
    }

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    ret &= checkStatus(__FILE__, __LINE__, true);

    return ret;
  }

  virtual void freeResources() {
    if (m_fboData.fb) {
      glDeleteFramebuffersEXT(1, &m_fboData.fb);
    }

    if (m_fboData.resolveFB) {
      glDeleteFramebuffersEXT(1, &m_fboData.resolveFB);
    }

    if (m_fboData.colorRB) {
      glDeleteRenderbuffersEXT(1, &m_fboData.colorRB);
    }

    if (m_fboData.depthRB) {
      glDeleteRenderbuffersEXT(1, &m_fboData.depthRB);
    }

    if (m_fboData.colorTex) {
      glDeleteTextures(1, &m_fboData.colorTex);
    }

    if (m_fboData.depthTex) {
      glDeleteTextures(1, &m_fboData.depthTex);
    }

    glDeleteProgramsARB(1, &m_textureProgram);
    glDeleteProgramsARB(1, &m_overlayProgram);
  }

  virtual bool checkStatus(const char *zfile, int line, bool silent) {
    GLenum status;
    status = (GLenum)glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

    if (status != GL_FRAMEBUFFER_COMPLETE_EXT) {
      printf("<%s : %d> - ", zfile, line);
    }

    switch (status) {
      case GL_FRAMEBUFFER_COMPLETE_EXT:
        break;

      case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
        if (!silent) {
          printf("Unsupported framebuffer format\n");
        }

        return false;

      case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
        if (!silent) {
          printf("Framebuffer incomplete, missing attachment\n");
        }

        return false;

      case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
        if (!silent) {
          printf("Framebuffer incomplete, duplicate attachment\n");
        }

        return false;

      case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
        if (!silent) {
          printf(
              "Framebuffer incomplete, attached images must have same "
              "dimensions\n");
        }

        return false;

      case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
        if (!silent) {
          printf(
              "Framebuffer incomplete, attached images must have same "
              "format\n");
        }

        return false;

      case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
        if (!silent) {
          printf("Framebuffer incomplete, missing draw buffer\n");
        }

        return false;

      case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
        if (!silent) {
          printf("Framebuffer incomplete, missing read buffer\n");
        }

        return false;

      default:
        assert(0);
        return false;
    }

    return true;
  }

  virtual void renderQuad(int width, int height, GLenum eTarget) {
    float width_norm = (float)width / (float)m_Width,
          height_norm = (float)height / (float)m_Height;

    // Bind the FBO texture for the display path
    glBindTexture(eTarget, m_fboData.colorTex);

    glGenerateMipmapEXT(GL_TEXTURE_2D);
    glBindTexture(eTarget, 0);

    // now render to the full screen using this texture
    glClearColor(0.2f, 0.2f, 0.2f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, m_textureProgram);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    {
      glVertex2f(0.0f, 0.0f);
      glTexCoord2f(0.0f, 0.0f);
      glVertex2f(0.0f, height_norm);
      glTexCoord2f(width_norm, 0.0f);
      glVertex2f(width_norm, height_norm);
      glTexCoord2f(width_norm, height_norm);
      glVertex2f(width_norm, 0.0f);
      glTexCoord2f(0.0f, height_norm);
    }
    glEnd();

    // Release the FBO texture (finished rendering)
    glBindTexture(eTarget, 0);
  }

  // bind to the Fragment Program
  void bindFragmentProgram() {
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, m_textureProgram);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
  }

  // bind to the FrameBuffer Object
  void bindRenderPath() {
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fboData.fb);
  }

  // release current FrameBuffer Object
  void unbindRenderPath() { glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0); }

  // bind to the FBO to Texture
  void bindTexture() { glBindTexture(m_eGLTarget, m_fboData.colorTex); }

  // release this bind
  void unbindTexture() { glBindTexture(m_eGLTarget, 0); }

  GLuint getFbo() { return m_fboData.fb; }
  GLuint getTex() { return m_fboData.colorTex; }
  GLuint getDepthTex() { return m_fboData.depthTex; }

 private:
  GLuint m_Width, m_Height;
  fboData m_fboData;
  fboConfig m_fboConfig;

  GLuint m_textureProgram;
  GLuint m_overlayProgram;

  bool m_bUseFloat;
  GLenum m_eGLTarget;
};

// CheckFBO - render and verify contents of the FBO
class CheckFBO : public CheckRender {
 public:
  CheckFBO(unsigned int width, unsigned int height, unsigned int Bpp)
      : CheckRender(width, height, Bpp, false, false, true),
        m_pFrameBufferObject(NULL) {}

  CheckFBO(unsigned int width, unsigned int height, unsigned int Bpp,
           CFrameBufferObject *pFrameBufferObject)
      : CheckRender(width, height, Bpp, false, true, true),
        m_pFrameBufferObject(pFrameBufferObject) {}

  virtual ~CheckFBO() {}

  virtual bool checkStatus(const char *zfile, int line, bool silent) {
    GLenum status;
    status = (GLenum)glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

    if (status != GL_FRAMEBUFFER_COMPLETE_EXT) {
      printf("<%s : %d> - ", zfile, line);
    }

    switch (status) {
      case GL_FRAMEBUFFER_COMPLETE_EXT:
        break;

      case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
        if (!silent) {
          printf("Unsupported framebuffer format\n");
        }

        return false;

      case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
        if (!silent) {
          printf("Framebuffer incomplete, missing attachment\n");
        }

        return false;

      case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
        if (!silent) {
          printf("Framebuffer incomplete, duplicate attachment\n");
        }

        return false;

      case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
        if (!silent) {
          printf(
              "Framebuffer incomplete, attached images must have same "
              "dimensions\n");
        }

        return false;

      case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
        if (!silent) {
          printf(
              "Framebuffer incomplete, attached images must have same "
              "format\n");
        }

        return false;

      case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
        if (!silent) {
          printf("Framebuffer incomplete, missing draw buffer\n");
        }

        return false;

      case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
        if (!silent) {
          printf("Framebuffer incomplete, missing read buffer\n");
        }

        return false;

      default:
        assert(0);
        return false;
    }

    return true;
  }

  virtual bool readback(GLuint width, GLuint height) {
    bool ret = false;

    if (m_bUsePBO) {
      // binds the PBO for readback
      bindReadback();

      // bind FBO buffer (we want to transfer FBO -> PBO)
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_pFrameBufferObject->getFbo());

      // Now initiate the readback to PBO
      glReadPixels(0, 0, width, height, getPixelFormat(), GL_UNSIGNED_BYTE,
                   BUFFER_OFFSET(0));
      ret = checkStatus(__FILE__, __LINE__, true);

      if (!ret) {
        printf("CheckFBO::readback() FBO->PBO checkStatus = %d\n", ret);
      }

      // map - unmap simulates readback without the copy
      void *ioMem = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
      memcpy(m_pImageData, ioMem, width * height * m_Bpp);

      glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);

      // release the FBO
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

      // release the PBO
      unbindReadback();
    } else {
      // Reading back from FBO using glReadPixels
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_pFrameBufferObject->getFbo());
      ret = checkStatus(__FILE__, __LINE__, true);

      if (!ret) {
        printf("CheckFBO::readback::glBindFramebufferEXT() checkStatus = %d\n",
               ret);
      }

      glReadBuffer(static_cast<GLenum>(GL_COLOR_ATTACHMENT0_EXT));
      ret &= checkStatus(__FILE__, __LINE__, true);

      if (!ret) {
        printf("CheckFBO::readback::glReadBuffer() checkStatus = %d\n", ret);
      }

      glReadPixels(0, 0, width, height, getPixelFormat(), GL_UNSIGNED_BYTE,
                   m_pImageData);

      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    }

    return CHECK_FBO;
  }

  virtual bool readback(GLuint width, GLuint height, GLuint bufObject) {
    bool ret = false;

    if (m_bUseFBO) {
      if (m_bUsePBO) {
        printf("CheckFBO::readback() FBO->PBO->m_pImageData\n");
        // binds the PBO for readback
        bindReadback();

        // bind FBO buffer (we want to transfer FBO -> PBO)
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, bufObject);

        // Now initiate the readback to PBO
        glReadPixels(0, 0, width, height, getPixelFormat(), GL_UNSIGNED_BYTE,
                     BUFFER_OFFSET(0));
        ret = checkStatus(__FILE__, __LINE__, true);

        if (!ret) {
          printf("CheckFBO::readback() FBO->PBO checkStatus = %d\n", ret);
        }

        // map - unmap simulates readback without the copy
        void *ioMem = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
        memcpy(m_pImageData, ioMem, width * height * m_Bpp);

        glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);

        // release the FBO
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

        // release the PBO
        unbindReadback();
      } else {
        printf("CheckFBO::readback() FBO->m_pImageData\n");
        // Reading direct to FBO using glReadPixels
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, bufObject);
        ret = checkStatus(__FILE__, __LINE__, true);

        if (!ret) {
          printf(
              "CheckFBO::readback::glBindFramebufferEXT() fbo=%d checkStatus = "
              "%d\n",
              (int)bufObject, (int)ret);
        }

        glReadBuffer(static_cast<GLenum>(GL_COLOR_ATTACHMENT0_EXT));
        ret &= checkStatus(__FILE__, __LINE__, true);

        if (!ret) {
          printf("CheckFBO::readback::glReadBuffer() fbo=%d checkStatus = %d\n",
                 (int)bufObject, (int)ret);
        }

        glReadPixels(0, 0, width, height, getPixelFormat(), GL_UNSIGNED_BYTE,
                     m_pImageData);

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
      }
    } else {
      printf("CheckFBO::readback() PBO->m_pImageData\n");
      // read from bufObject (PBO) to system memorys image
      glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, bufObject);  // Bind the PBO

      // map - unmap simulates readback without the copy
      void *ioMem = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);
      memcpy(m_pImageData, ioMem, width * height * m_Bpp);

      glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);

      // read from bufObject (PBO) to system memory image
      glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);  // unBind the PBO
    }

    return CHECK_FBO;
  }

  virtual bool readback(GLuint width, GLuint height, unsigned char *memBuf) {
    // let's flip the image as we copy
    for (unsigned int y = 0; y < height; y++) {
      memcpy((void *)&(m_pImageData[(height - y) * width * m_Bpp]),
             (void *)&(memBuf[y * width * m_Bpp]), width * m_Bpp);
    }

    return true;
  }

 private:
  CFrameBufferObject *m_pFrameBufferObject;
};

#endif  // _RENDERCHECK_GL_H_
