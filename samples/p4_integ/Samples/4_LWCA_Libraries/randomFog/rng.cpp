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

// Utilities and System includes

// Includes
#include <lwrand.h>
#include <stdexcept>
#include <sstream>
#include "rng.h"

// Shared Library Test Functions
#include <helper_timer.h>
#include <helper_lwda.h>

const unsigned int RNG::s_maxQrngDimensions = 20000;

RNG::RNG(unsigned long prngSeed, unsigned int qrngDimensions,
         unsigned int nSamples)
    : m_prngSeed(prngSeed),
      m_qrngDimensions(qrngDimensions),
      m_nSamplesBatchTarget(nSamples),
      m_nSamplesRemaining(0) {
  using std::ilwalid_argument;
  using std::runtime_error;
  using std::string;

  if (m_prngSeed == 0) {
    throw ilwalid_argument("PRNG seed must be non-zero");
  }

  if (m_qrngDimensions == 0) {
    throw ilwalid_argument("QRNG dimensions must be non-zero");
  }

  if (m_nSamplesBatchTarget == 0) {
    throw ilwalid_argument("RNG batch size must be non-zero");
  }

  if (m_nSamplesBatchTarget < s_maxQrngDimensions) {
    throw ilwalid_argument(
        "RNG batch size must be greater than RNG::s_maxQrngDimensions");
  }

  lwrandStatus_t lwrandResult;
  lwdaError_t lwdaResult;

  // Allocate sample array in host mem
  m_h_samples = (float *)malloc(m_nSamplesBatchTarget * sizeof(float));

  if (m_h_samples == NULL) {
    throw runtime_error("Could not allocate host memory for RNG::m_h_samples");
  }

  // Allocate sample array in device mem
  lwdaResult =
      lwdaMalloc((void **)&m_d_samples, m_nSamplesBatchTarget * sizeof(float));

  if (lwdaResult != lwdaSuccess) {
    string msg("Could not allocate device memory for RNG::m_d_samples: ");
    msg += lwdaGetErrorString(lwdaResult);
    throw runtime_error(msg);
  }

  // Create the Random Number Generators
  lwrandResult = lwrandCreateGenerator(&m_prng, LWRAND_RNG_PSEUDO_XORWOW);

  if (lwrandResult != LWRAND_STATUS_SUCCESS) {
    string msg("Could not create pseudo-random number generator: ");
    msg += lwrandResult;
    throw runtime_error(msg);
  }

  lwrandResult = lwrandCreateGenerator(&m_qrng, LWRAND_RNG_QUASI_SOBOL32);

  if (lwrandResult != LWRAND_STATUS_SUCCESS) {
    string msg("Could not create quasi-random number generator: ");
    msg += lwrandResult;
    throw runtime_error(msg);
  }

  lwrandResult =
      lwrandCreateGenerator(&m_sqrng, LWRAND_RNG_QUASI_SCRAMBLED_SOBOL32);

  if (lwrandResult != LWRAND_STATUS_SUCCESS) {
    string msg("Could not create scrambled quasi-random number generator: ");
    msg += lwrandResult;
    throw runtime_error(msg);
  }

  // Setup initial parameters
  resetSeed();
  updateDimensions();
  setBatchSize();

  // Set default RNG to be pseudo-random (XORWOW)
  m_pLwrrent = &m_prng;
}

RNG::~RNG() {
  lwrandDestroyGenerator(m_prng);
  lwrandDestroyGenerator(m_qrng);
  lwrandDestroyGenerator(m_sqrng);

  if (m_d_samples) {
    lwdaFree(m_d_samples);
  }

  if (m_h_samples) {
    free(m_h_samples);
  }
}

void RNG::generateBatch(void) {
  using std::runtime_error;
  using std::string;

  lwdaError_t lwdaResult;
  lwrandStatus_t lwrandResult;

  // Generate random numbers
  lwrandResult =
      lwrandGenerateUniform(*m_pLwrrent, m_d_samples, m_nSamplesBatchActual);

  if (lwrandResult != LWRAND_STATUS_SUCCESS) {
    string msg("Could not generate random numbers: ");
    msg += lwrandResult;
    throw runtime_error(msg);
  }

  // Copy random numbers to host
  lwdaResult =
      lwdaMemcpy(m_h_samples, m_d_samples,
                 m_nSamplesBatchActual * sizeof(float), lwdaMemcpyDeviceToHost);

  if (lwdaResult != lwdaSuccess) {
    string msg("Could not copy random numbers to host: ");
    msg += lwdaGetErrorString(lwdaResult);
    throw runtime_error(msg);
  }
}

float RNG::getNextU01(void) {
  if (m_nSamplesRemaining == 0) {
    generateBatch();
    m_nSamplesRemaining = m_nSamplesBatchActual;
  }

  if (m_pLwrrent == &m_prng) {
    return m_h_samples[m_nSamplesBatchActual - m_nSamplesRemaining--];
  } else {
    unsigned int index = m_nSamplesBatchActual - m_nSamplesRemaining--;
    unsigned int samplesPerDim = m_nSamplesBatchActual / m_qrngDimensions;
    unsigned int dimOffset = (index % m_qrngDimensions) * samplesPerDim;
    unsigned int drawOffset = index / m_qrngDimensions;
    return m_h_samples[dimOffset + drawOffset];
  }
}

void RNG::getInfoString(std::string &msg) {
  using std::stringstream;

  stringstream ss;

  if (m_pLwrrent == &m_prng) {
    ss << "XORWOW (seed=" << m_prngSeed << ")";
  } else if (m_pLwrrent == &m_qrng) {
    ss << "Sobol (dimensions=" << m_qrngDimensions << ")";
  } else if (m_pLwrrent == &m_sqrng) {
    ss << "Scrambled Sobol (dimensions=" << m_qrngDimensions << ")";
  } else {
    ss << "Invalid RNG";
  }

  msg.assign(ss.str());
}

void RNG::selectRng(RNG::RngType type) {
  switch (type) {
    case Quasi:
      m_pLwrrent = &m_qrng;
      break;

    case ScrambledQuasi:
      m_pLwrrent = &m_sqrng;
      break;

    case Pseudo:
    default:
      m_pLwrrent = &m_prng;
      break;
  }

  setBatchSize();
}

void RNG::resetSeed(void) {
  using std::runtime_error;

  lwrandStatus_t lwrandResult;
  lwrandResult = lwrandSetPseudoRandomGeneratorSeed(m_prng, m_prngSeed);

  if (lwrandResult != LWRAND_STATUS_SUCCESS) {
    std::string msg("Could not set pseudo-random number generator seed: ");
    msg += lwrandResult;
    throw runtime_error(msg);
  }

  lwrandResult = lwrandSetGeneratorOffset(m_prng, 0);

  if (lwrandResult != LWRAND_STATUS_SUCCESS) {
    std::string msg("Could not set pseudo-random number generator offset: ");
    msg += lwrandResult;
    throw runtime_error(msg);
  }

  setBatchSize();
}

void RNG::resetDimensions(void) {
  m_qrngDimensions = 1;
  updateDimensions();
  setBatchSize();
}

void RNG::incrementDimensions(void) {
  if (++m_qrngDimensions > s_maxQrngDimensions) {
    m_qrngDimensions = 1;
  }

  updateDimensions();
  setBatchSize();
}

void RNG::updateDimensions(void) {
  using std::runtime_error;

  lwrandStatus_t lwrandResult;
  lwrandResult =
      lwrandSetQuasiRandomGeneratorDimensions(m_qrng, m_qrngDimensions);

  if (lwrandResult != LWRAND_STATUS_SUCCESS) {
    std::string msg("Could not set quasi-random number generator dimensions: ");
    msg += lwrandResult;
    throw runtime_error(msg);
  }

  lwrandResult = lwrandSetGeneratorOffset(m_qrng, 0);

  if (lwrandResult != LWRAND_STATUS_SUCCESS) {
    std::string msg("Could not set quasi-random number generator offset: ");
    msg += lwrandResult;
    throw runtime_error(msg);
  }

  lwrandResult =
      lwrandSetQuasiRandomGeneratorDimensions(m_sqrng, m_qrngDimensions);

  if (lwrandResult != LWRAND_STATUS_SUCCESS) {
    std::string msg(
        "Could not set scrambled quasi-random number generator dimensions: ");
    msg += lwrandResult;
    throw runtime_error(msg);
  }

  lwrandResult = lwrandSetGeneratorOffset(m_sqrng, 0);

  if (lwrandResult != LWRAND_STATUS_SUCCESS) {
    std::string msg(
        "Could not set scrambled quasi-random number generator offset: ");
    msg += lwrandResult;
    throw runtime_error(msg);
  }
}

void RNG::setBatchSize(void) {
  if (m_pLwrrent == &m_prng) {
    m_nSamplesBatchActual = m_nSamplesBatchTarget;
  } else {
    m_nSamplesBatchActual =
        (m_nSamplesBatchTarget / m_qrngDimensions) * m_qrngDimensions;
  }

  m_nSamplesRemaining = 0;
}
