.. lwTensor documentation master file, created by
   sphinx-quickstart on Mon Aug 26 16:03:12 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

###############################################################
lwTENSOR: A High-Performance LWCA Library For Tensor Primitives
###############################################################

Welcome to the lwTENSOR library documentation.

lwTENSOR is a high-performance LWCA library for tensor primitives.

**Download:** https://developer.lwpu.com/lwtensor/downloads

Key Features
============

  * Extensive mixed-precision support:

    * FP64 inputs with FP32 compute.
    * FP32 inputs with FP16, BF16, or TF32 compute.
    * Complex-times-real operations.
    * Conjugate (without transpose) support.

  * Support for up to 40-dimensional tensors.
  * Arbitrary data layouts.
  * Trivially serializable data structures.
  * Main computational routines:

    * :ref:`Direct (i.e., transpose-free) tensor contractions<contraction-operations-label>`.
    * :ref:`Tensor reductions (including partial reductions)<reduction-operations-label>`.
    * :ref:`Element-wise tensor operations<elementwise-operations-label>`:

       * Support for various activation functions.
       * Arbitrary tensor permutations.
       * Colwersion between different data types.

The documentation consists of three main components:

  * A :ref:`user-guide-label` that introduces important basics of lwTENSOR including details on notation and accuracy.
  * A :ref:`getting-started-label` guide that steps through a simple tensor contraction example.
  * An :ref:`api-reference-label` that provides a comprehensive overview of all library routines, constants, and data types.
  
Support
=======
  * *Supported SM Architectures* : `SM 6.0`, `SM 7.0`, `SM 8.0`
  * *Supported OSs* : `RHEL 7/8`, `openSUSE 15`, `SLES 15`, `Ubuntu 20.04/18.04/16.04`, `Windows 10`
  * *Supported CPU Architectures* : `x86_64`, `ARM64`, `OpenPOWER`
  
Prerequisites
=============
  * *Dependencies* : `lwdart`, `lwtensor.h` headers

  
Contents
========

.. toctree::
   :maxdepth: 2

   release_notes
   user_guide
   getting_started
   plan_cache
   api/index
   license

Indices And Tables
==================

* :ref:`genindex`
* :ref:`search`
