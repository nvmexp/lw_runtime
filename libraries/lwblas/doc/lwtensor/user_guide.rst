
.. _user-guide-label:

User Guide
==========

.. _nomenclature-label:

Nomenclature
------------

The term tensor refers to an **order-n** (a.k.a., n-dimensional) array. 
One can think of tensors as a generalization of matrices to higher **orders**.
For example, scalars, vectors, and matrices are order-0, order-1, and order-2 
tensors, respectively.

An order-n tensor has :math:`n` **modes**. Each mode has an **extent** (a.k.a. size).
For each mode you can specify a **stride** :math:`s > 0`. This **stride**
describes how far apart two logically conselwtive elements along that mode are
in physical memory. They have a function similar to the leading dimension in
BLAS and allow, for example, operating on sub-tensors.

lwTENSOR, by default, adheres to a generalized **column-major** data layout.
For example: :math:`A_{a,b,c} \in {R}^{4\times 8 \times 12}`
is an order-3 tensor with the extent of the a-mode, b-mode, and c-mode
respectively being 4, 8, and 12. If not explicitly specified, the strides are
assumed to be: 
  * :math:`stride(a) = 1`
  * :math:`stride(b) = extent(a)`
  * :math:`stride(c) = extent(a) * extent(b)`.

For a general order-n tensor :math:`A_{i_1,i_2,...,i_n}` we require that the strides do
not lead to overlapping memory accesses; for instance, :math:`stride(i_1) >= 1`, and
:math:`stride(i_{l}) >=stride(i_{l-1}) * extent(i_{l-1})`.

We say that a tensor is **packed** if it is contiguously stored in memory along all
modes. That is, :math:`stride(i_1) = 1` and :math:`stride(i_l) = stride(i_{l-1}) *
extent(i_{l-1})`.

.. _einstein-notation-label:

Einstein Notation
-----------------

We adhere to the "`Einstein notation <https://en.wikipedia.org/wiki/Einstein_notation>`_": modes that appear in the input
tensors and not in the output tensor are implicitly contracted.

.. _performance-guidlines-label:

Performance Guidelines
----------------------

In this section we assume a generalized column-major data layout (i.e., the modes on the
left have the smallest stride). Most of the following performance guidelines are aimed
to facilitate more regular memory access patterns:

  * Try to arrange the modes (w.r.t. increasing strides) of the tensor similarly in all tensors. For instance, :math:`C_{a,b,c} = A_{a,k,c} B_{k,b}` is preferable to :math:`C_{a,b,c} = A_{c,k,a} B_{k,b}`.
  * Try to keep batched modes as the slowest-varying modes (i.e., with the largest strides).  For instance :math:`C_{a,b,c,l} = A_{a,k,c,l} B_{k,b,l}` is preferable to :math:`C_{a,l,b,c} = A_{l,a,k,c} B_{k,l,b}`.
  * Try to keep the extent of the fastest-varying mode (a.k.a. stride-one mode) as large as possible.

.. _plan-cache-overview-label:

Software-managed Plan Cache (beta)
---------------------------------

This section introduces the *software-managed plan cache*, it's key features are:

  * Minimize launch-related overhead (e.g., due to kernel selection)
  * Overhead-free autotuning (a.k.a. *incremental autotuning*)
    * This feature enables users to automatically find the best implementation for the
      given problem and thereby increasing the attained performance
  * The cache is implemented in a thread-safe manner and it's shared across all threads that use the same lwtensorHandle_t.
  * Store/read to/from file
    * Allows users to store the state of the cache to disc and reuse it at a later stage

In essence, the *plan cache* can be seen as a lookup table from a specific problem
instance (e.g., lwtensorContractionDescriptor_t) to an actual implementation (encoded by
lwtensorContractionPlan_t).

The plan cache is an experimental feature at this point -- future changes to the API are
possible.

Please refer to :ref:`plan-cache-label` for a detailed description.

.. _aclwracy-guarantees-label:

Accuracy Guarantees
-------------------

lwTENSOR uses its own compute type to set the floating-point accuracy across tensor operations.
The :ref:`lwtensorComputeType-label` refers to the minimal accuracy that is guaranteed throughout
computations. Because it is only a guarantee of minimal accuracy, it is possible that the library
chooses a higher accuracy than that requested by the user (e.g., if that compute type is not 
supported for a given problem, or to have more kernels available to choose from).

For instance, let us consider a tensor contraction for which all tensors are of type
`LWDA_R_32F` but the :ref:`lwtensorComputeType-label` is `LWTENSOR_COMPUTE_16F`, in that
case lwTENSOR would use Lwpu's Tensor Cores with an aclwmulation type of `LWDA_R_32F`
(i.e., providing higher precision than requested by the user). 

Another illustrative example is a tensor contraction for which all tensors are of type
`LWDA_R_16F` and the computeType is `LWDA_R_MIN_32F`: In this case the parallel reduction
(if required for performance) would have to be performed in `LWDA_R_32F` and thus require
auxiliary workspace. To be precise, in this case lwTENSOR would not choose a serial
reduction --via atomics-- through the output tensor since part of the final reduction
would be performed in `LWDA_R_16F`, which is lower than the :ref:`lwtensorComputeType-label`
requested by the user.

lwTENSOR follows the BLAS convention for NaN propagation: Whenever a scalar (`alpha`, `beta`, `gamma`)
is set to zero, NaN in the scaled tensor expression are ignored, i.e. a zero from a scalar
has precedence over a NaN from a tensor. However, NaN from a tensor follows normal IEEE 754 behavior.

To illustrate, let :math:`\alpha = 1; \beta = 0; A_{i, j} = 1; A'_{i, j} = 0; B_{i, j} = \textrm{NaN}`,
then :math:`\alpha A_{i,j} B_{i, j} = \textrm{NaN}`, :math:`\beta A_{i, j} B_{i, j} = 0`,
:math:`\alpha A'_{i,j} B_{i, j} = \textrm{NaN}`, and :math:`\beta A'_{i, j} B_{i, j} = 0`.

.. _scalar-types-label:

Scalar Types
-------------

Many operations support multiplication of arguments by a scalar.
The type of that scalar is a function of the output type and the compute type.
The following table lists the corresponding types:

.. list-table::
  :header-rows: 1
  :align: center

  * - Output type
    - :ref:`lwtensorComputeType-label`
    - Scalar type
  * - `LWDA_R_16F` or `LWDA_R_16BF` or `LWDA_R_32F`
    - `LWTENSOR_COMPUTE_16F` or `LWTENSOR_COMPUTE_16BF` or `LWTENSOR_COMPUTE_TF32` or `LWTENSOR_COMPUTE_32F`
    - `LWDA_R_32F`

  * - `LWDA_R_64F`
    - `LWTENSOR_COMPUTE_32F` or `LWTENSOR_COMPUTE_64F`
    - `LWDA_R_64F`

  * - `LWDA_C_16F` or `LWDA_C_16BF` or `LWDA_C_32F`
    - `LWTENSOR_COMPUTE_16F` or `LWTENSOR_COMPUTE_16BF` or `LWTENSOR_COMPUTE_TF32` or `LWTENSOR_COMPUTE_32F`
    - `LWDA_C_32F`

  * - `LWDA_C_32F`
    - `LWTENSOR_COMPUTE_TF32`
    - `LWDA_C_32F`

  * - `LWDA_C_64F`
    - `LWTENSOR_COMPUTE_32F` or `LWTENSOR_COMPUTE_64F`
    - `LWDA_C_64F`

As of lwTENSOR 1.2.0, :ref:`lwtensorComputeType-label` no longer distinguishes between real- and complex-valued compute types (e.g., `LWTENSOR_R_MIN_32F` and `LWTENSOR_C_MIN_32F`) have been deprecated.

.. _supported-gpus-label:

Supported GPUs
--------------

lwTENSOR supports any Lwpu GPU with a compute capability larger or equal to 6.0.

.. _graphs-label:

LWCA Graph Support
------------------

All operations in lwTENSOR can be captured using LWCA graphs.

The only mode of operation that is not supported for graph capture are contractions (`lwtensorContraction`) while the corresponding plan is actively being autotuned (see :ref:`plan-cache-overview-label`).
That restriction exists because during auto-tuning, lwTENSOR iterates through different kernels.
While graphs capture still works in that case, it is not recommended as it may capture a suboptimal kernel.

.. _elw-variables-label:

Environment Variables
---------------------

The environment variables in this section modify lwTENSOR's runtime behavior. Note that
these environment variables are read only when the handle is initialized (i.e.,
lwtensorInit()); hence, changes to the environment variables will only take effect for a
newly-initialized handle.


`LWTENSOR_LOGINFO_DBG`, when set to `1`, enables additional error diagnostics if an error is encountered.
These error diagnostics are printed to the standard output.

.. code-block:: bash

  export LWTENSOR_LOGINFO_DBG=1


`LWIDIA_TF32_OVERRIDE`, when set to `0`, will override any defaults or programmatic
configuration of LWPU libraries, and never accelerate FP32 computations with TF32 tensor
cores. This is meant to be a debugging tool only, and no code outside LWPU libraries
should change behavior based on this environment variable. Any other setting besides `0`
is reserved for future use.

.. code-block:: bash

  export LWIDIA_TF32_OVERRIDE=0


`LWTENSOR_DISABLE_PLAN_CACHE`, when set to `1`, disables the *plan cache* (see :ref:`plan-cache-overview-label`)

.. code-block:: bash

  export LWTENSOR_DISABLE_PLAN_CACHE=1
