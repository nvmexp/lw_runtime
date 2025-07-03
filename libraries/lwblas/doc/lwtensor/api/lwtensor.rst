
.. default-role:: cpp
.. highlight:: cpp

lwTENSOR Functions
==================

Helper Functions
----------------------------------

The helper functions initialize lwTENSOR, create tensor descriptors,
check error codes, and retrieve library and LWCA runtime versions.

------------

:code:`lwtensorInit()`
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorInit

------------

:code:`lwtensorInitTensorDescriptor()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorInitTensorDescriptor

------------

:code:`lwtensorGetAlignmentRequirement()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorGetAlignmentRequirement

------------

:code:`lwtensorGetErrorString()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorGetErrorString

------------

:code:`lwtensorGetVersion()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorGetVersion

------------

:code:`lwtensorGetLwdartVersion()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorGetLwdartVersion

.. _elementwise-operations-label:

Element-wise Operations
-------------------------------------

The following functions perform element-wise operations between tensors.

------------

:code:`lwtensorElementwiseTrinary()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorElementwiseTrinary

------------

:code:`lwtensorElementwiseBinary()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorElementwiseBinary

------------

:code:`lwtensorPermutation()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorPermutation

.. _contraction-operations-label:

Contraction Operations
-------------------------------

The following functions perform contractions between tensors.

------------

:code:`lwtensorInitContractionDescriptor()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorInitContractionDescriptor

------------

:code:`lwtensorContractionDescriptorSetAttribute()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorContractionDescriptorSetAttribute

------------

:code:`lwtensorInitContractionFind()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorInitContractionFind

------------

:code:`lwtensorContractionFindSetAttribute()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorContractionFindSetAttribute

------------

:code:`lwtensorContractionGetWorkspace()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorContractionGetWorkspace

------------

:code:`lwtensorInitContractionPlan()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorInitContractionPlan

------------

:code:`lwtensorContraction()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorContraction

------------

:code:`lwtensorContractionMaxAlgos()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorContractionMaxAlgos

.. _reduction-operations-label:

Reduction Operations
-----------------------------

The following functions perform tensor reductions.

------------

:code:`lwtensorReduction()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorReduction

------------

:code:`lwtensorReductionGetWorkspace()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorReductionGetWorkspace

------------


Cache-related Operations (beta)
-----------------------------

:code:`lwtensorHandleDetachPlanCachelines()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorHandleDetachPlanCachelines

------------

:code:`lwtensorHandleAttachPlanCachelines()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorHandleAttachPlanCachelines

------------


:code:`lwtensorHandleReadCacheFromFile()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorHandleReadCacheFromFile

------------


:code:`lwtensorHandleWriteCacheToFile()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lwtensorHandleWriteCacheToFile

------------
