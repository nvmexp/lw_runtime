
.. _plan-cache-label:

Plan Cache (beta)
=================

This section introduces the *software-managed plan cache* that has the following key features:

* Minimize launch-related overhead (e.g., due to kernel selection).
* Overhead-free autotuning (a.k.a. :ref:`plan-cache-inc-label`).

   * This feature enables users to automatically find the best implementation for the
     given problem and thereby increases performance.

* The cache is implemented in a thread-safe manner and is shared across all threads that use the same `lwtensorHandle_t`.
* Serialize and deserialize the cache:

   * Allows users to store the state of the cache to disc and reuse it later

In essence, the *plan cache* can be seen as a lookup table from a specific problem
instance (e.g., `lwtensorContractionDescriptor_t`) to an actual implementation (encoded by
`lwtensorContractionPlan_t`).

The remainder of this section assumes familiarity with :ref:`getting-started-label`.

.. note::
  The cache can always be deactivated via the `LWTENSOR_DISABLE_PLAN_CACHE`
  environment variable (see :ref:`elw-variables-label`).

.. _plan-cache-inc-label:

Incremental Autotuning
----------------------

The incremental autotuning feature enables users to automatically explore different
implementations, referred to as *candidates*, for a given problem.

When using the cache with the incremental auto-tuning feature (`LWTENSOR_AUTOTUNE_INCREMENTAL`),
successive ilwocations of the same contraction problem (albeit with potentially
different data pointers) will be performed by different candidates; the timing for those
candidates is automatically measured and the fastest candidate is stored in the *plan
cache*. The number of different candidates to explore is configurable by the user (via `LWTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT`);
all subsequent calls to the same problem will then be mapped to the fastest candidate
(stored in the cache), thus taking advantage of the fastest (real-world) candidate.

This autotuning approach has some key advantages:

* Candidates are evaluated at a point in time where hardware caches are in a production-environment state (i.e.,
  the hardware cache state reflects the real-world situation).
* Overhead is minimized (i.e., no timing loop, no synchronization).

   * Moreover, the candidates are evaluated in the order given by our performance model (from fastest to slowest).


Incremental autotuning is especially powerful if combined with
lwTENSOR's cache serialization feature (via `lwtensorHandleWriteCacheToFile` and
`lwtensorHandleReadCacheFromFile`) by writing the *tuned* cache to disc.

.. note::
  We recommend warming up the GPU (i.e., reaching steady-state
  performance) before auto-tuning is used to minimize fluctuations in measured performance.

Introductory Example
--------------------

This subsection provides a basic overview of the cache-related API calls and features. In
addition to the steps outlined in :ref:`getting-started-label`, in this example we also:

  * Attach cachelines to the handle (via `lwtensorHandleAttachPlanCachelines`).
  * Configure the cache behavior on a contraction-by-contraction basis (via `lwtensorContractionFindSetAttribute`).

Let's start from the same example outlined in :ref:`getting-started-label` and enhance it with the steps required to use the plan cache.
First we have to add calls to attach and detach cachelines:

.. code-block:: cpp

  // Set number of cache lines
  constexpr int32_t numCachelines = 1024;
  // Set cache size and allocate
  const size_t sizeCache = numCachelines * sizeof(lwtensorPlanCacheline_t);
  lwtensorPlanCacheline_t* cachelines = (lwtensorPlanCacheline_t*) malloc(sizeCache);
  // Attach cache
  HANDLE_ERROR( lwtensorHandleAttachPlanCachelines(&handle, cachelines, numCachelines) );

  // ...

  // Detach cache and free-up resources
  HANDLE_ERROR( lwtensorHandleDetachPlanCachelines(&handle) );

Note that the number of cachelines is configurable by the user; ideally we would like to supply as many cachelines 
as the applications has distinct contraction calls. Since this might not always be possible (due to memory constraints),
lwTENSOR's plan cache will evict cache entries using a least-recently-used (LRU) policy. Users
can choose to disable caching on a contraction-by-contraction basis (via `lwtensorCacheMode_t::LWTENSOR_CACHE_NONE`).

.. note::
  Until the cachelines are detached using `lwtensorHandleDetachPlanCachelines`, the user-allocated cachelines must not be freed.

Moreover, `lwtensorHandleDetachPlanCachelines` also deallocates other plan cache related resources.
It should be called once the plan cache is no longer needed in order to avoid resource leaks.

With the above mentioned changes (and a loop around the contraction call) our example now looks as follows:

.. code-block:: cpp
  :linenos:

  #include <stdlib.h>
  #include <stdio.h>

  #include <lwda_runtime.h>
  #include <lwtensor.h>

  #include <unordered_map>
  #include <vector>

  // Handle lwTENSOR errors
  #define HANDLE_ERROR(x) {                                                              \
    const auto err = x;                                                                  \
    if( err != LWTENSOR_STATUS_SUCCESS )                                                   \
    { printf("Error: %s in line %d\n", lwtensorGetErrorString(err), __LINE__); exit(-1); } \
  }

  int main(int argc, char** argv)
  {
    // Host element type definition
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    // LWCA types
    lwdaDataType_t typeA = LWDA_R_32F;
    lwdaDataType_t typeB = LWDA_R_32F;
    lwdaDataType_t typeC = LWDA_R_32F;
    lwtensorComputeType_t typeCompute = LWTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.9f;

    printf("Include headers and define data types\n");

    /* ***************************** */

    // Create vector of modes
    std::vector<int> modeC{'m','u','n','v'};
    std::vector<int> modeA{'m','h','k','n'};
    std::vector<int> modeB{'u','k','v','h'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    // Extents
    std::unordered_map<int, int64_t> extent;
    extent['m'] = 96;
    extent['n'] = 96;
    extent['u'] = 96;
    extent['v'] = 64;
    extent['h'] = 64;
    extent['k'] = 64;

    // Create a vector of extents for each tensor
    std::vector<int64_t> extentC;
    for(auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for(auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for(auto mode : modeB)
        extentB.push_back(extent[mode]);

    printf("Define modes and extents\n");

    /* ***************************** */

    // Number of elements of each tensor
    size_t elementsA = 1;
    for(auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for(auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for(auto mode : modeC)
        elementsC *= extent[mode];

    // Size in bytes
    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    // Allocate on device
    void *A_d, *B_d, *C_d;
    lwdaMalloc((void**)&A_d, sizeA);
    lwdaMalloc((void**)&B_d, sizeB);
    lwdaMalloc((void**)&C_d, sizeC);

    // Allocate on host
    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

    // Initialize data on host
    for(int64_t i = 0; i < elementsA; i++)
        A[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for(int64_t i = 0; i < elementsB; i++)
        B[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for(int64_t i = 0; i < elementsC; i++)
        C[i] = (((float) rand())/RAND_MAX - 0.5)*100;

    // Copy to device
    lwdaMemcpy(C_d, C, sizeC, lwdaMemcpyHostToDevice);
    lwdaMemcpy(A_d, A, sizeA, lwdaMemcpyHostToDevice);
    lwdaMemcpy(B_d, B, sizeB, lwdaMemcpyHostToDevice);

    printf("Allocate, initialize and transfer tensors\n");

    /* ***************************** */

    // Initialize lwTENSOR library
    lwtensorHandle_t handle;
    lwtensorInit(&handle);

    /**********************
     * Setup plan cache
     **********************/
    printf("Attach cachelines\n");

    constexpr int32_t numCachelines = 1024;
    const size_t sizeCache = numCachelines * sizeof(lwtensorPlanCacheline_t);
    printf("Allocating: %.2f kB for the cache\n", sizeCache / 1000.);
    lwtensorPlanCacheline_t* cachelines = (lwtensorPlanCacheline_t*) malloc(sizeCache);
    HANDLE_ERROR( lwtensorHandleAttachPlanCachelines(&handle, cachelines, numCachelines) );

    // Create Tensor Descriptors
    lwtensorTensorDescriptor_t descA;
    HANDLE_ERROR( lwtensorInitTensorDescriptor( &handle,
                &descA,
                nmodeA,
                extentA.data(),
                NULL,/*stride*/
                typeA, LWTENSOR_OP_IDENTITY ) );

    lwtensorTensorDescriptor_t descB;
    HANDLE_ERROR( lwtensorInitTensorDescriptor( &handle,
                &descB,
                nmodeB,
                extentB.data(),
                NULL,/*stride*/
                typeB, LWTENSOR_OP_IDENTITY ) );

    lwtensorTensorDescriptor_t descC;
    HANDLE_ERROR( lwtensorInitTensorDescriptor( &handle,
                &descC,
                nmodeC,
                extentC.data(),
                NULL,/*stride*/
                typeC, LWTENSOR_OP_IDENTITY ) );

    printf("Initialize lwTENSOR and tensor descriptors\n");

    /* ***************************** */

    //Retrieve the memory alignment for each tensor
    uint32_t alignmentRequirementA;
    HANDLE_ERROR( lwtensorGetAlignmentRequirement( &handle,
               A_d,
               &descA,
               &alignmentRequirementA) );

    uint32_t alignmentRequirementB;
    HANDLE_ERROR( lwtensorGetAlignmentRequirement( &handle,
               B_d,
               &descB,
               &alignmentRequirementB) );

    uint32_t alignmentRequirementC;
    HANDLE_ERROR( lwtensorGetAlignmentRequirement( &handle,
               C_d,
               &descC, 
               &alignmentRequirementC) );

    printf("Query best alignment requirement for our pointers\n");

    /* ***************************** */

    // Create the Contraction Descriptor
    lwtensorContractionDescriptor_t desc;
    HANDLE_ERROR( lwtensorInitContractionDescriptor( &handle, 
                &desc,
                &descA, modeA.data(), alignmentRequirementA,
                &descB, modeB.data(), alignmentRequirementB,
                &descC, modeC.data(), alignmentRequirementC,
                &descC, modeC.data(), alignmentRequirementC,
                typeCompute) );

    printf("Initialize contraction descriptor\n");

    /* ***************************** */

    // Set the algorithm to use
    lwtensorContractionFind_t find;
    HANDLE_ERROR( lwtensorInitContractionFind( 
                &handle, &find, 
                LWTENSOR_ALGO_DEFAULT) );

    printf("Initialize settings to find algorithm\n");

    /* ***************************** */

    // Query workspace
    size_t worksize = 0;
    HANDLE_ERROR( lwtensorContractionGetWorkspace(&handle,
                &desc,
                &find,
                LWTENSOR_WORKSPACE_RECOMMENDED, &worksize ) );

    // Allocate workspace
    void *work = nullptr;
    if(worksize > 0)
    {
        if( lwdaSuccess != lwdaMalloc(&work, worksize) ) // This is optional!
        {
            work = nullptr;
            worksize = 0;
        }
    }

    printf("Query recommended workspace size and allocate it\n");

    /* ***************************** */
    printf("Execute contraction from plan\n");

    int numRuns = 5;
    for(int i=0; i < numRuns; ++i)
    {
       // Create Contraction Plan && look-up cache (if attached)
       lwtensorContractionPlan_t plan;
       HANDLE_ERROR( lwtensorInitContractionPlan(&handle,
                                                 &plan,
                                                 &desc,
                                                 &find,
                                                 worksize) );

       printf("Create plan for contraction\n");

       /* ***************************** */

       lwtensorStatus_t err;

       // Execute the tensor contraction
       err = lwtensorContraction(&handle,
                                 &plan,
                          (void*)&alpha, A_d,
                                         B_d,
                          (void*)&beta,  C_d,
                                         C_d, 
                                 work, worksize, 0 /* stream */);
       lwdaDeviceSynchronize();

       // Check for errors
       if(err != LWTENSOR_STATUS_SUCCESS)
       {
           printf("ERROR: %s\n", lwtensorGetErrorString(err));
       }
    }

    /* ***************************** */

    // Detach cache and free-up resources
    HANDLE_ERROR( lwtensorHandleDetachPlanCachelines(&handle) );

    if ( A ) free( A );
    if ( B ) free( B );
    if ( C ) free( C );
    if ( cachelines ) free(cachelines);
    if ( A_d ) lwdaFree( A_d );
    if ( B_d ) lwdaFree( B_d );
    if ( C_d ) lwdaFree( C_d );
    if ( work ) lwdaFree( work );

    printf("Successful completion\n");

    return 0;
  }

This minimal change suffices to cache the plan for the tensor contraction call in line 233 and 246.
It's important to note that the call to `lwtensorInitContractionPlan` is inside of the
loop; the lookup from the cache happens here.

To disable caching for a certain contraction, the
corresponding `lwtensorContractionFind_t` needs to be modified accordingly:

.. code-block:: cpp

  const lwtensorCacheMode_t cacheMode = LWTENSOR_CACHE_MODE_NONE;
  HANDLE_ERROR(lwtensorContractionFindSetAttribute(
       &handle,
       &find,
       LWTENSOR_CONTRACTION_FIND_CACHE_MODE,
       &cacheMode,
       sizeof(lwtensorCacheMode_t)));

This concludes the introductory example.

Advanced Example
----------------

This example will augment the previous example and explains how to:

* Take advantage of incremental auto-tuning

   * It is recommended to warm up the GPU (i.e., reach steady-state performance) before using auto-tuning (to avoid big fluctuations in measured performance)

* Use tags to distinguish two otherwise identical tensor contractions from each other

   * This is usful if the hardware cache of the GPU is (likely) substantially different
     between the two calls (e.g., if one of the operands was just read/written by a
     preceeding call) *and* it is expected that the state of the cache has
     significant impact on the performance (e.g., for bandwidth-bound contractions)

* Write the plan cache state to a file and read it back in

Let us start by enabling incremental autotuning.
To do so, we modify `lwtensorContractionFind_t` as follows:

.. code-block:: cpp

   const lwtensorAutotuneMode_t autotuneMode = LWTENSOR_AUTOTUNE_INCREMENTAL;
   HANDLE_ERROR(lwtensorContractionFindSetAttribute(
       &handle,
       &find,
       LWTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE,
       &autotuneMode ,
       sizeof(lwtensorAutotuneMode_t)));

   const uint32_t incCount = 4;
   HANDLE_ERROR(lwtensorContractionFindSetAttribute(
       &handle,
       &find,
       LWTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT,
       &incCount,
       sizeof(uint32_t)));

The first call to `lwtensorContractionFindSetAttribute` enables incremental auto-tuning,
while the second call sets the `LWTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT`; this 
value corresponds to the number of different candidates that should be explored via
*incremental autotuning* before subsequent calls look-up from the plan cache.
Higher values of `incCount` explore more candidates, and as such cause a larger overhead
initially, but they can also result in better performance -- if the initial overhead
can be amortized (e.g., when writing the cache to disc).
We feel that a `LWTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT` of four is a good default value.

The following code incorporates those changes:

.. code-block:: cpp
  :linenos:

  #include <stdlib.h>
  #include <stdio.h>

  #include <lwda_runtime.h>
  #include <lwtensor.h>

  #include <unordered_map>
  #include <vector>

  // Handle lwTENSOR errors
  #define HANDLE_ERROR(x) {                                                              \
    const auto err = x;                                                                  \
    if( err != LWTENSOR_STATUS_SUCCESS )                                                   \
    { printf("Error: %s in line %d\n", lwtensorGetErrorString(err), __LINE__); exit(-1); } \
  }

  int main(int argc, char** argv)
  {
    // Host element type definition
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    // LWCA types
    lwdaDataType_t typeA = LWDA_R_32F;
    lwdaDataType_t typeB = LWDA_R_32F;
    lwdaDataType_t typeC = LWDA_R_32F;
    lwtensorComputeType_t typeCompute = LWTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.9f;

    printf("Include headers and define data types\n");

    /* ***************************** */

    // Create vector of modes
    std::vector<int> modeC{'m','u','n','v'};
    std::vector<int> modeA{'m','h','k','n'};
    std::vector<int> modeB{'u','k','v','h'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    // Extents
    std::unordered_map<int, int64_t> extent;
    extent['m'] = 96;
    extent['n'] = 96;
    extent['u'] = 96;
    extent['v'] = 64;
    extent['h'] = 64;
    extent['k'] = 64;

    // Create a vector of extents for each tensor
    std::vector<int64_t> extentC;
    for(auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for(auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for(auto mode : modeB)
        extentB.push_back(extent[mode]);

    printf("Define modes and extents\n");

    /* ***************************** */

    // Number of elements of each tensor
    size_t elementsA = 1;
    for(auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for(auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for(auto mode : modeC)
        elementsC *= extent[mode];

    // Size in bytes
    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    // Allocate on device
    void *A_d, *B_d, *C_d;
    lwdaMalloc((void**)&A_d, sizeA);
    lwdaMalloc((void**)&B_d, sizeB);
    lwdaMalloc((void**)&C_d, sizeC);

    // Allocate on host
    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

    // Initialize data on host
    for(int64_t i = 0; i < elementsA; i++)
        A[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for(int64_t i = 0; i < elementsB; i++)
        B[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for(int64_t i = 0; i < elementsC; i++)
        C[i] = (((float) rand())/RAND_MAX - 0.5)*100;

    // Copy to device
    lwdaMemcpy(C_d, C, sizeC, lwdaMemcpyHostToDevice);
    lwdaMemcpy(A_d, A, sizeA, lwdaMemcpyHostToDevice);
    lwdaMemcpy(B_d, B, sizeB, lwdaMemcpyHostToDevice);

    printf("Allocate, initialize and transfer tensors\n");

    /* ***************************** */

    // Initialize lwTENSOR library
    lwtensorHandle_t handle;
    lwtensorInit(&handle);

    /**********************
     * Setup plan cache
     **********************/
    printf("Attach cachelines\n");

    constexpr int32_t numCachelines = 1024;
    const size_t sizeCache = numCachelines * sizeof(lwtensorPlanCacheline_t);
    printf("Allocating: %.2f kB for the cache\n", sizeCache / 1000.);
    lwtensorPlanCacheline_t* cachelines = (lwtensorPlanCacheline_t*) malloc(sizeCache);
    HANDLE_ERROR( lwtensorHandleAttachPlanCachelines(&handle, cachelines, numCachelines) );

    // Create Tensor Descriptors
    lwtensorTensorDescriptor_t descA;
    HANDLE_ERROR( lwtensorInitTensorDescriptor( &handle,
                &descA,
                nmodeA,
                extentA.data(),
                NULL,/*stride*/
                typeA, LWTENSOR_OP_IDENTITY ) );

    lwtensorTensorDescriptor_t descB;
    HANDLE_ERROR( lwtensorInitTensorDescriptor( &handle,
                &descB,
                nmodeB,
                extentB.data(),
                NULL,/*stride*/
                typeB, LWTENSOR_OP_IDENTITY ) );

    lwtensorTensorDescriptor_t descC;
    HANDLE_ERROR( lwtensorInitTensorDescriptor( &handle,
                &descC,
                nmodeC,
                extentC.data(),
                NULL,/*stride*/
                typeC, LWTENSOR_OP_IDENTITY ) );

    printf("Initialize lwTENSOR and tensor descriptors\n");

    /* ***************************** */

    //Retrieve the memory alignment for each tensor
    uint32_t alignmentRequirementA;
    HANDLE_ERROR( lwtensorGetAlignmentRequirement( &handle,
               A_d,
               &descA,
               &alignmentRequirementA) );

    uint32_t alignmentRequirementB;
    HANDLE_ERROR( lwtensorGetAlignmentRequirement( &handle,
               B_d,
               &descB,
               &alignmentRequirementB) );

    uint32_t alignmentRequirementC;
    HANDLE_ERROR( lwtensorGetAlignmentRequirement( &handle,
               C_d,
               &descC, 
               &alignmentRequirementC) );

    printf("Query best alignment requirement for our pointers\n");

    /* ***************************** */

    // Create the Contraction Descriptor
    lwtensorContractionDescriptor_t desc;
    HANDLE_ERROR( lwtensorInitContractionDescriptor( &handle, 
                &desc,
                &descA, modeA.data(), alignmentRequirementA,
                &descB, modeB.data(), alignmentRequirementB,
                &descC, modeC.data(), alignmentRequirementC,
                &descC, modeC.data(), alignmentRequirementC,
                typeCompute) );

    printf("Initialize contraction descriptor\n");

    /* ***************************** */

    // Set the algorithm to use
    lwtensorContractionFind_t find;
    HANDLE_ERROR( lwtensorInitContractionFind( 
                &handle, &find, 
                LWTENSOR_ALGO_DEFAULT) );

   const lwtensorAutotuneMode_t autotuneMode = LWTENSOR_AUTOTUNE_INCREMENTAL;
   HANDLE_ERROR(lwtensorContractionFindSetAttribute(
       &handle,
       &find,
       LWTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE,
       &autotuneMode ,
       sizeof(lwtensorAutotuneMode_t)));

   const uint32_t incCount = 4;
   HANDLE_ERROR(lwtensorContractionFindSetAttribute(
       &handle,
       &find,
       LWTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT,
       &incCount,
       sizeof(uint32_t)));

    printf("Initialize settings to find algorithm\n");

    /* ***************************** */

    // Query workspace
    size_t worksize = 0;
    HANDLE_ERROR( lwtensorContractionGetWorkspace(&handle,
                &desc,
                &find,
                LWTENSOR_WORKSPACE_RECOMMENDED, &worksize ) );

    // Allocate workspace
    void *work = nullptr;
    if(worksize > 0)
    {
        if( lwdaSuccess != lwdaMalloc(&work, worksize) ) // This is optional!
        {
            work = nullptr;
            worksize = 0;
        }
    }

    printf("Query recommended workspace size and allocate it\n");

    /* ***************************** */
    printf("Execute contraction from plan\n");

    int numRuns = 5;
    for(int i=0; i < numRuns; ++i)
    {
       // Create Contraction Plan && look-up cache (if attached)
       lwtensorContractionPlan_t plan;
       HANDLE_ERROR( lwtensorInitContractionPlan(&handle,
                                                 &plan,
                                                 &desc,
                                                 &find,
                                                 worksize) );

       printf("Create plan for contraction\n");

       /* ***************************** */

       lwtensorStatus_t err;

       // Execute the tensor contraction
       err = lwtensorContraction(&handle,
                                 &plan,
                          (void*)&alpha, A_d,
                                         B_d,
                          (void*)&beta,  C_d,
                                         C_d, 
                                 work, worksize, 0 /* stream */);
       lwdaDeviceSynchronize();

       // Check for errors
       if(err != LWTENSOR_STATUS_SUCCESS)
       {
           printf("ERROR: %s\n", lwtensorGetErrorString(err));
       }
    }

    /* ***************************** */

    // Detach cache and free-up resources
    HANDLE_ERROR( lwtensorHandleDetachPlanCachelines(&handle) );

    if ( A ) free( A );
    if ( B ) free( B );
    if ( C ) free( C );
    if ( cachelines ) free(cachelines);
    if ( A_d ) lwdaFree( A_d );
    if ( B_d ) lwdaFree( B_d );
    if ( C_d ) lwdaFree( C_d );
    if ( work ) lwdaFree( work );

    printf("Successful completion\n");

    return 0;
  }

Let us further augment the above example by writing the plan cache to a file and reading it in
(provided it was previously written):

.. code-block:: cpp

   const char cacheFilename[] = "./cache.bin";
   uint32_t numCachelinesRead = 0;
   lwtensorStatus_t status = lwtensorHandleReadCacheFromFile(&handle, cacheFilename, &numCachelinesRead);
   if (status == LWTENSOR_STATUS_SUCCESS)
   {
       printf("%d cachelines have been successfully read from file (%s).\n", numCachelinesRead, cacheFilename);
   }
   else if (status == LWTENSOR_STATUS_IO_ERROR)
   {
       printf("File (%s) doesn't seem to exist.\n", cacheFilename);
   }
   else if (status == LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
   {
       printf("Cannot read cache: Please attach at least %d cachelines to the handle.\n", numCachelines);
   }

   // ...

   const char filename[] = "./cache.bin";
   HANDLE_ERROR( lwtensorHandleWriteCacheToFile(&handle, filename) );
   printf("Cache has been successfully written to file (%s).\n", filename);

.. warning:: 
  `lwtensorHandleReadCacheFromFile` only succeeds if the number of attached
  cachelines is sufficient to read **all** cachelines stored in the file; otherwise
  `LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE` is returned and the sufficient number of cachelines
  is stored in `numCachelinesRead`.

With these changes the example now looks as follows:

.. code-block:: cpp
  :linenos:

  #include <stdlib.h>
  #include <stdio.h>

  #include <lwda_runtime.h>
  #include <lwtensor.h>

  #include <unordered_map>
  #include <vector>

  // Handle lwTENSOR errors
  #define HANDLE_ERROR(x) {                                                              \
    const auto err = x;                                                                  \
    if( err != LWTENSOR_STATUS_SUCCESS )                                                   \
    { printf("Error: %s in line %d\n", lwtensorGetErrorString(err), __LINE__); exit(-1); } \
  }

  int main(int argc, char** argv)
  {
    // Host element type definition
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    // LWCA types
    lwdaDataType_t typeA = LWDA_R_32F;
    lwdaDataType_t typeB = LWDA_R_32F;
    lwdaDataType_t typeC = LWDA_R_32F;
    lwtensorComputeType_t typeCompute = LWTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.9f;

    printf("Include headers and define data types\n");

    /* ***************************** */

    // Create vector of modes
    std::vector<int> modeC{'m','u','n','v'};
    std::vector<int> modeA{'m','h','k','n'};
    std::vector<int> modeB{'u','k','v','h'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    // Extents
    std::unordered_map<int, int64_t> extent;
    extent['m'] = 96;
    extent['n'] = 96;
    extent['u'] = 96;
    extent['v'] = 64;
    extent['h'] = 64;
    extent['k'] = 64;

    // Create a vector of extents for each tensor
    std::vector<int64_t> extentC;
    for(auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for(auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for(auto mode : modeB)
        extentB.push_back(extent[mode]);

    printf("Define modes and extents\n");

    /* ***************************** */

    // Number of elements of each tensor
    size_t elementsA = 1;
    for(auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for(auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for(auto mode : modeC)
        elementsC *= extent[mode];

    // Size in bytes
    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    // Allocate on device
    void *A_d, *B_d, *C_d;
    lwdaMalloc((void**)&A_d, sizeA);
    lwdaMalloc((void**)&B_d, sizeB);
    lwdaMalloc((void**)&C_d, sizeC);

    // Allocate on host
    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

    // Initialize data on host
    for(int64_t i = 0; i < elementsA; i++)
        A[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for(int64_t i = 0; i < elementsB; i++)
        B[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for(int64_t i = 0; i < elementsC; i++)
        C[i] = (((float) rand())/RAND_MAX - 0.5)*100;

    // Copy to device
    lwdaMemcpy(C_d, C, sizeC, lwdaMemcpyHostToDevice);
    lwdaMemcpy(A_d, A, sizeA, lwdaMemcpyHostToDevice);
    lwdaMemcpy(B_d, B, sizeB, lwdaMemcpyHostToDevice);

    printf("Allocate, initialize and transfer tensors\n");

    /* ***************************** */

    // Initialize lwTENSOR library
    lwtensorHandle_t handle;
    lwtensorInit(&handle);

    /**********************
     * Setup plan cache
     **********************/
    printf("Attach cachelines\n");

    constexpr int32_t numCachelines = 1024;
    const size_t sizeCache = numCachelines * sizeof(lwtensorPlanCacheline_t);
    printf("Allocating: %.2f kB for the cache\n", sizeCache / 1000.);
    lwtensorPlanCacheline_t* cachelines = (lwtensorPlanCacheline_t*) malloc(sizeCache);
    HANDLE_ERROR( lwtensorHandleAttachPlanCachelines(&handle, cachelines, numCachelines) );


    const char cacheFilename[] = "./cache.bin";
    uint32_t numCachelinesRead = 0;
    lwtensorStatus_t status = lwtensorHandleReadCacheFromFile(&handle, cacheFilename, &numCachelinesRead);
    if (status == LWTENSOR_STATUS_SUCCESS)
    {
        printf("%d cachelines have been successfully read from file (%s).\n", numCachelinesRead, cacheFilename);
    }
    else if (status == LWTENSOR_STATUS_IO_ERROR)
    {
        printf("File (%s) doesn't seem to exist.\n", cacheFilename);
    }
    else if (status == LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
    {
        printf("Cannot read cache: Please attach at least %d cachelines to the handle.\n", numCachelines);
    }

    // Create Tensor Descriptors
    lwtensorTensorDescriptor_t descA;
    HANDLE_ERROR( lwtensorInitTensorDescriptor( &handle,
                &descA,
                nmodeA,
                extentA.data(),
                NULL,/*stride*/
                typeA, LWTENSOR_OP_IDENTITY ) );

    lwtensorTensorDescriptor_t descB;
    HANDLE_ERROR( lwtensorInitTensorDescriptor( &handle,
                &descB,
                nmodeB,
                extentB.data(),
                NULL,/*stride*/
                typeB, LWTENSOR_OP_IDENTITY ) );

    lwtensorTensorDescriptor_t descC;
    HANDLE_ERROR( lwtensorInitTensorDescriptor( &handle,
                &descC,
                nmodeC,
                extentC.data(),
                NULL,/*stride*/
                typeC, LWTENSOR_OP_IDENTITY ) );

    printf("Initialize lwTENSOR and tensor descriptors\n");

    /* ***************************** */

    //Retrieve the memory alignment for each tensor
    uint32_t alignmentRequirementA;
    HANDLE_ERROR( lwtensorGetAlignmentRequirement( &handle,
               A_d,
               &descA,
               &alignmentRequirementA) );

    uint32_t alignmentRequirementB;
    HANDLE_ERROR( lwtensorGetAlignmentRequirement( &handle,
               B_d,
               &descB,
               &alignmentRequirementB) );

    uint32_t alignmentRequirementC;
    HANDLE_ERROR( lwtensorGetAlignmentRequirement( &handle,
               C_d,
               &descC, 
               &alignmentRequirementC) );

    printf("Query best alignment requirement for our pointers\n");

    /* ***************************** */

    // Create the Contraction Descriptor
    lwtensorContractionDescriptor_t desc;
    HANDLE_ERROR( lwtensorInitContractionDescriptor( &handle, 
                &desc,
                &descA, modeA.data(), alignmentRequirementA,
                &descB, modeB.data(), alignmentRequirementB,
                &descC, modeC.data(), alignmentRequirementC,
                &descC, modeC.data(), alignmentRequirementC,
                typeCompute) );

    printf("Initialize contraction descriptor\n");

    /* ***************************** */

    // Set the algorithm to use
    lwtensorContractionFind_t find;
    HANDLE_ERROR( lwtensorInitContractionFind( 
                &handle, &find, 
                LWTENSOR_ALGO_DEFAULT) );

    const lwtensorAutotuneMode_t autotuneMode = LWTENSOR_AUTOTUNE_INCREMENTAL;
    HANDLE_ERROR(lwtensorContractionFindSetAttribute(
        &handle,
        &find,
        LWTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE,
        &autotuneMode ,
        sizeof(lwtensorAutotuneMode_t)));

    const uint32_t incCount = 4;
    HANDLE_ERROR(lwtensorContractionFindSetAttribute(
        &handle,
        &find,
        LWTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT,
        &incCount,
        sizeof(uint32_t)));

    printf("Initialize settings to find algorithm\n");

    /* ***************************** */

    // Query workspace
    size_t worksize = 0;
    HANDLE_ERROR( lwtensorContractionGetWorkspace(&handle,
                &desc,
                &find,
                LWTENSOR_WORKSPACE_RECOMMENDED, &worksize ) );

    // Allocate workspace
    void *work = nullptr;
    if(worksize > 0)
    {
        if( lwdaSuccess != lwdaMalloc(&work, worksize) ) // This is optional!
        {
            work = nullptr;
            worksize = 0;
        }
    }

    printf("Query recommended workspace size and allocate it\n");

    /* ***************************** */
    printf("Execute contraction from plan\n");

    int numRuns = 5;
    for(int i=0; i < numRuns; ++i)
    {
       // Create Contraction Plan && look-up cache (if attached)
       lwtensorContractionPlan_t plan;
       HANDLE_ERROR( lwtensorInitContractionPlan(&handle,
                                                 &plan,
                                                 &desc,
                                                 &find,
                                                 worksize) );

       printf("Create plan for contraction\n");

       /* ***************************** */

       lwtensorStatus_t err;

       // Execute the tensor contraction
       err = lwtensorContraction(&handle,
                                 &plan,
                          (void*)&alpha, A_d,
                                         B_d,
                          (void*)&beta,  C_d,
                                         C_d, 
                                 work, worksize, 0 /* stream */);
       lwdaDeviceSynchronize();

       // Check for errors
       if(err != LWTENSOR_STATUS_SUCCESS)
       {
           printf("ERROR: %s\n", lwtensorGetErrorString(err));
       }
    }


    /* ***************************** */
    HANDLE_ERROR( lwtensorHandleWriteCacheToFile(&handle, cacheFilename) );
    printf("Cache has been successfully written to file (%s).\n", cacheFilename);

    // Detach cache and free-up resources
    HANDLE_ERROR( lwtensorHandleDetachPlanCachelines(&handle) );

    if ( A ) free( A );
    if ( B ) free( B );
    if ( C ) free( C );
    if ( cachelines ) free(cachelines);
    if ( A_d ) lwdaFree( A_d );
    if ( B_d ) lwdaFree( B_d );
    if ( C_d ) lwdaFree( C_d );
    if ( work ) lwdaFree( work );

    printf("Successful completion\n");

    return 0;
  }


Finally, let us add a second contraction loop, but this time we want the --otherwise
identical-- contraction to be cached using a different cacheline:
This can be useful if the state of the hardware cache between these two calls is substantially different (i.e.,
affecting the measured runtime of the kernels).
To that end, we use the `LWTENSOR_CONTRACTION_DESCRIPTOR_TAG` attribute:

.. code-block:: cpp

   uint32_t tag = 1;
   HANDLE_ERROR( lwtensorContractionDescriptorSetAttribute(
        &handle,
        &desc,
        LWTENSOR_CONTRACTION_DESCRIPTOR_TAG,
        &tag,
        sizeof(uint32_t)));

With this change, the example code now looks as follows:

.. code-block:: cpp
  :linenos:

  #include <stdlib.h>
  #include <stdio.h>

  #include <lwda_runtime.h>
  #include <lwtensor.h>

  #include <unordered_map>
  #include <vector>

  // Handle lwTENSOR errors
  #define HANDLE_ERROR(x) {                                                              \
    const auto err = x;                                                                  \
    if( err != LWTENSOR_STATUS_SUCCESS )                                                   \
    { printf("Error: %s in line %d\n", lwtensorGetErrorString(err), __LINE__); exit(-1); } \
  }

  int main(int argc, char** argv)
  {
    // Host element type definition
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    // LWCA types
    lwdaDataType_t typeA = LWDA_R_32F;
    lwdaDataType_t typeB = LWDA_R_32F;
    lwdaDataType_t typeC = LWDA_R_32F;
    lwtensorComputeType_t typeCompute = LWTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.9f;

    printf("Include headers and define data types\n");

    /* ***************************** */

    // Create vector of modes
    std::vector<int> modeC{'m','u','n','v'};
    std::vector<int> modeA{'m','h','k','n'};
    std::vector<int> modeB{'u','k','v','h'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    // Extents
    std::unordered_map<int, int64_t> extent;
    extent['m'] = 96;
    extent['n'] = 96;
    extent['u'] = 96;
    extent['v'] = 64;
    extent['h'] = 64;
    extent['k'] = 64;

    // Create a vector of extents for each tensor
    std::vector<int64_t> extentC;
    for(auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for(auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for(auto mode : modeB)
        extentB.push_back(extent[mode]);

    printf("Define modes and extents\n");

    /* ***************************** */

    // Number of elements of each tensor
    size_t elementsA = 1;
    for(auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for(auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for(auto mode : modeC)
        elementsC *= extent[mode];

    // Size in bytes
    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    // Allocate on device
    void *A_d, *B_d, *C_d;
    lwdaMalloc((void**)&A_d, sizeA);
    lwdaMalloc((void**)&B_d, sizeB);
    lwdaMalloc((void**)&C_d, sizeC);

    // Allocate on host
    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

    // Initialize data on host
    for(int64_t i = 0; i < elementsA; i++)
        A[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for(int64_t i = 0; i < elementsB; i++)
        B[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for(int64_t i = 0; i < elementsC; i++)
        C[i] = (((float) rand())/RAND_MAX - 0.5)*100;

    // Copy to device
    lwdaMemcpy(C_d, C, sizeC, lwdaMemcpyHostToDevice);
    lwdaMemcpy(A_d, A, sizeA, lwdaMemcpyHostToDevice);
    lwdaMemcpy(B_d, B, sizeB, lwdaMemcpyHostToDevice);

    printf("Allocate, initialize and transfer tensors\n");

    /* ***************************** */

    // Initialize lwTENSOR library
    lwtensorHandle_t handle;
    lwtensorInit(&handle);

    /**********************
     * Setup plan cache
     **********************/
    printf("Attach cachelines\n");

    constexpr int32_t numCachelines = 1024;
    const size_t sizeCache = numCachelines * sizeof(lwtensorPlanCacheline_t);
    printf("Allocating: %.2f kB for the cache\n", sizeCache / 1000.);
    lwtensorPlanCacheline_t* cachelines = (lwtensorPlanCacheline_t*) malloc(sizeCache);
    HANDLE_ERROR( lwtensorHandleAttachPlanCachelines(&handle, cachelines, numCachelines) );


    const char cacheFilename[] = "./cache.bin";
    uint32_t numCachelinesRead = 0;
    lwtensorStatus_t status = lwtensorHandleReadCacheFromFile(&handle, cacheFilename, &numCachelinesRead);
    if (status == LWTENSOR_STATUS_SUCCESS)
    {
        printf("%d cachelines have been successfully read from file (%s).\n", numCachelinesRead, cacheFilename);
    }
    else if (status == LWTENSOR_STATUS_IO_ERROR)
    {
        printf("File (%s) doesn't seem to exist.\n", cacheFilename);
    }
    else if (status == LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
    {
        printf("Cannot read cache: Please attach at least %d cachelines to the handle.\n", numCachelines);
    }

    // Create Tensor Descriptors
    lwtensorTensorDescriptor_t descA;
    HANDLE_ERROR( lwtensorInitTensorDescriptor( &handle,
                &descA,
                nmodeA,
                extentA.data(),
                NULL,/*stride*/
                typeA, LWTENSOR_OP_IDENTITY ) );

    lwtensorTensorDescriptor_t descB;
    HANDLE_ERROR( lwtensorInitTensorDescriptor( &handle,
                &descB,
                nmodeB,
                extentB.data(),
                NULL,/*stride*/
                typeB, LWTENSOR_OP_IDENTITY ) );

    lwtensorTensorDescriptor_t descC;
    HANDLE_ERROR( lwtensorInitTensorDescriptor( &handle,
                &descC,
                nmodeC,
                extentC.data(),
                NULL,/*stride*/
                typeC, LWTENSOR_OP_IDENTITY ) );

    printf("Initialize lwTENSOR and tensor descriptors\n");

    /* ***************************** */

    //Retrieve the memory alignment for each tensor
    uint32_t alignmentRequirementA;
    HANDLE_ERROR( lwtensorGetAlignmentRequirement( &handle,
               A_d,
               &descA,
               &alignmentRequirementA) );

    uint32_t alignmentRequirementB;
    HANDLE_ERROR( lwtensorGetAlignmentRequirement( &handle,
               B_d,
               &descB,
               &alignmentRequirementB) );

    uint32_t alignmentRequirementC;
    HANDLE_ERROR( lwtensorGetAlignmentRequirement( &handle,
               C_d,
               &descC, 
               &alignmentRequirementC) );

    printf("Query best alignment requirement for our pointers\n");

    /* ***************************** */

    // Create the Contraction Descriptor
    lwtensorContractionDescriptor_t desc;
    HANDLE_ERROR( lwtensorInitContractionDescriptor( &handle, 
                &desc,
                &descA, modeA.data(), alignmentRequirementA,
                &descB, modeB.data(), alignmentRequirementB,
                &descC, modeC.data(), alignmentRequirementC,
                &descC, modeC.data(), alignmentRequirementC,
                typeCompute) );

    printf("Initialize contraction descriptor\n");

    /* ***************************** */

    // Set the algorithm to use
    lwtensorContractionFind_t find;
    HANDLE_ERROR( lwtensorInitContractionFind( 
                &handle, &find, 
                LWTENSOR_ALGO_DEFAULT) );

    const lwtensorAutotuneMode_t autotuneMode = LWTENSOR_AUTOTUNE_INCREMENTAL;
    HANDLE_ERROR(lwtensorContractionFindSetAttribute(
        &handle,
        &find,
        LWTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE,
        &autotuneMode ,
        sizeof(lwtensorAutotuneMode_t)));

    const uint32_t incCount = 4;
    HANDLE_ERROR(lwtensorContractionFindSetAttribute(
        &handle,
        &find,
        LWTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT,
        &incCount,
        sizeof(uint32_t)));

    printf("Initialize settings to find algorithm\n");

    /* ***************************** */

    // Query workspace
    size_t worksize = 0;
    HANDLE_ERROR( lwtensorContractionGetWorkspace(&handle,
                &desc,
                &find,
                LWTENSOR_WORKSPACE_RECOMMENDED, &worksize ) );

    // Allocate workspace
    void *work = nullptr;
    if(worksize > 0)
    {
        if( lwdaSuccess != lwdaMalloc(&work, worksize) ) // This is optional!
        {
            work = nullptr;
            worksize = 0;
        }
    }

    printf("Query recommended workspace size and allocate it\n");

    /* ***************************** */
    printf("Execute contraction from plan\n");

    int numRuns = 5;
    for(int i=0; i < numRuns; ++i)
    {
       // Create Contraction Plan && look-up cache (if attached)
       lwtensorContractionPlan_t plan;
       HANDLE_ERROR( lwtensorInitContractionPlan(&handle,
                                                 &plan,
                                                 &desc,
                                                 &find,
                                                 worksize) );

       printf("Create plan for contraction\n");

       /* ***************************** */

       lwtensorStatus_t err;

       // Execute the tensor contraction
       err = lwtensorContraction(&handle,
                                 &plan,
                          (void*)&alpha, A_d,
                                         B_d,
                          (void*)&beta,  C_d,
                                         C_d, 
                                 work, worksize, 0 /* stream */);
       lwdaDeviceSynchronize();

       // Check for errors
       if(err != LWTENSOR_STATUS_SUCCESS)
       {
           printf("ERROR: %s\n", lwtensorGetErrorString(err));
       }
    }

    uint32_t tag = 1;
    HANDLE_ERROR( lwtensorContractionDescriptorSetAttribute(
        &handle,
        &desc,
        LWTENSOR_CONTRACTION_DESCRIPTOR_TAG,
        &tag,
        sizeof(uint32_t)));

    for(int i=0; i < numRuns; ++i)
    {
       // Create Contraction Plan && look-up cache (if attached)
       lwtensorContractionPlan_t plan;
       HANDLE_ERROR( lwtensorInitContractionPlan(&handle,
                                                 &plan,
                                                 &desc,
                                                 &find,
                                                 worksize) );

       printf("Create plan for contraction\n");

       /* ***************************** */

       lwtensorStatus_t err;

       // Execute the tensor contraction
       err = lwtensorContraction(&handle,
                                 &plan,
                          (void*)&alpha, A_d,
                                         B_d,
                          (void*)&beta,  C_d,
                                         C_d, 
                                 work, worksize, 0 /* stream */);
       lwdaDeviceSynchronize();

       // Check for errors
       if(err != LWTENSOR_STATUS_SUCCESS)
       {
           printf("ERROR: %s\n", lwtensorGetErrorString(err));
       }
    }


    /* ***************************** */
    HANDLE_ERROR( lwtensorHandleWriteCacheToFile(&handle, cacheFilename) );
    printf("Cache has been successfully written to file (%s).\n", cacheFilename);

    // Detach cache and free-up resources
    HANDLE_ERROR( lwtensorHandleDetachPlanCachelines(&handle) );

    if ( A ) free( A );
    if ( B ) free( B );
    if ( C ) free( C );
    if ( cachelines ) free(cachelines);
    if ( A_d ) lwdaFree( A_d );
    if ( B_d ) lwdaFree( B_d );
    if ( C_d ) lwdaFree( C_d );
    if ( work ) lwdaFree( work );

    printf("Successful completion\n");

    return 0;
  }


You can confirm that the cache has two entries now by ilwoking the binary once again;
this time it should report that "*2 cachelines have been successfully read from
file (./cache.bin)*".

This concludes our example of the plan cache; you can find these examples (including timings and
warm-up runs) in the `samples repository <https://github.com/LWPU/LWDALibrarySamples/tree/master/lwTENSOR>`_.

If you have any further question or suggestions, please do not hesitate to reach out.
