
.. role:: cpp(code)
   :language: cpp
   :class: highlight

.. _getting-started-label:

Getting Started
===============

In this section, we show how to implement a first tensor contraction using lwTENSOR.
Our code will compute the following operation using single-precision arithmetic.

.. math::

  C_{m,u,n,v} = \alpha A_{m,h,k,n} B_{u,k,v,h} + \beta C_{m,u,n,v}

We build the code up step by step, each step adding code at the end.
The steps are separated by comments consisting of multiple stars.

Installation and Compilation
----------------------------

The download for lwTENSOR is available `here
<https://developer.lwpu.com/lwtensor/downloads>`_.
Assuming lwTENSOR has been downloaded to *LWTENSOR_DOWNLOAD_FILE*, we extract it, remember lwTENSOR's
root location, and update the library path accordingly (in this example we are using the
10.1 version, depending on your toolkit, you might have to choose a different version).

.. code-block::

  tar xf ${LWTENSOR_DOWNLOAD_FILE}
  export LWTENSOR_ROOT=${PWD}/liblwtensor
  export LD_LIBRARY_PATH=${LWTENSOR_ROOT}/lib/10.1/:${LD_LIBRARY_PATH}

If we store the  following code in a file called *contraction.lw*, we can compile it via the following
command:

.. code-block::

  lwcc contraction.lw -L${LWTENSOR_ROOT}/lib/10.1/ -I${LWTENSOR_ROOT}/include -std=c++11 -llwtensor -o contraction


When compiling intermediate steps of this example, the compiler might warn about unused variables. This is 
due to the example not being complete. The final step should issue no warnings.

Headers and Data Types
----------------------

To start off, we begin with a trivial :code:`main()` function, include some headers, and define some data types.

.. code-block:: cpp

  #include <stdlib.h>
  #include <stdio.h>

  #include <lwda_runtime.h>
  #include <lwtensor.h>

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

    return 0;
  }

Define Tensor Sizes
---------------------

Next, we define the modes and extents of our tensors.
For the sake of the example, let us say that the modes :math:`m`,
:math:`n` and :math:`u` have an extent of 96; and that :math:`v`,
:math:`h` and :math:`k` have an extent of 64.
Note how modes are labeled by integers and how that allows us to
label the modes using characters.
See :ref:`nomenclature-label` for an explanation of the terms mode
and extent.

.. code-block:: cpp

  #include <stdlib.h>
  #include <stdio.h>

  #include <lwda_runtime.h>
  #include <lwtensor.h>

  #include <unordered_map>
  #include <vector>

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

    return 0;
  }

Initialize Tensor Data
----------------------

Next, we need to allocate and initialize host and device memory for our tensors:

.. code-block:: cpp

  #include <stdlib.h>
  #include <stdio.h>

  #include <lwda_runtime.h>
  #include <lwtensor.h>

  #include <unordered_map>
  #include <vector>

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

    return 0;
  }

Create Tensor Descriptors
-----------------------------

We are now ready to use the lwTENSOR library:
We add a macro to handle errors and initialize the lwTENSOR by creating a handle.
Then, we create a descriptor for each tensor by providing its data type, order, data type, and element-wise operation.
The latter sets an element-wise operation that is applied to that tensor when it is used during computation.
In this case, the operator is the identity; see :ref:`lwtensorOperator-label` for other possibilities.


.. code-block:: cpp

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

    floatTypeCompute alpha = (floatTypeCompute) 1.1f;
    floatTypeCompute beta  = (floatTypeCompute) 0.9f;

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

    return 0;
  }

Create Contraction Descriptor
-----------------------------

In the next step we query the library to find out the best alignment requirements that the data
pointer allow for and create the descriptor for the contraction problem:

.. code-block:: cpp

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

    return 0;
  }

Determine Algorithm and Workspace
---------------------------------

Now that we have defined both the tensors and the contraction that we want to perform, we
must select an algorithm to perform the contraction.
That algorithm is specified by :ref:`lwtensorAlgo-label`.
Specifying :code:`LWTENSOR_ALGO_DEFAULT` allows us to let lwTENSOR's internal heuristic
choose the best approach.
All the information to find a good algorithm is stored in the :ref:`lwtensorFind-label`
data structure.
We can also query the library to find the amount of workspace required to have the most
options when selecting an algorithms.
While workspace memory is not mandatory, it is required for some algorithms.

.. code-block:: cpp

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

    return 0;
  }

Plan and Execute
---------------------------------

Finally, we are ready to create the contraction plan and execute the tensor contraction:

.. code-block:: cpp

  #include <stdlib.h>
  #include <stdio.h>

  #include <lwda_runtime.h>
  #include <lwtensor.h>

  #include <unordered_map>
  #include <vector>

  // Handle lwTENSOR errors
  #define HANDLE_ERROR(x) {                                                              \
    const auto err = x;                                                                  \
    if( err != LWTENSOR_STATUS_SUCCESS )                                                 \
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

    // Create Contraction Plan
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

    printf("Execute contraction from plan\n");

    /* ***************************** */

    if ( A ) free( A );
    if ( B ) free( B );
    if ( C ) free( C );
    if ( A_d ) lwdaFree( A_d );
    if ( B_d ) lwdaFree( B_d );
    if ( C_d ) lwdaFree( C_d );
    if ( work ) lwdaFree( work );

    printf("Successful completion\n");

    return 0;
  }


That is it. We have run our first lwTENSOR contraction! You can find this and other examples in the
`samples repository <https://github.com/LWPU/LWDALibrarySamples/tree/master/lwTENSOR>`_.
