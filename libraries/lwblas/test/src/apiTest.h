#include <stdio.h>

#include <lwda_runtime.h>
#include <lwda_fp16.h>
//#include <lwToolsExt.h>

#include <unordered_map>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <list>

#include <lwtensor.h>
#include <lwtensor/internal/util.h>
#include <lwtensor/internal/lwtensor.h>
#include <lwtensor/internal/operators.h>
#include <lwtensor/internal/defines.h>
#include "elementwise_ref.h"
#include "gtest/gtest.h"

#pragma once

typedef int mode_type_external;

using namespace LWTENSOR_NAMESPACE;

#define REL(a,b) (std::abs((a)-(b))/std::max((a),(b)))

inline
lwdaError_t lwdaCheck(lwdaError_t result)
{
  if (result != lwdaSuccess) {
    fprintf(stderr, "LWCA Runtime Error: %s(%d)\n", lwdaGetErrorString(result), result);
    assert(result == lwdaSuccess);
  }
  return result;
}

/** Generate random number in uniform [a, b] distribution. */
    template<typename typeCompute>
typeCompute UniformRandomNumber( double a, double b )
{
    /** Generate uniform [ 0, 1 ] first. */
    auto u01 = (double)( rand() % 1000000 ) / 1000000;
    /** Scale and shift. */
    return ( b - a ) * u01 + a;
}

/** Specialization for half. */
    template<>
half UniformRandomNumber( double a, double b )
{
    return lwGet<half>( UniformRandomNumber<double>( a, b ) );
}

    template<>
int8_t UniformRandomNumber( double a, double b )
{
    auto u01 = (double)( rand() % 1000000 ) / 1000000;
    return int8_t(( b - a ) * u01 + a);
}

    template<>
int32_t UniformRandomNumber( double a, double b )
{
    auto u01 = (double)( rand() % 1000000 ) / 1000000;
    return int32_t(( b - a ) * u01 + a);
}

/** Specialization for single complex. */
    template<>
lwComplex UniformRandomNumber( double a, double b )
{
    lwComplex x;
    x.x = UniformRandomNumber<double>( a, b );
    x.y = UniformRandomNumber<double>( a, b );
    return x;
}

/** Specialization for double complex. */
    template<>
lwDoubleComplex UniformRandomNumber( double a, double b )
{
    lwDoubleComplex x;
    x.x = UniformRandomNumber<double>( a, b );
    x.y = UniformRandomNumber<double>( a, b );
    return x;
}


    template<typename typeCompute>
double RelativeError( typeCompute lhs, typeCompute rhs )
{
    //auto x = Operator<typeCompute, typeCompute, typeCompute, LWTENSOR_OP_SUB>::execute( lhs, rhs );
    auto x = lwSub( lhs, rhs );
    auto y = Operator<typeCompute, typeCompute, typeCompute, LWTENSOR_OP_MAX>::execute( lhs, rhs );
    return std::sqrt( lwSquare2Norm( x ) / lwSquare2Norm( y ) );
}


    template<typename typeCompute>
double SquareError( typeCompute lhs, typeCompute rhs )
{
    //auto x = Operator<typeCompute, typeCompute, typeCompute, LWTENSOR_OP_SUB>::execute( lhs, rhs );
    auto x = lwSub( lhs, rhs );
    return lwSquare2Norm( x );
}

    template<typename typeA>
double verify( typeA *A, typeA *B, size_t m )
{
    double norm = 0;
    double normA = 0;
    double normB = 0;
    float error = 0;
    double threshold = (sizeof(typeA) == 2) ? 1e-2 : 1e-4;
    for ( int i = 0; i < m; i ++ )
    {
        auto a = A[ i ];
        auto b = B[ i ];
        auto rel = RelativeError( a, b );
        if ( rel > threshold )
        {
            if ( error <= 0 ) printf("log: %d %e\n", i, (float)rel );
            error += 1.;
        }
        norm  += SquareError( a, b );
        normA += lwSquare2Norm( a );
        normA += lwSquare2Norm( b );
    }

    /** Compute relative error. */
    norm = std::sqrt(norm);
    normA = std::sqrt(normA);
    normB = std::sqrt(normB);
    double relError = norm/(std::max(normA,normB));
    //	printf("%e %e\n", normA, normB);

    /** Return the relative error while exceeding the threshold. */
    if ( relError > threshold || error > 0 || relError != relError ) return relError;

    /** Otherwise return 0. */
    return 0;
}

double verify( void*A, void *B, size_t m, lwdaDataType_t type )
{
    switch ( type )
    {
        case LWDA_R_8I:
            {
                return verify<int8_t>( (int8_t*)A, (int8_t*)B, m );
            }
        case LWDA_R_16F:
            {
                return verify<half>( (half*)A, (half*)B, m );
            }
        case LWDA_R_32F:
            {
                return verify<float>( (float*)A, (float*)B, m );
            }
        case LWDA_R_64F:
            {
                return verify<double>( (double*)A, (double*)B, m );
            }
        case LWDA_C_32F:
            {
                return verify<lwComplex>( (lwComplex*)A, (lwComplex*)B, m );
            }
        case LWDA_C_64F:
            {
                return verify<lwDoubleComplex>( (lwDoubleComplex*)A, (lwDoubleComplex*)B, m );
            }
        case LWDA_R_32I:
            {
                return verify<int32_t>( (int32_t*)A, (int32_t*)B, m );
            }
        default:
            {
                printf("TYPE UNKNOWN\n");
                exit(0);
            }
    }
}

template<typename typeA>
void print_matrix(const typeA *A, int m, int n, char* str)
{
    printf("%s:\n",str);
    for(int i=0; i < m; ++i){
        for(int j=0; j < n; ++j)
            std::cout<< A[i + j * m] << " ";
        printf("\n");
    }
}

//void initModes(std::vector<mode_type_external> &tokens, const std::vector<std::string>&& vec)
//{
//    for(auto s : vec)
//        tokens.emplace_back(s[0]);
//}
std::vector<std::string> split(const char* str, const char* delim)
{
    std::vector<std::string> tokens;
    char * pch;
    char buffer[256];
    strcpy(buffer, str);
    pch = strtok (buffer, delim);
    while (pch != NULL)
    {
        tokens.push_back(std::string(pch));
        pch = strtok (NULL, delim);
    }
    return tokens;
}

    template<typename typeOut>
__device__ typeOut myUnaryIdentity(typeOut x, const void *args)
{
    return x;
}

    template<typename typeOut>
__device__ typeOut myBinaryAdd(typeOut x, typeOut y, const void *args)
{
    return x + y;
}

template<typename T>
using myFptrUnary = T (*)(T,const void*);
template<typename T>
using myFptrBinary = T (*)(T,T,const void*);

    template<typename T>
__global__ void init( myFptrUnary<T> *opA, myFptrBinary<T> *opAB)
{
    *opA = myUnaryIdentity<T>;
    *opAB= myBinaryAdd<T>;
}
char typeToString( lwdaDataType_t type)
{
    switch(type)
    {
        case LWDA_R_8I:
            return 'i';
        case LWDA_R_16F:
            return 'h';
        case LWDA_R_32F:
            return 's';
        case LWDA_R_64F:
            return 'd';
        case LWDA_C_32F:
            return 'c';
        case LWDA_C_64F:
            return 'z';
        default:
            printf("ERROR: TYPE UNKNOWN. %d\n",type);
            exit(-1);
    }
}
void initType( lwdaDataType_t &type, char arg)
{
    switch(arg)
    {
        case 'i':
            type = LWDA_R_8I;
            break;
        case 'h':
            type = LWDA_R_16F;
            break;
        case 's':
            type = LWDA_R_32F;
            break;
        case 'd':
            type = LWDA_R_64F;
            break;
        case 'c':
            type = LWDA_C_32F;
            break;
        case 'z':
            type = LWDA_C_64F;
            break;
        default:
            printf("ERROR: TYPE UNKNOWN. %c\n",arg);
            exit(-1);
    }
}

void initialize( void* A, lwdaDataType_t typeA, size_t numElements, float value = 999 )
{
    double a = 0.0;
    double b = 0.1;
    if ( value != 999 )
    {
        a = value;
        b = value;
    }

    for ( int64_t i = 0; i < numElements; i++)
    {
        switch ( typeA )
        {
            case LWDA_R_8I:
                {
                    //printf( "ERROR: NO PROPER RANDOM NUMBER GENERATOR %d\n", typeA );
                    //exit(-1);
                    ((int8_t*)A)[ i ] = UniformRandomNumber<int8_t>( a, b );
                    break;
                }
            case LWDA_R_16F:
                {
                    ((half*)A)[ i ] = UniformRandomNumber<half>( a, b );
                    break;
                }
            case LWDA_R_32F:
                {
                    ((float*)A)[ i ] = UniformRandomNumber<float>( a, b );
                    break;
                }
            case LWDA_R_64F:
                {
                    ((double*)A)[ i ] = UniformRandomNumber<double>( a, b );
                    break;
                }
            case LWDA_C_32F:
                {
                    ((lwComplex*)A)[ i ] = UniformRandomNumber<lwComplex>( a, b );
                    break;
                }
            case LWDA_C_64F:
                {
                    ((lwDoubleComplex*)A)[ i ] = UniformRandomNumber<lwDoubleComplex>( a, b );
                    break;
                }
            case LWDA_R_32I:
                {
                    ((int32_t*)A)[ i ] = UniformRandomNumber<int32_t>( a, b );
                    break;
                }
            default:
                {
                    printf( "ERROR: TYPE UNKNOWN. %d\n", typeA );
                    exit(-1);
                }
        }
    }
}



#define SWITCH_CHAR '-'

typedef enum { LWTENSOR_PW, LWTENSOR_TC, UNKNOWN } Routine;

struct TestOptions
{
    TestOptions() : typeA(LWDA_R_32F), typeB(LWDA_R_32F), typeC(LWDA_R_32F),
    typeCompute(LWDA_R_32F), routine(UNKNOWN), autotuning(true),
    disableVerification(false), deviceId(0), verbose(false)
    {
        opA = LWTENSOR_OP_IDENTITY;
        opB = LWTENSOR_OP_IDENTITY;
        opC = LWTENSOR_OP_IDENTITY;
        opAB = LWTENSOR_OP_ADD;
        opUnaryAfterBinary = LWTENSOR_OP_IDENTITY;
        opABC = LWTENSOR_OP_ADD;
        conjA = false;
        conjB = false;
        conjC = false;
        vectorWidthA = 1;
        vectorWidthB = 1;
        vectorWidthC = 1;
        vectorModeA  = 0;
        vectorModeB  = 0;
        vectorModeC  = 0;
        strideA = nullptr;
        strideB = nullptr;
        strideC = nullptr;
    }

    ~TestOptions() {}

    int deviceId;
    bool verbose;
    bool conjA;
    bool conjB;
    bool conjC;
    bool autotuning;
    bool disableVerification;
    std::vector<mode_type_external> modeA;
    std::vector<mode_type_external> modeB;
    std::vector<mode_type_external> modeC;
    std::unordered_map<mode_type_external, uint64_t> extent;
    lwdaDataType_t typeA;
    lwdaDataType_t typeB;
    lwdaDataType_t typeC;
    lwdaDataType_t typeCompute;
    Routine routine;
    lwtensorOperator_t opA;
    lwtensorOperator_t opB;
    lwtensorOperator_t opC;
    lwtensorOperator_t opAB;
    lwtensorOperator_t opUnaryAfterBinary;
    lwtensorOperator_t opABC;

    void * alpha = nullptr, *beta = nullptr, *gamma = nullptr;

    int vectorWidthA;
    int vectorWidthB;
    int vectorWidthC;

    int vectorModeA;
    int vectorModeB;
    int vectorModeC;

    int64_t *strideA;
    int64_t *strideB;
    int64_t *strideC;

    void print()
    {
        printf("typeA: %c\n", typeToString(typeA));
        printf("typeB: %c\n", typeToString(typeB));
        printf("typeC: %c\n", typeToString(typeC));
        printf("typeCompute: %c\n", typeToString(typeCompute));
        printf("alpha: %p\n", alpha);
        printf("beta: %p\n", beta);
        printf("gamma: %p\n", gamma);
        printf("modeC: ");
        for(auto mode : modeC)
            printf("%c ", mode);
        printf("\nmodeA: ");
        for(auto mode : modeA)
            printf("%c ", mode);
        printf("\nmodeB: ");
        for(auto mode : modeB)
            printf("%c ", mode);
        printf("\nExtents: ");
        for(auto ex : extent)
            printf("%c=%ld ", ex.first,ex.second);
        printf("\n");
    }
};

typedef struct TestOptions TestOptions_t;

namespace APITESTING {
    using namespace LWTENSOR_NAMESPACE;
#define DefaultModeAndExtent() do { \
    opts.modeA = {'c','b','a'};     \
    opts.modeB = {'c','a','b'};     \
    opts.modeC = {'a','b','c','d'}; \
    opts.extent.insert({'a', 40});  \
    opts.extent.insert({'b', 20});  \
    opts.extent.insert({'c', 30});  \
    opts.extent.insert({'d', 30}); } while(0)

    class ApiTestBase
    {
        protected:
            ApiTestBase() {}

            ~ApiTestBase() {}

            virtual void OptsInitialization()  = 0;

            void SetUp()
            {
                // Code here will be called immediately after the constructor (right
                // before each test).
                std::cout<<"[ Setup    ] "<<"opts initializtion."<<std::endl;
                std::cout<<"[ Setup    ] "<<"memory allocation and  initializtion."<<std::endl;
                std::cout<<"[ Setup    ] "<<"create lwTensor descriptor."<<std::endl;

                lwdaStreamCreateWithFlags(&pStream, lwdaStreamNonBlocking);

                //opts initialization, implemented in the derived class
                OptsInitialization();

                //initialization of alpha, beta and gamma
                opts.alpha = (void *)malloc(1 * getDataTypeSize(opts.typeCompute));
                opts.beta = (void *)malloc(1 * getDataTypeSize(opts.typeCompute));
                opts.gamma = (void *)malloc(1 * getDataTypeSize(opts.typeCompute));
                initialize(opts.alpha, opts.typeCompute, 1, 2.0f);
                ///WIP: the support for B has been temporarily deactivated.
                initialize(opts.beta, opts.typeCompute, 1, 0.0f);
                initialize(opts.gamma, opts.typeCompute, 1, 1.0f);

//                lwdaSetDevice(opts.deviceId);

                size_t elementSizeA = getDataTypeSize(opts.typeA);
                size_t elementSizeB = getDataTypeSize(opts.typeB);
                size_t elementSizeC = getDataTypeSize(opts.typeC);
                size_t elementSizeCompute = getDataTypeSize(opts.typeCompute);

                int nmodeA = opts.modeA.size();
                int nmodeB = opts.modeB.size();
                int nmodeC = opts.modeC.size();

                size_t elementsA = 1;
                for (int i = 0; i < nmodeA; ++i)
                    elementsA *= opts.extent[opts.modeA[i]];
                size_t elementsB = 1;
                for (int i = 0; i < nmodeB; ++i)
                    elementsB *= opts.extent[opts.modeB[i]];
                size_t elementsC = 1;
                for (int i = 0; i < nmodeC; ++i)
                    elementsC *= opts.extent[opts.modeC[i]];

                sizeA = elementSizeA * elementsA;
                sizeB = elementSizeB * elementsB;
                sizeC = elementSizeC * elementsC;

                lwdaMalloc((void**)&A_d, sizeA);
                lwdaMalloc((void**)&B_d, sizeB);
                lwdaMalloc((void**)&C_d, sizeC);
                lwdaMalloc((void**)&C_ref_d, sizeC);
                lwdaMalloc((void**)&D_d, sizeC);
                {
                    auto err = lwdaGetLastError();
                    if(err != lwdaSuccess)
                    {
                        fprintf(stderr, "Error: %s in %s at line %d\n", lwdaGetErrorString(err), __FILE__, __LINE__);
                        exit(-1);
                    }
                }

                A = malloc(sizeA);
                B = malloc(sizeB);
                C = malloc(sizeC);
                D = malloc(sizeC);
                C_ref = malloc(sizeC);
                C_copy = malloc(sizeC);

                initialize( A, opts.typeA, elementsA );
                initialize( B, opts.typeB, elementsB );
                initialize( C, opts.typeC, elementsC );

                memcpy(C_copy, C, elementSizeC * elementsC);
                memcpy(C_ref, C, elementSizeC * elementsC);

                lwdaMemcpy2DAsync(C_ref_d, sizeC, C, sizeC, sizeC, 1, lwdaMemcpyDefault, pStream);
                lwdaMemcpy2DAsync(D_d, sizeC, C, sizeC, sizeC, 1, lwdaMemcpyDefault, pStream);
                lwdaMemcpy2DAsync(C_d, sizeC, C, sizeC, sizeC, 1, lwdaMemcpyDefault, pStream);
                lwdaMemcpy2DAsync(B_d, sizeB, B, sizeB, sizeB, 1, lwdaMemcpyDefault, pStream);
                lwdaMemcpy2DAsync(A_d, sizeA, A, sizeA, sizeA, 1, lwdaMemcpyDefault, pStream);
                lwdaStreamSynchronize(pStream);

                std::vector<int64_t> extentA;
                for(auto mode : opts.modeA)
                    extentA.push_back(opts.extent[mode]);
                std::vector<int64_t> extentB;
                for(auto mode : opts.modeB)
                    extentB.push_back(opts.extent[mode]);
                std::vector<int64_t> extentC;
                for(auto mode : opts.modeC)
                    extentC.push_back(opts.extent[mode]);

                EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);

                EXPECT_EQ( lwtensorInitTensorDescriptor( &handle, &descA, nmodeA, &extentA[0],
                            opts.strideA, opts.typeA, opts.opA, opts.vectorWidthA, opts.vectorModeA), LWTENSOR_STATUS_SUCCESS );
                if (opts.vectorWidthA > 1)
                {
                    FAIL() << "Vectorization not supported.";
                }

                EXPECT_EQ( lwtensorInitTensorDescriptor( &handle, &descB, nmodeB, &extentB[0],
                            opts.strideB, opts.typeB, opts.opB, opts.vectorWidthB, opts.vectorModeB), LWTENSOR_STATUS_SUCCESS );
                if (opts.vectorWidthA > 1)
                {
                    FAIL() << "Vectorization not supported.";
                }

                EXPECT_EQ( lwtensorInitTensorDescriptor( &handle, &descC, nmodeC, &extentC[0],
                            opts.strideC, opts.typeC, opts.opC, opts.vectorWidthC, opts.vectorModeC), LWTENSOR_STATUS_SUCCESS );
                if (opts.vectorWidthA > 1)
                {
                    FAIL() << "Vectorization not supported.";
                }

                //EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
                /*struct lwtensorContext_t *ctx = (struct lwtensorContext_t *)handle;*/
                /*printf("dsds");*/
            }

            void TearDown() {
                std::cout<<"[ TearDown ] "<<"destroy lwTensor."<<std::endl;
                std::cout<<"[ TearDown ] "<<"free allocated memory."<<std::endl;
                // Code here will be called immediately after each test (right
                // before the destructor).

                vectorizedModes.clear();

                if(C_copy) free(C_copy);
                if(C_ref) free(C_ref);
                if(D) free(D);
                if(C) free(C);
                if(B) free(B);
                if(A) free(A);

                if(A_d) lwdaFree(A_d);
                if(B_d) lwdaFree(B_d);
                if(C_d) lwdaFree(C_d);
                if(C_ref_d) lwdaFree(C_ref_d);
                if(D_d) lwdaFree(D_d);

                if (opts.alpha)
                    free(opts.alpha);
                if(opts.beta)
                    free(opts.beta);
                if(opts.gamma)
                    free(opts.gamma);

                lwdaStreamDestroy(pStream);
            }

            // Objects declared here can be used by all tests in the test case for Foo.
        public:
            size_t sizeA, sizeB, sizeC;
            void *A_d,  *B_d, *C_d, *D_d, *C_ref_d;
            void *A, *B, *C, *C_ref, *C_copy, *D;
            lwtensorTensorDescriptor_t descA;
            lwtensorTensorDescriptor_t descB;
            lwtensorTensorDescriptor_t descC;
            std::vector<mode_type> vectorizedModes;

            TestOptions_t opts;
            lwtensorHandle_t handle;
            lwdaStream_t pStream;
    };

    class ApiTestDefault : public ApiTestBase, public ::testing::Test {
        protected:
            ApiTestDefault() {}
            ~ApiTestDefault() {}
            void OptsInitialization() {
                std::cout<<"[Initialize] "<<"define the default mode and extent."<<std::endl;
                DefaultModeAndExtent();
                opts.disableVerification = true;
            }

            void SetUp() override {
                /*OptsInitialization();*/
                ApiTestBase::SetUp();
            }

            void TearDown() override {
                ApiTestBase::TearDown();
            }
    };

    class ApiTestOperator : public ApiTestBase, public ::testing::TestWithParam<std::tuple<lwtensorOperator_t,
    lwtensorOperator_t, lwtensorOperator_t, lwtensorOperator_t, lwtensorOperator_t> > {
        protected:
            ApiTestOperator() {}
            ~ApiTestOperator() {}

            void OptsInitialization() override {
                std::cout<<"[Initialize] "<<"define the default mode and extent."<<std::endl;
                DefaultModeAndExtent();

                std::tie(opts.opA,
                        opts.opB,
                        opts.opC,
                        opts.opAB,
                        opts.opABC) = GetParam();
            }

            void SetUp() override {
                ApiTestBase::SetUp();
            }

            void TearDown() override {
                ApiTestBase::TearDown();
            }
    };

    class ApiTestTypes : public ApiTestBase, public ::testing::TestWithParam<std::vector<lwdaDataType_t> > {
        protected:
            ApiTestTypes() {}
            ~ApiTestTypes() {}

            void OptsInitialization() override {
                std::cout<<"[Initialize] "<<"define the default mode and extent."<<std::endl;
                DefaultModeAndExtent();

                std::vector<lwdaDataType_t> types;
                types = GetParam();
                opts.typeA = types[0];
                opts.typeB = types[1];
                opts.typeC = types[2];
                opts.typeCompute = types[3];
            }

            void SetUp() override {
                ApiTestBase::SetUp();
            }

            void TearDown() override {
                ApiTestBase::TearDown();
            }
    };



    class ApiTestStrides : public ApiTestBase, public ::testing::TestWithParam<int> {
        protected:
            ApiTestStrides() {
                strideA = (int64_t *) malloc(3 * sizeof(int64_t)); // 3 is nmodeA
                strideB = (int64_t *) malloc(3 * sizeof(int64_t)); // 3 is nmodeB
                strideC = (int64_t *) malloc(4 * sizeof(int64_t)); // 3 is nmodeC
            }
            ~ApiTestStrides() {
                if(strideA) free(strideA);
                if(strideB) free(strideB);
                if(strideC) free(strideC);
            }

            int randInt()
            {
                int data = GetParam();
                return ((rand()+data)%9 + 1);
            }

            void OptsInitialization() override {
                std::cout<<"[Initialize] "<<"define the default mode and extent."<<std::endl;

                DefaultModeAndExtent();

                strideA[opts.modeA.size() - 1] = randInt();
                strideB[opts.modeB.size() - 1] = randInt();
                strideC[opts.modeC.size() - 1] = randInt();

                for(int i = opts.modeA.size() - 2; i >= 0; i--)
                    strideA[i] = strideA[i+1] * opts.extent[opts.modeA[i+1]];

                for(int i = opts.modeB.size() - 2; i >= 0; i--)
                    strideB[i] = strideB[i+1] * opts.extent[opts.modeB[i+1]];

                for(int i = opts.modeC.size() - 2; i >= 0; i--)
                    strideC[i] = strideC[i+1] * opts.extent[opts.modeC[i+1]];

                opts.strideA = strideA;
                opts.strideB = strideB;
                opts.strideC = strideC;

                std::cout<<"The strides are: "<<std::endl;
                std::cout<<"StrideA: "<<strideA[0]<<" "<<strideA[1]<<" "<<strideA[2]<<std::endl;
                std::cout<<"StrideB: "<<strideB[0]<<" "<<strideB[1]<<" "<<strideB[2]<<std::endl;
                std::cout<<"StrideC: "<<strideC[0]<<" "<<strideC[1]<<" "<<strideC[2]<<" "<<strideC[3]<<std::endl;
            }

            void SetUp() override {
                ApiTestBase::SetUp();
            }

            void TearDown() override {
                ApiTestBase::TearDown();
            }

            int64_t *strideA;
            int64_t *strideB;
            int64_t *strideC;
    };

    class ApiTestCoefficient : public ApiTestBase, public ::testing::TestWithParam<std::tuple<float, float, float> > { //float is the default type for alpha/beta/gamma

        protected:
            ApiTestCoefficient() {}
            ~ApiTestCoefficient() {}
            void OptsInitialization() override {
                std::cout<<"[Initialize] "<<"define the mode and extent."<<std::endl;
                opts.modeA = {'c','b','a'};
                opts.modeB = {'c'}; //reduced to extent 1
                opts.modeC = {'a','b','c','d'};
                opts.extent.insert({'a', 40});
                opts.extent.insert({'b', 20});
                opts.extent.insert({'c', 30});
                opts.extent.insert({'d', 30});

            }
            void SetUp() override {
                ApiTestBase::SetUp();

                //assignment of alpha should be put after the SetUp()
                float alpha, beta, gamma;
                std::tie(alpha,
                        beta,
                        gamma) = GetParam();
                std::cout<<"[Initialize] "<<"initialize alpha, beta and gamma."<<std::endl;
                initialize(opts.alpha, opts.typeCompute, 1, alpha);
                initialize(opts.beta, opts.typeCompute, 1, beta);
                initialize(opts.gamma, opts.typeCompute, 1, gamma);
            }
            void TearDown() override {
                ApiTestBase::TearDown();
            }
    };

    class ApiTestOperatorNegative : public ApiTestBase, public ::testing::TestWithParam<std::tuple<lwtensorOperator_t,
    lwtensorOperator_t, lwtensorOperator_t, lwtensorOperator_t, lwtensorOperator_t> >        {
        protected:
            ApiTestOperatorNegative() {}
            ~ApiTestOperatorNegative() {}
            void OptsInitialization() override {
                std::cout<<"[Initialize] "<<"define the mode and extent."<<std::endl;
                DefaultModeAndExtent();
                std::tie(opts.opA,
                        opts.opB,
                        opts.opC,
                        opts.opAB,
                        opts.opABC) = GetParam();
            }
            void SetUp() override {
                ApiTestBase::SetUp();
            }
            void TearDown() override {
                ApiTestBase::TearDown();
            }
    };

    class ApiTestTypeNegative : public ApiTestBase, public ::testing::TestWithParam<std::vector<lwdaDataType_t> > {
        protected:
            ApiTestTypeNegative() {}
            ~ApiTestTypeNegative() {}
            void OptsInitialization() override {
                std::cout<<"[Initialize] "<<"define the mode and extent."<<std::endl;
                DefaultModeAndExtent();

                std::vector<lwdaDataType_t> types;
                types = GetParam();
                opts.typeA = types[0];
                opts.typeB = types[1];
                opts.typeC = types[2];
                opts.typeCompute = types[3];
            }

            void SetUp() override {
                ApiTestBase::SetUp();
            }
            void TearDown() override {
                ApiTestBase::TearDown();
            }
    };

    //negative test for beta
    class ApiTestBetaNegative : public ApiTestBase, public ::testing::Test {
        protected:
            ApiTestBetaNegative() {}
            ~ApiTestBetaNegative() {}
            void OptsInitialization() override {
                std::cout<<"[Initialize] "<<"define the mode and extent."<<std::endl;
                DefaultModeAndExtent();
            }
            void SetUp() override {
                ApiTestBase::SetUp();
                float beta = 1.5f;
                initialize(opts.beta, opts.typeCompute, 1, beta);
            }
            void TearDown() override {
                ApiTestBase::TearDown();
            }
    };

    class ApiTestNegative : public ApiTestBase, public ::testing::Test {
        protected:
            ApiTestNegative() {}
            ~ApiTestNegative() {}
            void OptsInitialization() override {
                std::cout<<"[Initialize] "<<"define the mode and extent."<<std::endl;
                DefaultModeAndExtent();
            }
            void SetUp() override {
                ApiTestBase::SetUp();
            }
            void TearDown() override {
                ApiTestBase::TearDown();
            }
    };

    class ApiTestOneModeC : public ApiTestBase, public ::testing::TestWithParam<std::vector<lwdaDataType_t> > {
        protected:
            ApiTestOneModeC() {}
            ~ApiTestOneModeC() {}
            void OptsInitialization() override {
                std::cout<<"[Initialize] "<<"define the mode and extent."<<std::endl;
                opts.modeA = {'a'};
                opts.modeB = {'a'};
                opts.modeC = {'a'};
                opts.extent.insert({'a', 40});

                std::vector<lwdaDataType_t> types;
                types = GetParam();
                opts.typeA = types[0];
                opts.typeB = types[1];
                opts.typeC = types[2];
                opts.typeCompute = types[3];
            }
            void SetUp() override {
                ApiTestBase::SetUp();
            }
            void TearDown() override {
                ApiTestBase::TearDown();
            }
    };

    class ApiTestSmallExtent : public ApiTestBase, public ::testing::TestWithParam<std::vector<lwdaDataType_t> > {
        protected:
            ApiTestSmallExtent() {}
            ~ApiTestSmallExtent() {}
            void OptsInitialization() override {
                std::cout<<"[Initialize] "<<"define the mode and extent."<<std::endl;
                opts.modeA = {'b', 'c'};
                opts.modeB = {'a'};
                opts.modeC = {'a', 'b', 'c'};
                opts.extent.insert({'a', 10});
                opts.extent.insert({'b', 1});
                opts.extent.insert({'c', 1});

                std::vector<lwdaDataType_t> types;
                types = GetParam();
                opts.typeA = types[0];
                opts.typeB = types[1];
                opts.typeC = types[2];
                opts.typeCompute = types[3];
            }
            void SetUp() override {
                ApiTestBase::SetUp();
            }
            void TearDown() override {
                ApiTestBase::TearDown();
            }
    };

    class PermutationTestBase {
        protected:
            PermutationTestBase() {}
            ~PermutationTestBase() {}

            virtual void OptsInitialization()  = 0;

            void SetUp() {
                std::cout<<"[ Setup    ] "<<"opts initializtion."<<std::endl;
                std::cout<<"[ Setup    ] "<<"memory allocation and  initializtion."<<std::endl;
                std::cout<<"[ Setup    ] "<<"create lwTensor descriptor."<<std::endl;

                lwdaStreamCreateWithFlags(&pStream, lwdaStreamNonBlocking);
                OptsInitialization();

                opts.alpha = (void *)malloc(1 * getDataTypeSize(opts.typeCompute));
                initialize(opts.alpha, opts.typeCompute, 1);
//                lwdaSetDevice(opts.deviceId);

                size_t elementSizeA = getDataTypeSize(opts.typeA);
                int nmodeA = opts.modeA.size();
                size_t elementsA = 1;
                for (int i = 0; i < nmodeA; ++i) {
                    elementsA *= opts.extent[opts.modeA[i]];
                }
                sizeA = elementSizeA * elementsA;

                size_t elementSizeB = getDataTypeSize(opts.typeB);
                int nmodeB = opts.modeB.size();
                size_t elementsB = 1;
                for (int i = 0; i < nmodeB; ++i) {
                    elementsB *= opts.extent[opts.modeB[i]];
                }
                sizeB = elementSizeB * elementsB;

                lwdaMalloc((void**)&A_d, sizeA);
                lwdaMalloc((void**)&B_d, sizeB);

                A = malloc(sizeA);
                B = malloc(sizeB);
                B_ref = malloc(sizeB);
                initialize(A, opts.typeA, elementsA);
                initialize(B, opts.typeB, elementsB);
                initialize(B_ref, opts.typeB, elementsB);

                lwdaMemcpy2DAsync(A_d, sizeA, A, sizeA, sizeA, 1, lwdaMemcpyDefault, pStream);
                lwdaMemcpy2DAsync(B_d, sizeB, B, sizeB, sizeB, 1, lwdaMemcpyDefault, pStream);
                lwdaStreamSynchronize(pStream);

                {
                    auto err = lwdaGetLastError();
                    if(err != lwdaSuccess) {
                        fprintf(stderr, "Error: %s in %s at line %d\n", lwdaGetErrorString(err), __FILE__, __LINE__);
                        exit(-1);
                    }
                }
                std::vector<int64_t> extentA;
                for(auto mode : opts.modeA)
                    extentA.push_back(opts.extent[mode]);

                std::vector<int64_t> extentB;
                for(auto mode : opts.modeB)
                    extentB.push_back(opts.extent[mode]);

                EXPECT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);

                EXPECT_EQ( lwtensorInitTensorDescriptor( &handle, &descA, nmodeA, &extentA[0],
                            opts.strideA, opts.typeA, opts.opA, 1, 0), LWTENSOR_STATUS_SUCCESS);
                
                if (opts.vectorWidthA > 1)
                {
                    FAIL() << "Vectorization not supported.";
                }

                EXPECT_EQ( lwtensorInitTensorDescriptor( &handle, &descB, nmodeB, &extentB[0],
                            opts.strideB, opts.typeB, opts.opB, 1, 0), LWTENSOR_STATUS_SUCCESS);

                if (opts.vectorWidthA > 1)
                {
                    FAIL() << "Vectorization not supported.";
                }
            }
            void TearDown() {
                std::cout<<"[ TearDown ] "<<"destroy lwTensor."<<std::endl;
                std::cout<<"[ TearDown ] "<<"free allocated memory."<<std::endl;
                free(A);
                free(B);
                free(B_ref);

                lwdaFree(A_d);
                lwdaFree(B_d);

                lwdaStreamDestroy(pStream);

                if(opts.alpha)
                    free(opts.alpha);
            }

        public:
            size_t sizeA, sizeB;
            void *A_d,  *B_d;
            void *A, *B, *B_ref;
            lwtensorTensorDescriptor_t descA;
            lwtensorTensorDescriptor_t descB;
            TestOptions_t opts;
            lwtensorHandle_t handle;
            lwdaStream_t pStream;
    };

    class PermutationTestDefault : public PermutationTestBase, public ::testing::Test {
        protected:
            void OptsInitialization() override {
                std::cout<<"[Initialize] "<<"define the default mode and extent."<<std::endl;
                opts.modeA = {'c','b','a'};
                opts.modeB = {'d','c', 'b','a'};
                opts.extent.insert({'a', 40});
                opts.extent.insert({'b', 20});
                opts.extent.insert({'c', 30});
                opts.extent.insert({'d', 10});
            }
            void SetUp() override {
                PermutationTestBase::SetUp();
            }
            void TearDown() override {
                PermutationTestBase::TearDown();
            }
    };

    class lwtensorUnaryOpIntTest : public ::testing::TestWithParam<std::tuple<int, lwtensorOperator_t> >
    {
        public:
            lwtensorUnaryOpIntTest() {}
            ~lwtensorUnaryOpIntTest() {}

            void SetUp() {
                std::cout<<"[ Setup    ] "<<"get the parameters of input."<<std::endl;
                std::tie(x, opt) = GetParam();
            }
            void TearDown() {
                std::cout<<"[ TearDown ] "<<"..."<<std::endl;
            }

            int x = lwGet<int>(0);
            lwtensorOperator_t opt = LWTENSOR_OP_UNKNOWN;
    };

    class lwtensorBinaryOpIntTest : public ::testing::TestWithParam<
                                    std::tuple<int, int, lwtensorOperator_t> >
    {
        public:
            lwtensorBinaryOpIntTest() {}
            ~lwtensorBinaryOpIntTest() {}

            void SetUp() {
                std::cout<<"[ Setup    ] "<<"get the parameters of input."<<std::endl;
                std::tie(x, y, opt) = GetParam();
            }
            void TearDown() {
                std::cout<<"[ TearDown ] "<<"..."<<std::endl;
            }

            int x = lwGet<int>(0);
            int y = lwGet<int>(0);
            lwtensorOperator_t opt = LWTENSOR_OP_UNKNOWN;
    };

    void callingInfo(const char * apiName)
    {
        std::cout<<"[ Calling  ] "<<apiName<<std::endl;
    }

    template<class Tin, class Tout>
        __global__ void lwGet_kernel(Tin  *in, Tout * out)
        {
            out[0] = lwGet<Tout>(in[0]);
        }

    template <class Tin, class Tout>
        bool lwGetTest(Tin * hin, lwdaStream_t pstream)
        {
            Tin * din;
            Tout *dout;
            Tout *hout;
            lwdaCheck(lwdaMalloc((void **)&din, sizeof(Tin)));
            lwdaCheck(lwdaMalloc((void **)&dout, sizeof(Tout)));

            hout = (Tout *)malloc(sizeof(Tout));

            lwdaCheck(lwdaMemcpyAsync(din, hin, sizeof(Tin), lwdaMemcpyDefault, pstream));
            lwdaCheck(lwdaStreamSynchronize(pstream));
            lwGet_kernel<Tin, Tout><<<1,1,0,pstream>>>(din, dout);
            lwdaCheck(lwdaMemcpyAsync(hout, dout, sizeof(Tout), lwdaMemcpyDefault, pstream));
            lwdaCheck(lwdaStreamSynchronize(pstream));

            Tout h_ref = lwGet<Tout>(hin[0]);
            bool tmp = lwIsEqual(hout[0], h_ref);

            free(hout);
            lwdaCheck(lwdaFree(dout));
            lwdaCheck(lwdaFree(din));

            return tmp;
        }

    bool AlmostEqualRelative(float A, float B,
            float maxRelDiff = 0.01f)
    {
        // Callwlate the difference.
        float diff = fabs(A - B);
        A = fabs(A);
        B = fabs(B);
        // Find the largest
        float largest = (B > A) ? B : A;

        if (diff <= largest * maxRelDiff)
            return true;
        return false;
    }

    // for unary operations
    template<class Tin, class Tout>
        using funPtr = Tout (*)(Tin args);

    template<class Tin, class Tout>
        __global__ void deviceFun_kernel(Tin  *in, Tout * out, funPtr<Tin, Tout> f)
        {
            out[0] = (*f)(in[0]);
        }

    template <class Tin, class Tout>
        bool deviceCodeUnaryTest(Tin * hin, funPtr<Tin, Tout> fd, funPtr<Tin, Tout> fh, lwdaStream_t pstream)
        {
            Tin * din;
            Tout *dout;
            Tout *hout;
            lwdaCheck(lwdaMalloc((void **)&din, sizeof(Tin)));
            lwdaCheck(lwdaMalloc((void **)&dout, sizeof(Tout)));

            hout = (Tout *)malloc(sizeof(Tout));

            lwdaCheck(lwdaMemcpyAsync(din, hin, sizeof(Tin), lwdaMemcpyDefault, pstream));
            lwdaCheck(lwdaStreamSynchronize(pstream));
            deviceFun_kernel<Tin, Tout><<<1,1,0,pstream>>>(din, dout, fd);
            lwdaCheck(lwdaMemcpyAsync(hout, dout, sizeof(Tout), lwdaMemcpyDefault, pstream));
            lwdaCheck(lwdaStreamSynchronize(pstream));

            Tout h_ref = fh(hin[0]);
            std::cout << "Actual: " << lwGet<float>(hout[0]) << std::endl;
            std::cout << "Expected: " << lwGet<float>(h_ref) << std::endl;
            bool tmp = AlmostEqualRelative(lwGet<float>(hout[0]), lwGet<float>(h_ref));

            free(hout);
            lwdaCheck(lwdaFree(dout));
            lwdaCheck(lwdaFree(din));

            return tmp;
        }

    // for binary operations
    template<class Tin0, class Tin1, class Tout>
        using funPtr1 = Tout (*)(Tin0, Tin1);

    template<class Tin0, class Tin1, class Tout>
    __global__ void deviceFun_kernel1(Tin0 *in0, Tin1 *in1, Tout *out, funPtr1<Tin0, Tin1, Tout> f)
    {
        out[0] = (*f)(in0[0], in1[0]);
    }

    template<class Tin0, class Tin1, class Tout>
        bool deviceCodeBinaryTest(Tin0 *hin0, Tin1 *hin1, funPtr1<Tin0, Tin1, Tout> fd, funPtr1<Tin0, Tin1, Tout> fh, lwdaStream_t pstream)
        {
            Tin0 * din0;
            Tin1 * din1;
            Tout * dout;
            Tout * hout;
            lwdaCheck(lwdaMalloc((void **)&din0, sizeof(Tin0)));
            lwdaCheck(lwdaMalloc((void **)&din1, sizeof(Tin1)));
            lwdaCheck(lwdaMalloc((void **)&dout, sizeof(Tout)));

            hout = (Tout *)malloc(sizeof(Tout));
            lwdaCheck(lwdaMemcpyAsync(din0, hin0, sizeof(Tin0), lwdaMemcpyDefault));
            lwdaCheck(lwdaMemcpyAsync(din1, hin1, sizeof(Tin1), lwdaMemcpyDefault));
            lwdaCheck(lwdaStreamSynchronize(pstream));
            deviceFun_kernel1<Tin0, Tin1, Tout><<<1,1,0,pstream>>>(din0, din1, dout, fd);
            lwdaCheck(lwdaMemcpyAsync(hout, dout, sizeof(Tout), lwdaMemcpyDefault));
            lwdaCheck(lwdaStreamSynchronize(pstream));

            Tout h_ref = fh(hin0[0], hin1[0]);
            std::cout << "Actual: " << lwGet<float>(hout[0]) << std::endl;
            std::cout << "Expected: " << lwGet<float>(h_ref) << std::endl;
            bool tmp = AlmostEqualRelative(lwGet<float>(hout[0]), lwGet<float>(h_ref));

            free(hout);
            lwdaCheck(lwdaFree(dout));
            lwdaCheck(lwdaFree(din0));
            lwdaCheck(lwdaFree(din1));

            return tmp;
        }

    class PublicApiTestDefault : public ApiTestBase, public ::testing::Test {
        protected:
            PublicApiTestDefault() {}
            ~PublicApiTestDefault() {}
            void OptsInitialization() {
                std::cout<<"[Initialize] "<<"define the default mode and extent."<<std::endl;
                opts.modeC = {'m','u','n','v'};
                opts.modeA = {'m','h','k','n'};
                opts.modeB = {'u','k','v','h'};
                opts.extent.insert({'m', 48});
                opts.extent.insert({'n', 48});
                opts.extent.insert({'u', 48});
                opts.extent.insert({'v', 32});
                opts.extent.insert({'h', 32});
                opts.extent.insert({'k', 32});
                opts.disableVerification = true;
            }

            void SetUp() override {
                /*OptsInitialization();*/
                ApiTestBase::SetUp();

                // retrieve the memory alignment for each tensor
                lwtensorGetAlignmentRequirement(&handle,
                        A_d, &descA, &alignmentRequirementA);
                lwtensorGetAlignmentRequirement(&handle,
                        B_d, &descB, &alignmentRequirementB);
                lwtensorGetAlignmentRequirement(&handle,
                        C_d, &descC, &alignmentRequirementC);
            }

            void TearDown() override {
                ApiTestBase::TearDown();
            }

            uint32_t alignmentRequirementA;
            uint32_t alignmentRequirementB;
            uint32_t alignmentRequirementC;
    };

    class publicApiTestContraction : public ApiTestBase, public ::testing::Test {
        protected:
            publicApiTestContraction() {}
            ~publicApiTestContraction() {}
            void OptsInitialization() {
                std::cout<<"[Initialize] "<<"define the default mode and extent."<<std::endl;
                opts.modeC = {'m','n'};
                opts.modeA = {'m','k'};
                opts.modeB = {'n','k'};
                opts.extent.insert({'m', 48});
                opts.extent.insert({'n', 48});
                opts.extent.insert({'k', 32});
                opts.typeA = LWDA_R_32F;
                opts.typeB = LWDA_R_32F;
                opts.typeC = LWDA_R_32F;
                opts.disableVerification = true;
            }

            void SetUp() override {
                /*OptsInitialization();*/
                ApiTestBase::SetUp();

                // retrieve the memory alignment for each tensor
                lwtensorGetAlignmentRequirement(&handle,
                        A_d, &descA, &alignmentRequirementA);
                lwtensorGetAlignmentRequirement(&handle,
                        B_d, &descB, &alignmentRequirementB);
                lwtensorGetAlignmentRequirement(&handle,
                        C_d, &descC, &alignmentRequirementC);

                lwtensorAlgo_t algo = LWTENSOR_ALGO_GETT;

                lwtensorComputeType_t computeType = LWTENSOR_COMPUTE_32F;
                EXPECT_EQ( lwtensorInitContractionDescriptor( &handle, &descContr,
                                                             &descA, opts.modeA.data(), alignmentRequirementA,
                                                             &descB, opts.modeB.data(), alignmentRequirementB,
                                                             &descC, opts.modeC.data(), alignmentRequirementC,
                                                             &descC, opts.modeC.data(), alignmentRequirementC,
                                                             computeType), LWTENSOR_STATUS_SUCCESS);
                EXPECT_EQ( lwtensorInitContractionFind(&handle, &find, algo), LWTENSOR_STATUS_SUCCESS);
                EXPECT_EQ( lwtensorInitContractionPlan(&handle, &plan, &descContr, &find, /*worksize =*/ 0), LWTENSOR_STATUS_SUCCESS);
            }

            void TearDown() override {
                ApiTestBase::TearDown();
            }

            lwtensorContractionPlan_t plan;
            lwtensorContractionDescriptor_t descContr;
            lwtensorContractionFind_t find;
            uint32_t alignmentRequirementA;
            uint32_t alignmentRequirementB;
            uint32_t alignmentRequirementC;
    };
} //namespacass
