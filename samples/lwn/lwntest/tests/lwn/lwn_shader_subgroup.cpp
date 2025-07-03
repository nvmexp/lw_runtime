/*
 * Copyright (c) 2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//
// lwn_shader_subgroup.cpp
//
// Exercise some subgroup reduction/scan builtin functions
// Ported from vulkan/vk_subgroup.cpp
//

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include <limits>

using namespace lwn;

typedef enum { TYPE_FLOAT, TYPE_INT, TYPE_UINT } TypeEnum;

template<class funcT>
class LwnSubgroup {
public:
    LwnSubgroup<funcT>(uint32_t idU, float idF, TypeEnum t) :
        m_identityU(idU), m_identityF(idF), m_type(t) {}

    uint32_t m_identityU;
    float m_identityF;
    TypeEnum m_type;

    OGTEST_CppMethods();
};

template<class funcT>
void LwnSubgroup<funcT>::initGraphics(void)
{
    lwnDefaultInitGraphics();
}

template<class funcT>
void LwnSubgroup<funcT>::exitGraphics(void)
{
    lwnDefaultExitGraphics();
}

template<class funcT>
int LwnSubgroup<funcT>::isSupported()
{
    return g_lwnDeviceCaps.supportsShaderSubgroup;
}
template<class funcT>
lwString LwnSubgroup<funcT>::getDescription()
{
    lwStringBuf sb;

    sb << "Touch test various subgroup instructions. Each test folwses on one type/instruction. "
          "Each test runs a workgroup with 32 subgroups, where each subgroup "
          "forces some ilwocations inactive and tests four subgroup operations. "
          "The result is compared against a reference value and the window is "
          "cleared to green/red for pass/fail.";

    return sb.str();
}

// The arithmetic classes are meant to provide the actual operations to perform (Execute) as well as
// the string used in the shader (GetOpStr() - appended to subgroup functions during shader string
// construction).
template <typename T>
class ArithmeticClass {
public:

    // Performs the arithmetic operation
    virtual T Execute(T a, T b) = 0;

    // Returns the string of the op, used for constructing the shader strings.
    virtual const char * GetOpStr() = 0;
};

template <typename T>
class Add : public ArithmeticClass<T> {
public:
    virtual T Execute(T a, T b) { return a + b; }
    virtual const char * GetOpStr() { return "Add"; }
};

template <typename T>
class Mul : public ArithmeticClass<T> {
public:
    virtual T Execute(T a, T b) { return a * b; }
    virtual const char * GetOpStr() { return "Mul"; }
};

template <typename T>
class Min : public ArithmeticClass<T> {
public:
    virtual T Execute(T a, T b) { return a < b ? a : b; }
    virtual const char * GetOpStr() { return "Min"; }
};

template <typename T>
class Max : public ArithmeticClass<T> {
public:
    virtual T Execute(T a, T b) { return a > b ? a : b; }
    virtual const char * GetOpStr() { return "Max"; }
};

template <typename T>
class And : public ArithmeticClass<T> {
public:
    virtual T Execute(T a, T b) { return a & b; }
    virtual const char * GetOpStr() { return "And"; }
};

template <typename T>
class Or : public ArithmeticClass<T> {
public:
    virtual T Execute(T a, T b) { return a | b; }
    virtual const char * GetOpStr() { return "Or"; }
};

template <typename T>
class Xor : public ArithmeticClass<T> {
public:
    virtual T Execute(T a, T b) { return a ^ b; }
    virtual const char * GetOpStr() { return "Xor"; }
};

// Combine all 32 input elements, and write that value to all output elements
template <typename T, class funcT>
void reduce(T *inptr, uint32_t mask, T *outptr) {
    bool first = true;
    funcT Func;
    T ret = 0;

    for (uint32_t i = 0; i < 32; ++i) {
        if (mask & (1u << i)) {
            if (first) {
                ret = inptr[i];
                first = false;
            } else {
                ret = Func.Execute(ret, inptr[i]);
            }
        }
    }
    for (uint32_t i = 0; i < 32; ++i) {
        if (mask & (1u << i)) {
            outptr[i] = ret;
        }
    }
}

// Combine all <clusterSize> input elements, and write that value to <clusterSize> output elements
template <typename T, class funcT>
void cluster2(T *inptr, uint32_t mask, uint32_t clusterSize, T *outptr) {
    bool first = true;
    funcT Func;
    T ret = 0;

    for (uint32_t i = 0; i < clusterSize; ++i) {
        if (mask & (1u << i)) {
            if (first) {
                ret = inptr[i];
                first = false;
            } else {
                ret = Func.Execute(ret, inptr[i]);
            }
        }
    }
    for (uint32_t i = 0; i < clusterSize; ++i) {
        if (mask & (1u << i)) {
            outptr[i] = ret;
        }
    }
}

// Loop over the 32/clusterSize clusters
template <typename T, class funcT>
void cluster(T *inptr, uint32_t mask, uint32_t clusterSize, T *outptr) {
    for (uint32_t i = 0; i < 32; i += clusterSize) {
        cluster2<T, funcT>(&inptr[i], mask >> i, clusterSize, &outptr[i]);
    }
}

// First active element gets the identity, the rest are the previous element's result plus its value.
// This corresponds to the subgroupExclusiveScan* opterations where the instance i's value is the <op>
// combination of all the instances before it (i.e. having lesser gl_SubgroupIlwocationID) and EXCLUDING
// instance itself.
template <typename T, class funcT>
void exclusivescan(T *inptr, uint32_t mask, T *outptr,T identity) {
    int prev = -1;
    funcT Func;

    for (uint32_t i = 0; i < 32; ++i) {
        if (mask & (1u << i)) {
            if (prev == -1) {
                outptr[i] = identity;
            } else {
                outptr[i] = Func.Execute(outptr[prev], inptr[prev]);
            }
            prev = i;
        }
    }
}

// First active element gets its input value, the rest are the previous element's result plus their own value
// This corresponds to the subgroupInclusiveScan* opterations where the instance i's value is the <op>
// combination of all the instances before it (i.e. having lesser gl_SubgroupIlwocationID) and INCLUDING the
// instance itself.
template <typename T, class funcT>
void inclusivescan(T *inptr, uint32_t mask, T *outptr) {
    int prev = -1;
    funcT Func;

    for (uint32_t i = 0; i < 32; ++i) {
        if (mask & (1u << i)) {
            if (prev == -1) {
                outptr[i] = inptr[i];
            } else {
                outptr[i] = Func.Execute(outptr[prev], inptr[i]);
            }
            prev = i;
        }
    }
}

#define NUM_BUFFERS 7

template<class funcT>
void LwnSubgroup<funcT>::doGraphics()
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    funcT OpFunc;

    bool bOK = true;

    // Create the SSBOS buffers.
    size_t bufferSize = 32 * 32 * sizeof(uint32_t);

    MemoryPoolAllocator ssboAllocator(
        device, NULL, NUM_BUFFERS*bufferSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *buffers[NUM_BUFFERS];
    uint32_t *bufferPtrs[NUM_BUFFERS];
    BufferAddress bufferAddresses[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        buffers[i] = ssboAllocator.allocBuffer(
            &bb, BUFFER_ALIGN_SHADER_STORAGE_BIT, bufferSize);
        bufferPtrs[i] = (uint32_t *)(buffers[i]->Map());
        bufferAddresses[i] = buffers[i]->GetAddress();

        memset(bufferPtrs[i], 0, bufferSize);
    }

    uint32_t * masks = bufferPtrs[0];
    uint32_t * dataIn = bufferPtrs[1];

    // Fill the input buffer with random signed integer data
    for (int i = 0; i < 32*32; ++i) {
        if (m_type == TYPE_FLOAT) {
            ((float *)dataIn)[i] = lwFloatRand(-10, 10);
        } else {
            dataIn[i] = lwIntRand(-100, 100);
        }
    }
    // Fill the mask buffer with semi-random data. The first 11 cases are intended to exercise fully or
    // "incomplete" warps. The random tests exercise divergent flow control.
    for (int i = 0; i < 32; ++i) {
        switch (i) {
        case 0:
        case 1:
        case 2:
            masks[i] = ~0;
            break;
        case 3: masks[i] = 0xF;             break;
        case 4: masks[i] = 0xFF;            break;
        case 5: masks[i] = 0x7F;            break;
        case 6: masks[i] = 0x3FFF;          break;
        case 7: masks[i] = 0x7FFF;          break;
        case 8: masks[i] = 0xFFFF;          break;
        case 9: masks[i] = 0x1FFFF;         break;
        case 10: masks[i] = 0x3FFFF;        break;
        default: masks[i] = lwBitRand(32);  break;
        }
    }

    const char *typestr = m_type == TYPE_FLOAT ? "float" : (m_type == TYPE_INT ? "int" : "uint");

    // Shaders
    lwShader css;
    css = ComputeShader(450);

    css.addExtension(lwShaderExtension::KHR_shader_subgroup_basic);
    css.addExtension(lwShaderExtension::KHR_shader_subgroup_vote);
    css.addExtension(lwShaderExtension::KHR_shader_subgroup_arithmetic);
    css.addExtension(lwShaderExtension::KHR_shader_subgroup_ballot);
    css.addExtension(lwShaderExtension::KHR_shader_subgroup_shuffle);
    css.addExtension(lwShaderExtension::KHR_shader_subgroup_shuffle_relative);
    css.addExtension(lwShaderExtension::KHR_shader_subgroup_clustered);
    css.addExtension(lwShaderExtension::KHR_shader_subgroup_quad);
    css.addExtension(lwShaderExtension::LW_shader_subgroup_partitioned);

    css << 
        // Each row is a subgroup (on LW HW)
        "layout (local_size_x=32,local_size_y=32,local_size_z=1) in;\n"
        "layout(binding = 0) buffer Block { uint masks[32*32]; } maskBuf;\n"
        // b[0] is the input, b[1-4] are output buffers
        "layout(binding = 1) buffer Block2 { " << typestr << " data[32*32]; } b[" << NUM_BUFFERS-1 << "];\n"

        "void main()\n"
        "{\n"
        // Use 32 contiguous elements for each subgroup
        "   uint idx = gl_SubgroupSize*gl_SubgroupID + gl_SubgroupIlwocationID;\n"
        "   " << typestr << " x = b[0].data[idx];\n"
        // Mask of threads to keep active for this subgroup
        "   uint mask = maskBuf.masks[gl_SubgroupID];\n"

        "   " << typestr << " result[5];\n"
        "   result[0] = result[1] = result[2] = result[3] = result[4] = 0;\n"
        "   uint anyEqualMask = 0;\n"
        // Force threads not in <mask> inactive, then test reduce, inclusive/exclusive scan, and clustered reduction
        "   if ((mask & (1 << gl_SubgroupIlwocationID)) != 0) {\n"
        "        result[0] = subgroup" << OpFunc.GetOpStr() << "(x);\n"
        "        result[1] = subgroupInclusive" << OpFunc.GetOpStr() << "(x);\n"
        "        result[2] = subgroupExclusive" << OpFunc.GetOpStr() << "(x);\n"
        "        result[3] = subgroupClustered" << OpFunc.GetOpStr() << "(x, 4);\n"
        "   }\n";

        // LW_shader_subgroup_partitioned 
    css << 
        "#ifdef GL_LW_shader_subgroup_partitioned\n"
        "   if ((mask & (1 << gl_SubgroupIlwocationID)) != 0) {\n"
        "        anyEqualMask = subgroupPartitionLW(x).x;\n"
        "   }\n"
        "   uvec4 ballot = ((mask & (1 << gl_SubgroupIlwocationID)) != 0) ? uvec4(mask) : uvec4(~mask);\n"
        "   result[4] = subgroupPartitionedInclusive" << OpFunc.GetOpStr() << "LW(x, ballot);\n"
        "#endif\n";

    css <<
        "   b[1].data[idx] = result[0];\n"
        "   b[2].data[idx] = result[1];\n"
        "   b[3].data[idx] = result[2];\n"
        "   b[4].data[idx] = result[3];\n"
        "   b[5].data[idx] = result[4];\n"
        // Test that several built-in functions/variables are working as expected. Output an "error code" if not.
        "   if (gl_SubgroupSize != 32) b[2].data[idx] = 0;\n"
        "   if (gl_NumSubgroups != (gl_WorkGroupSize.x*gl_WorkGroupSize.y*gl_WorkGroupSize.z + 31) / 32) b[2].data[idx] = 2;\n"
        "   if (gl_SubgroupID != (gl_LocalIlwocationIndex / 32)) b[2].data[idx] = 3;\n"
        // Test that the way we compute subgroupID is actually a unique value across the subgroup
        "   if (!subgroupAllEqual(gl_SubgroupID)) b[2].data[idx] = 5;\n";

        // LW_shader_subgroup_partitioned 
    css << 
        "#ifdef GL_LW_shader_subgroup_partitioned\n"
        "   if ((mask & (1 << gl_SubgroupIlwocationID)) != 0) {\n"
        "       uint m = 0; for (int i = 0; i < gl_SubgroupSize; ++i) if ((mask & (1 << i)) != 0 && subgroupShuffle(x, i) == x) m |= (1<<i);\n"
        "       if ((anyEqualMask & (1 << gl_SubgroupIlwocationID)) == 0) b[2].data[idx] = 7;\n"
        "       if (anyEqualMask != m) b[2].data[idx] = 8;\n"
        "   }\n"
        "#endif\n";

    css << "}\n";

    Program *pgm = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(pgm, css)) {
        LWNFailTest();
        return;
    }

    queueCB.BindProgram(pgm, ShaderStageBits::COMPUTE);

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        queueCB.BindStorageBuffer(ShaderStage::COMPUTE, i, bufferAddresses[i], bufferSize);
    }

    queueCB.DispatchCompute(1, 1, 1);
    queueCB.submit();
    queue->Finish();

    // Check results in buffers[1-4].
    for (int i = 0; i < 32; ++i) {
        // Generate reference values on the CPU
        switch (m_type) {
        case TYPE_FLOAT:
            {
                float temp[NUM_BUFFERS-2][32];
                memset(temp, 0, sizeof(temp));
                reduce<float, funcT>((float *)&dataIn[i*32], masks[i], &temp[0][0]);
                inclusivescan<float, funcT>((float *)&dataIn[i*32], masks[i], &temp[1][0]);
                exclusivescan<float, funcT>((float *)&dataIn[i*32], masks[i], &temp[2][0], m_identityF);
                cluster<float, funcT>((float *)&dataIn[i*32], masks[i], 4, &temp[3][0]);
                inclusivescan<float, funcT>((float *)&dataIn[i*32], masks[i], &temp[4][0]);
                inclusivescan<float, funcT>((float *)&dataIn[i*32], ~masks[i], &temp[4][0]);

                // Compare buffers 1-4 against the reference values. If anything is wrong, clear to red.
                for (int k = 0; k < NUM_BUFFERS-2; ++k) {
                    float *dataOut = (float *)(bufferPtrs[2+k]);
                    for (int j = 0; j < 32; ++j) {
                        if (fabs((dataOut[i*32+j] - temp[k][j]) / temp[k][j]) > 0.0001f) {
                            bOK = false;
                        }
                    }
                }
            }
            break;
        case TYPE_UINT:
            {
                uint32_t temp[NUM_BUFFERS-2][32];
                memset(temp, 0, sizeof(temp));
                reduce<uint32_t, funcT>(&dataIn[i*32], masks[i], &temp[0][0]);
                inclusivescan<uint32_t, funcT>(&dataIn[i*32], masks[i], &temp[1][0]);
                exclusivescan<uint32_t, funcT>(&dataIn[i*32], masks[i], &temp[2][0], m_identityU);
                cluster<uint32_t, funcT>(&dataIn[i*32], masks[i], 4, &temp[3][0]);
                inclusivescan<uint32_t, funcT>(&dataIn[i*32], masks[i], &temp[4][0]);
                inclusivescan<uint32_t, funcT>(&dataIn[i*32], ~masks[i], &temp[4][0]);

                // Compare buffers 1-4 against the reference values. If anything is wrong, clear to red.
                for (int k = 0; k < NUM_BUFFERS-2; ++k) {
                    uint32_t *dataOut = bufferPtrs[2+k];
                    for (int j = 0; j < 32; ++j) {
                        if (dataOut[i*32+j] != temp[k][j]) {
                            printf("dataOut[%d*32+%d]: %d\n", i, j, dataOut[i*32 + j]);
                            bOK = false;
                        }
                    }
                }
            }
            break;
        case TYPE_INT:
            {
                int32_t temp[NUM_BUFFERS-2][32];
                memset(temp, 0, sizeof(temp));
                reduce<int32_t, funcT>((int32_t *)&dataIn[i*32], masks[i], &temp[0][0]);
                inclusivescan<int32_t, funcT>((int32_t *)&dataIn[i*32], masks[i], &temp[1][0]);
                exclusivescan<int32_t, funcT>((int32_t *)&dataIn[i*32], masks[i], &temp[2][0], m_identityU);
                cluster<int32_t, funcT>((int32_t *)&dataIn[i*32], masks[i], 4, &temp[3][0]);
                inclusivescan<int32_t, funcT>((int32_t *)&dataIn[i*32], masks[i], &temp[4][0]);
                inclusivescan<int32_t, funcT>((int32_t *)&dataIn[i*32], ~masks[i], &temp[4][0]);

                // Compare buffers 1-4 against the reference values. If anything is wrong, clear to red.
                for (int k = 0; k < NUM_BUFFERS-2; ++k) {
                    int32_t *dataOut = (int32_t *)bufferPtrs[2+k];
                    for (int j = 0; j < 32; ++j) {
                        if (dataOut[i*32+j] != temp[k][j]) {
                            bOK = false;
                        }
                    }
                }
            }
            break;
        }
    }

    // CLEAR to red/green based on bOK.
    if (bOK) {
        queueCB.ClearColor(0, 0.0f, 1.0f, 0.0f, 1.0f);
    } else {
        queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f, 1.0f);
    }

    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LwnSubgroup<Add<uint32_t> >, lwn_shader_subgroup_uint_add,  (0, 0, TYPE_UINT));
OGTEST_CppTest(LwnSubgroup<Mul<uint32_t> >, lwn_shader_subgroup_uint_mul,  (1, 0, TYPE_UINT));
OGTEST_CppTest(LwnSubgroup<Min<uint32_t> >, lwn_shader_subgroup_uint_min,  (~0, 0, TYPE_UINT));
OGTEST_CppTest(LwnSubgroup<Max<uint32_t> >, lwn_shader_subgroup_uint_max,  (0, 0, TYPE_UINT));
OGTEST_CppTest(LwnSubgroup<And<uint32_t> >, lwn_shader_subgroup_uint_and,  (~0, 0, TYPE_UINT));
OGTEST_CppTest(LwnSubgroup<Or<uint32_t> >,  lwn_shader_subgroup_uint_or,   (0, 0, TYPE_UINT));
OGTEST_CppTest(LwnSubgroup<Xor<uint32_t> >, lwn_shader_subgroup_uint_xor,  (0, 0, TYPE_UINT));

OGTEST_CppTest(LwnSubgroup<Add<int32_t> >,  lwn_shader_subgroup_int_add,   (0, 0, TYPE_INT));
OGTEST_CppTest(LwnSubgroup<Mul<int32_t> >,  lwn_shader_subgroup_int_mul,   (1, 0, TYPE_INT));
OGTEST_CppTest(LwnSubgroup<Min<int32_t> >,  lwn_shader_subgroup_int_min,   (0x7FFFFFFF, 0, TYPE_INT));
OGTEST_CppTest(LwnSubgroup<Max<int32_t> >,  lwn_shader_subgroup_int_max,   (0x80000000, 0, TYPE_INT));
OGTEST_CppTest(LwnSubgroup<And<int32_t> >,  lwn_shader_subgroup_int_and,   (~0, 0, TYPE_INT));
OGTEST_CppTest(LwnSubgroup<Or<int32_t> >,   lwn_shader_subgroup_int_or,    (0, 0, TYPE_INT));
OGTEST_CppTest(LwnSubgroup<Xor<int32_t> >,  lwn_shader_subgroup_int_xor,   (0, 0, TYPE_INT));

OGTEST_CppTest(LwnSubgroup<Add<float> >,    lwn_shader_subgroup_float_add, (0, 0, TYPE_FLOAT));
OGTEST_CppTest(LwnSubgroup<Mul<float> >,    lwn_shader_subgroup_float_mul, (0, 1.0f, TYPE_FLOAT));
OGTEST_CppTest(LwnSubgroup<Min<float> >,    lwn_shader_subgroup_float_min, (0, std::numeric_limits<float>::infinity(), TYPE_FLOAT));
OGTEST_CppTest(LwnSubgroup<Max<float> >,    lwn_shader_subgroup_float_max, (0, -std::numeric_limits<float>::infinity(), TYPE_FLOAT));
