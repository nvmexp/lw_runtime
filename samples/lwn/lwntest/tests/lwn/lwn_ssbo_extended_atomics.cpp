/*
 * Copyright (c) 2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;

class LWNSSBOAtomicTest
{
public:
private:
    static const int ctaWidth = 2;
    static const int ctaHeight = 1;
    enum atomicOp
    {
        Add,
        Or,
        Xor,
        Min,
        Max,  
        Exchange,
        CompSwap,
        OpEnd,
    };

    enum atomicType
    {
        TypeUint32,
        TypeInt32,
        TypeUint64,
        TypeInt64,
        TypeFloat32,
        TypeFloat64,
        TypeFloat16x2,
        TypeFloat16x4,
        TypeEnd
    };

    // Host side layout for SSBO
    struct SSBOLayout {
      int64_t i64;
      uint64_t u64;
      double f64;
      int64_t i64arr[2];
      uint64_t u64arr[2];
      double f64arr[2];
      int i32;
      uint32_t u32;
      float f32;
      int i32arr[2];
      uint32_t u32arr[2];
      float f32arr[2];
      uint32_t f16v2;
      uint64_t f16v4;
      uint32_t f16v2arr[2];
      uint64_t f16v4arr[4];
    };

    //Buffer declarations
    const char *interfaceBlock =
      "struct SSBO {\n"
      "  int64_t i64; \n"
      "  uint64_t u64; \n"
      "  double f64; \n"
      "  int64_t i64arr[2]; \n"
      "  uint64_t u64arr[2]; \n"
      "  double f64arr[2]; \n"
      "  int i32; \n"
      "  uint u32; \n"
      "  float f32; \n"
      "  int i32arr[2]; \n"
      "  uint u32arr[2]; \n"
      "  float f32arr[2]; \n"
      "  f16vec2 f16v2; \n"
      "  f16vec4 f16v4; \n"
      "  f16vec2 f16v2arr[2]; \n"
      "  f16vec4 f16v4arr[2]; \n"
      "};\n"
      "layout(std140, binding = 0) uniform UBO {\n"
      "  SSBO *ssbo;\n"
      "};\n";

    bool isSupportedAtomicForType(Device *device, int op, int type) const;
    Program* CreateTestProgram(Device *device, int op, int type) const;
    bool VerifyResultData(SSBOLayout *mem, int op, int type) const;
    void InitializeTestData(SSBOLayout *mem, int op, int type) const;
public:
    LWNTEST_CppMethods();
};

lwString LWNSSBOAtomicTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Tests atomic functions under LW_extended_pointer_atomics"
          "Tests all supported combinations based on opcode and type";
    return sb.str();
}

int LWNSSBOAtomicTest::isSupported() const
{
    //TBD:
    return lwogCheckLWNAPIVersion(21, 4);
}

bool LWNSSBOAtomicTest::isSupportedAtomicForType(Device *device, int op, int type) const
{
    bool rv = true;
    // HACK : LW_shader_atomic_fp16_vector is supported only on GM20X and beyond
    //        LW_shader_atomic_int64 is supported only on GK11x and beyond
    // We do not have a device capability bit specific for GK11x so just use GM20X
    // at the cost of missing coverage on few Kepler chips
    int gm20x = g_lwnDeviceCaps.supportsMaxwell2Features;
    if (type == TypeFloat64) {
        //Unsupported for LWN : gm20y
        rv = false;
    } else if (type == TypeFloat32) {
        if (op != Add && op != Exchange)
            rv = false;
    } else if (type == TypeFloat16x2 || type == TypeFloat16x4) {
        if (op == CompSwap || op == Or || op == Xor || !gm20x)
            rv = false;
    } else if (type == TypeUint64 || type == TypeInt64) {
            rv = !!gm20x;
    }
    return rv;

}

Program* LWNSSBOAtomicTest::CreateTestProgram(Device *device, int op, int type) const
{

    Program *program = device->CreateProgram();
    ComputeShader csTest(450);
    csTest.setCSGroupSize(ctaWidth, ctaHeight);
    csTest.addExtension(lwShaderExtension::LW_gpu_shader5);
    csTest.addExtension(lwShaderExtension::LW_shader_atomic_int64);
    csTest.addExtension(lwShaderExtension::LW_shader_atomic_fp16_vector);
    csTest.addExtension(lwShaderExtension::LW_shader_atomic_float);
    csTest.addExtension(lwShaderExtension::LW_extended_pointer_atomics);
    
    csTest << interfaceBlock;
    csTest <<
        "void main() {\n";

    switch (type) {
    case TypeUint32:
        csTest << "#define MTYPE u32 \n";
        csTest << "#define MTYPEARR u32arr\n";
        csTest << "#define DTYPE uint\n";
        break;

    case TypeInt32:
        csTest << "#define MTYPE i32 \n";
        csTest << "#define MTYPEARR i32arr\n";
        csTest << "#define DTYPE int\n";
        break;

    case TypeUint64:
        csTest << "#define MTYPE u64 \n";
        csTest << "#define MTYPEARR u64arr\n";
        csTest << "#define DTYPE uint64_t\n";
        break;

    case TypeInt64:
        csTest << "#define MTYPE i64 \n";
        csTest << "#define MTYPEARR i64arr\n";
        csTest << "#define DTYPE int64_t\n";
        break;

    case TypeFloat32:
        csTest << "#define MTYPE f32 \n";
        csTest << "#define MTYPEARR f32arr\n";
        csTest << "#define DTYPE float\n";
        break;

    case TypeFloat64:
        csTest << "#define MTYPE f64 \n";
        csTest << "#define MTYPEARR f64arr\n";
        csTest << "#define DTYPE double\n";
        break;

    case TypeFloat16x2:
        csTest << "#define MTYPE f16v2 \n";
        csTest << "#define MTYPEARR f16v2arr\n";
        csTest << "#define DTYPE f16vec2\n";
        break;

    case TypeFloat16x4:
        csTest << "#define MTYPE f16v4 \n";
        csTest << "#define MTYPEARR f16v4arr\n";
        csTest << "#define DTYPE f16vec4\n";
        break;

    }
    
    switch (op) {
        case Add:
        csTest << "atomicAdd(ssbo->MTYPE, DTYPE(1)); \n";
        csTest << "atomicAdd(ssbo->MTYPEARR[gl_LocalIlwocationID.x], DTYPE(1)); \n";
        break;

        case Or:
        csTest << "atomicOr(ssbo->MTYPE, DTYPE(1) << gl_LocalIlwocationID.x); \n";
        csTest << "atomicOr(ssbo->MTYPEARR[gl_LocalIlwocationID.x], DTYPE(1)); \n";
        break;

        case Xor:
        csTest << "atomicXor(ssbo->MTYPE, DTYPE(1)); \n";
        csTest << "atomicXor(ssbo->MTYPEARR[gl_LocalIlwocationID.x], DTYPE(1));\n";
        break;

        case Min:
        csTest << "atomicMin(ssbo->MTYPE, DTYPE(gl_LocalIlwocationID.x)); \n;";
        csTest << "atomicMin(ssbo->MTYPEARR[gl_LocalIlwocationID.x], DTYPE(gl_LocalIlwocationID.x)); \n";
        csTest << "atomicMin(ssbo->MTYPEARR[1], DTYPE(0)); \n";
        break;

        case Max:
        csTest << "atomicMax(ssbo->MTYPE, DTYPE(gl_LocalIlwocationID.x + 1)); \n";
        csTest << "atomicMax(ssbo->MTYPEARR[gl_LocalIlwocationID.x], DTYPE(1)); \n";
        break;

        case Exchange:
        csTest << "atomicExchange(ssbo->MTYPE, DTYPE(gl_LocalIlwocationID.x));\n";
        csTest << "atomicExchange(ssbo->MTYPEARR[gl_LocalIlwocationID.x], DTYPE(1)); \n";
        break;

        case CompSwap:
        csTest << "atomicCompSwap(ssbo->MTYPE, DTYPE(0), DTYPE(gl_LocalIlwocationID.x)); \n";
        csTest << "atomicCompSwap(ssbo->MTYPEARR[gl_LocalIlwocationID.x], DTYPE(0), DTYPE(1));\n";
        break;
    }

    csTest << 
    "  }\n";

    if (!g_glslcHelper->CompileAndSetShaders(program, csTest)) {
        LWNFailTest();
        return nullptr;
    }
    return program;
}
void LWNSSBOAtomicTest::InitializeTestData(SSBOLayout *mem, int op, int type) const {
    if (op == Min) {
        switch (type) {
        case TypeUint32:
            mem->u32 = mem->u32arr[0] = mem->u32arr[1] = 2;
            break;
        case TypeInt32:
            mem->i32 = mem->i32arr[0] = mem->i32arr[1] = 2;
            break;
        case TypeUint64:
            mem->u64 = mem->u64arr[0] = mem->u64arr[1] = 2;
            break;
        case TypeInt64:
            mem->i64 = mem->i64arr[0] = mem->i64arr[1] = 2;
            break;
        case TypeFloat16x2:
            mem->f16v2 = mem->f16v2arr[0] = mem->f16v2arr[1] = 0x40004000;
            break;
        case TypeFloat16x4:
            mem->f16v4 = mem->f16v4arr[0] = mem->f16v4arr[1] =
                0x4000400040004000;
            break;
        default:
            memset(mem, 0, sizeof(SSBOLayout));
            break;
            
        }

    } else {
        memset(mem, 0, sizeof(SSBOLayout));
    }
}
bool LWNSSBOAtomicTest::VerifyResultData(SSBOLayout *mem, int op, int type) const
{
    bool rv = false;
    switch (type) {
    case TypeUint32:
    {
        switch (op) {
        case Add:
        case Max:
            rv = (mem->u32 == 2 && mem->u32arr[0] == 1 &&
                mem->u32arr[1] == 1);
            break;
        case Or:
            rv = (mem->u32 == 3 && mem->u32arr[0] == 1 &&
                mem->u32arr[1] == 1);
            break;
        case Xor:
            rv = (mem->u32 == 0 && mem->u32arr[0] == 1 &&
                mem->u32arr[1] == 1);
            break;
        case Min:
            rv = (mem->u32 == 0 && mem->u32arr[0] == 0 &&
                mem->u32arr[1] == 0);
            break;
        case CompSwap:
            rv = (mem->u32 == 0 || mem->u32 == 1) &&
                (mem->u32arr[0] == 1 && mem->u32arr[1] == 1);
            break;
        case Exchange:
            rv = (mem->u32 == 1 && mem->u32arr[0] == 1 &&
                mem->u32arr[1] == 1);
            break;
        }
    }
    break;
    case TypeUint64:
    {
        switch (op) {
        case Add:
        case Max:
            rv = (mem->u64 == 2 && mem->u64arr[0] == 1 &&
                mem->u64arr[1] == 1);
            break;
        case Or:
            rv = (mem->u64 == 3 && mem->u64arr[0] == 1 &&
                mem->u64arr[1] == 1);
            break;
        case Xor:
            rv = (mem->u64 == 0 && mem->u64arr[0] == 1 &&
                mem->u64arr[1] == 1);
            break;
        case Min:
            rv = (mem->u64 == 0 && mem->u64arr[0] == 0 &&
                mem->u64arr[1] == 0);
            break;
        case CompSwap:
            rv = (mem->u64 == 0 || mem->u64 == 1) &&
                (mem->u64arr[0] == 1 && mem->u64arr[1] == 1);
            break;
        case Exchange:
            rv = (mem->u64 == 1 && mem->u64arr[0] == 1 &&
                mem->u64arr[1] == 1);
            break;
        }
    }
    break;
    case TypeInt32:
    {
        switch (op) {
        case Add:
        case Max:
            rv = (mem->i32 == 2 && mem->i32arr[0] == 1 &&
                mem->i32arr[1] == 1);
            break;
        case Or:
            rv = (mem->i32 == 3 && mem->i32arr[0] == 1 &&
                mem->i32arr[1] == 1);
            break;
        case Xor:
            rv = (mem->i32 == 0 && mem->i32arr[0] == 1 &&
                mem->i32arr[1] == 1);
            break;
        case Min:
            rv = (mem->i32 == 0 && mem->i32arr[0] == 0 &&
                mem->i32arr[1] == 0);
            break;
        case CompSwap:
            rv = (mem->i32 == 0 || mem->i32 == 1) &&
                (mem->i32arr[0] == 1 && mem->i32arr[1] == 1);
            break;
        case Exchange:
            rv = (mem->i32 == 1 && mem->i32arr[0] == 1 &&
                mem->i32arr[1] == 1);
            break;
        }
    }
    break;
    case TypeInt64:
    {
        switch (op) {
        case Add:
        case Max:
            rv = (mem->i64 == 2 && mem->i64arr[0] == 1 &&
                mem->i64arr[1] == 1);
            break;
        case Or:
            rv = (mem->i64 == 3 && mem->i64arr[0] == 1 &&
                mem->i64arr[1] == 1);
            break;
        case Xor:
            rv = (mem->i64 == 0 && mem->i64arr[0] == 1 &&
                mem->i64arr[1] == 1);
            break;
        case Min:
            rv = (mem->i64 == 0 && mem->i64arr[0] == 0 &&
                mem->i64arr[1] == 0);
            break;
        case CompSwap:
            rv = (mem->i64 == 0 || mem->i64 == 1) &&
                (mem->i64arr[0] == 1 && mem->i64arr[1] == 1);
            break;
        case Exchange:
            rv = (mem->i64 == 1 && mem->i64arr[0] == 1 &&
                mem->i64arr[1] == 1);
            break;
        }
    }
    break;
    case TypeFloat16x2:
    {
        switch (op) {
        case Add:
        case Max:
            rv = (mem->f16v2 == 0x40004000 && mem->f16v2arr[0] == 0x3c003c00 &&
                mem->f16v2arr[1] == 0x3c003c00);
            break;
        case Min:
            rv = (mem->f16v2 == 0 && mem->f16v2arr[0] == 0 &&
                mem->f16v2arr[1] == 0);
            break;
        case Exchange:
            rv = (mem->f16v2 == 0x3c003c00 && mem->f16v2arr[0] == 0x3c003c00 &&
                mem->f16v2arr[1] == 0x3c003c00);
            break;
        }

    }
    break;
    case TypeFloat16x4:
    {
        switch (op) {
        case Add:
        case Max:
            rv = (mem->f16v4 == 0x4000400040004000 && mem->f16v4arr[1] == 0x3c003c003c003c00 &&
                mem->f16v4arr[1] == 0x3c003c003c003c00);
            break;
        case Min:
            rv = (mem->f16v2 == 0 && mem->f16v4arr[0] == 0 &&
                mem->f16v4arr[1] == 0);
            break;
        case Exchange:
            rv = (mem->f16v4 == 0x3c003c003c003c00 && mem->f16v4arr[0] == 0x3c003c003c003c00 &&
                mem->f16v4arr[1] == 0x3c003c003c003c00);
            break;
        }

    }
    break;
    case TypeFloat32:
    {
        switch (op) {
        case Add:
            rv = (mem->f32 == 2.0f && mem->f32arr[0] == 1.0f &&
                mem->f32arr[1] == 1.0f);
            break;
        case Exchange:
            rv = (mem->f32 == 1.0f && mem->f32arr[0] == 1.0f &&
                mem->f32arr[1] == 1.0f);
            break;
        }

    }
    break;
    case TypeFloat64:
    {
        switch (op) {
        case Add:
            rv = (mem->f64 == 2.0 && mem->f64arr[0] == 1.0 &&
                mem->f64arr[1] == 1.0);
            break;
        case Exchange:
            rv = (mem->f64 == 1.0 && mem->f64arr[0] == 1.0 &&
                mem->f64arr[1] == 1.0);
            break;
        }

    }
    break;
    }
    return rv;
}
void LWNSSBOAtomicTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const int numCols = OpEnd;
    const int numRows = TypeEnd;
    const int cellWidth = lwrrentWindowWidth / numCols;
    const int cellHeight = lwrrentWindowHeight / numRows;

    
    int ssboSize = sizeof(SSBOLayout);
    //Pad to multiple of 32
    ssboSize = (ssboSize + 0x1F) & ~(0x1F);


    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();

    MemoryPoolAllocator ssboAllocator(device, NULL, ssboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ssbo = ssboAllocator.allocBuffer(&bb, BUFFER_ALIGN_SHADER_STORAGE_BIT, ssboSize);
    BufferAddress ssboAddr = ssbo->GetAddress();

    LWNint uboAlignment;
    device->GetInteger(DeviceInfo::UNIFORM_BUFFER_ALIGNMENT, &uboAlignment);
    int uboSize = uboAlignment;
    if (uboSize < LWNint(sizeof(dt::ivec4))) {
        uboSize = sizeof(dt::ivec4);
    }

    //Create UBO and populate with address of SSBO
    MemoryPoolAllocator uboAllocator(device, NULL, uboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *ubo = uboAllocator.allocBuffer(&bb, BUFFER_ALIGN_UNIFORM_BIT, uboSize);
    BufferAddress uboAddr = ubo->GetAddress();
    char *uboMem = (char *) ubo->Map();

    struct UBOLayout {
      BufferAddress ssbo;
    };
    
    
    UBOLayout *data = (UBOLayout *)uboMem;
    data->ssbo = ssboAddr;


    ShaderStageBits programBindMask = ShaderStageBits::ALL_GRAPHICS_BITS;
    programBindMask = ShaderStageBits::COMPUTE;

    //Clear FB to black
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0, LWN_CLEAR_COLOR_MASK_RGBA);
    queueCB.submit();

    int testNum = 0;
    for (int lwrType = 0; lwrType < TypeEnd; lwrType++) {
        for (int lwrOp = 0; lwrOp < OpEnd; lwrOp++) {

            queueCB.SetViewportScissor(cellWidth * lwrOp + 1, cellHeight * lwrType + 1, 
                cellWidth - 2, cellHeight - 2);

            //Certain atomic combinations are unsupported
            if (isSupportedAtomicForType(device, lwrOp, lwrType)) {
                if (!cellAllowed(testNum % numCols, testNum / numCols))
                    continue;

                Program *program = CreateTestProgram(device, lwrOp, lwrType);
                SSBOLayout *ssboMem = (SSBOLayout *)ssbo->Map();
                if (program) {
                    //Setup test data
                    InitializeTestData(ssboMem, lwrOp, lwrType);

                    queueCB.BindProgram(program, programBindMask);
                    queueCB.BindUniformBuffer(ShaderStage::COMPUTE, 0, uboAddr, uboSize);
                    queueCB.DispatchCompute(1, 1, 1);

                    queueCB.submit();
                    queue->Finish();
                }
                //Verify results based on type and opcode
                bool compileFail = !program;
                bool success = !program ? false : VerifyResultData(ssboMem, lwrOp, lwrType);

                if (compileFail) {
                    //Fill dark red for compilation fail
                    queueCB.ClearColor(0, 0.5, 0.0, 0.0, 1.0, LWN_CLEAR_COLOR_MASK_RGBA);
                } else if (success) {
                    //Fill green square for success
                    queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0, LWN_CLEAR_COLOR_MASK_RGBA);
                }
                else {
                    //Fill red square for runtime fail
                    queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0, LWN_CLEAR_COLOR_MASK_RGBA);
                }
            } else {
                //Fill blue color for unsupported
                queueCB.ClearColor(0, 0.0, 0.0, 1.0, 1.0, LWN_CLEAR_COLOR_MASK_RGBA);
            }

            queueCB.submit();
            queue->Finish();

            testNum++;
        }
    }
}


OGTEST_CppTest(LWNSSBOAtomicTest, lwn_ssbo_extended_atomics, );
