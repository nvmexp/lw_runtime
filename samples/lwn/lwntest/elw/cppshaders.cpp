/*
 * Copyright (c) 2009 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//
// cppshaders.cpp
//
// Implementations for the C++ shader object classes.
//
#include <vector>

#include "lwntest_c.h"
#include "cppshaders.h"
#include "contexts.h"
#include "cmdline.h"

// lwShaderHandlePool maintains a list of active dynamically allocated shader
// objects, which will be wiped clean by shaderCleanup().  Actual objects are
// referred to by lwShaderHandle (a 64-bit index) and translated to pointers by
// this class.  Each allocation is assigned an incrementing index as a handle,
// and the handle is considered valid if it was allocated after the most recent
// shaderCleanup() call.  Invalid handles are translated to NULL pointers.
//
// The pool keeps an array of pointers allocated since the last cleanup where a
// handle of X maps to array element X-B, where B was the next free handle at
// the time of the last cleanup.  If the index is less than B or greater than
// the next index (in the unlikely event of wrapping), the pointer should be
// considered invalid.
//
// We do this so that tests using non-local (static, global, object) lwShader
// don't attempt to reuse pointers to objects deleted by shader cleanup.  These
// classes dynamically allocate new objects as needed, but used to try to reuse
// an existing one if it held a stale pointer.  We don't attempt to use smart
// pointers for these dynamic allocations because storing one in a global will
// ensure the memory is never freed.

template <class T>
class lwShaderHandlePool {
public:
    lwShaderHandlePool() : nextAlloc(1), start(1) {;}

    // Look up the handle, checking to make sure the handle was allocated 
    // since the last destroy
    T* operator[](lwShaderHandle h) {
        if (h < start || h >= nextAlloc) {
            return NULL;
        }
        return data[h-start];
    }

    // Allocate a handle for a T* pointer allocated by new
    // This pointer will be deleted by the pool
    lwShaderHandle alloc(T *newT) {
        lwShaderHandle rv = nextAlloc++;

        data.push_back(newT);
        assert(data.size() == rv - start + 1);

        return rv;
    }

    // delete all lwrrently tracked allocations
    void destroy() {
        for (int i=0; i<(int)data.size(); i++) {
            delete data[i];
        }
        data.clear();
        start = nextAlloc;
    }
private:
    // Half-open interval [start, nextAlloc) is valid
    // Next handle value to be given out
    lwShaderHandle nextAlloc;
    // Start of the current region
    lwShaderHandle start;
    std::vector<T*> data;
};

// ShaderMachine:  Internal class to track shader objects.
class ShaderMachine {
private:
    lwShaderHandlePool<lwShader::lwShaderObject> m_shaderObjects;
public:
    ShaderMachine();

    // Access to the handle pool to manage the inner allocations
    lwShaderHandlePool<lwShader::lwShaderObject>& shaderObjects()
        { return m_shaderObjects; }

    // cleanup:  Called to free memory for all previously allocated objects.
    void cleanup(void);
};

ShaderMachine::ShaderMachine(void)
{
}


void ShaderMachine::cleanup(void)
{
    // Destroy all the shader objects we're still tracking and
    // reset the track lists.
    m_shaderObjects.destroy(); 
}

// shaderMachine:  Global shader machine object to keep track of the current
// state of affairs.
ShaderMachine shaderMachine;

// shaderCleanup:  Unbinds and deletes all shader objects being
// tracked.
void shaderCleanup()
{
    shaderMachine.cleanup();
}

////////////////////////////////////////////////////////////////////////

// Macros to generate a case statement returning appropriate OPTION or
// #extension directives for a specific extension enum.
#define ADD_ASSEMBLY_EXTENSION(_x)                          \
    case _x:                                                \
        assert(programClass == LWProgramClassAssembly ||    \
               programClass == LWProgramClassAssemblyLW);   \
        return "OPTION " #_x ";\n"

#define ADD_GLSL_EXTENSION(_x)                              \
    case _x:                                                \
        assert(programClass != LWProgramClassAssembly &&    \
               programClass != LWProgramClassAssemblyLW);   \
        return "#extension GL_" #_x  " : enable\n";

#define ADD_ASSEMBLY_OR_GLSL_EXTENSION(_x)              \
    case _x:                                            \
        if (programClass == LWProgramClassAssembly ||   \
            programClass == LWProgramClassAssemblyLW)   \
        {                                               \
            return "OPTION " #_x ";\n";                 \
        } else {                                        \
            return "#extension GL_" #_x " : enable\n";  \
        }


// Returns an appropriate "#extension" or OPTION string for each supported
// extension, when used to compile a shader of class <programClass>.
const char * lwShaderExtension::string(LWProgramClass programClass) const
{
    switch (m_extension) {
    ADD_GLSL_EXTENSION(ARB_compatibility);
    ADD_ASSEMBLY_EXTENSION(LW_vertex_program2);
    ADD_ASSEMBLY_EXTENSION(LW_vertex_program3);
    ADD_ASSEMBLY_EXTENSION(LW_fragment_program);
    ADD_ASSEMBLY_EXTENSION(LW_fragment_program2);
    ADD_GLSL_EXTENSION(EXT_gpu_shader4);
    ADD_GLSL_EXTENSION(LW_geometry_shader4);
    ADD_GLSL_EXTENSION(LW_gpu_shader5);
    ADD_GLSL_EXTENSION(ARB_tessellation_shader);
    ADD_ASSEMBLY_OR_GLSL_EXTENSION(LW_shader_atomic_float);
    ADD_ASSEMBLY_OR_GLSL_EXTENSION(LW_shader_atomic_float64);
    ADD_ASSEMBLY_OR_GLSL_EXTENSION(LW_shader_atomic_fp16_vector);
    ADD_ASSEMBLY_OR_GLSL_EXTENSION(LW_shader_atomic_int64);
    ADD_ASSEMBLY_EXTENSION(LW_gpu_program5_mem_extended);
    ADD_GLSL_EXTENSION(EXT_shader_image_load_formatted);
    ADD_GLSL_EXTENSION(EXT_shader_image_load_store);
    ADD_GLSL_EXTENSION(ARB_shader_image_load_store);
    ADD_GLSL_EXTENSION(ARB_shader_storage_buffer_object);
    ADD_ASSEMBLY_EXTENSION(LW_shader_storage_buffer);
    ADD_GLSL_EXTENSION(ARB_texture_lwbe_map_array);
    ADD_GLSL_EXTENSION(EXT_geometry_shader);
    ADD_GLSL_EXTENSION(OES_geometry_shader);
    ADD_ASSEMBLY_OR_GLSL_EXTENSION(LW_geometry_shader_passthrough);
    ADD_ASSEMBLY_OR_GLSL_EXTENSION(LW_shader_thread_group);
    ADD_ASSEMBLY_OR_GLSL_EXTENSION(LW_shader_thread_shuffle);
    ADD_ASSEMBLY_OR_GLSL_EXTENSION(ARB_viewport_array);
    ADD_ASSEMBLY_OR_GLSL_EXTENSION(LW_viewport_array2);
    ADD_GLSL_EXTENSION(EXT_tessellation_shader);
    ADD_GLSL_EXTENSION(OES_tessellation_shader);
    ADD_GLSL_EXTENSION(EXT_shader_integer_mix);
    ADD_GLSL_EXTENSION(EXT_geometry_point_size);
    ADD_GLSL_EXTENSION(OES_geometry_point_size);
    ADD_GLSL_EXTENSION(EXT_tessellation_point_size);
    ADD_GLSL_EXTENSION(OES_tessellation_point_size);
    ADD_GLSL_EXTENSION(EXT_shader_io_blocks);
    ADD_GLSL_EXTENSION(OES_shader_io_blocks);
    ADD_GLSL_EXTENSION(ARB_gpu_shader_fp64);
    ADD_ASSEMBLY_EXTENSION(LW_gpu_program_fp64);
    ADD_GLSL_EXTENSION(OES_shader_image_atomic);
    ADD_GLSL_EXTENSION(EXT_texture_buffer);
    ADD_GLSL_EXTENSION(EXT_texture_lwbe_map_array);
    ADD_GLSL_EXTENSION(OES_texture_lwbe_map_array);
    ADD_GLSL_EXTENSION(EXT_gpu_shader5);
    ADD_GLSL_EXTENSION(OES_gpu_shader5);
    ADD_GLSL_EXTENSION(ANDROID_extension_pack_es31a);
    ADD_GLSL_EXTENSION(OES_texture_storage_multisample_2d_array);
    ADD_GLSL_EXTENSION(LW_image_formats);
    ADD_GLSL_EXTENSION(LW_viewport_array);
    ADD_GLSL_EXTENSION(LW_bindless_texture);
    ADD_GLSL_EXTENSION(OES_texture_buffer);
    ADD_GLSL_EXTENSION(ARB_bindless_texture);
    ADD_GLSL_EXTENSION(LW_desktop_lowp_mediump);
    ADD_GLSL_EXTENSION(ARB_shader_ballot);
    ADD_GLSL_EXTENSION(ARB_lwll_distance);
    ADD_GLSL_EXTENSION(ARB_arrays_of_arrays);
    ADD_GLSL_EXTENSION(EXT_post_depth_coverage);
    ADD_GLSL_EXTENSION(LW_fragment_shader_interlock);
    ADD_GLSL_EXTENSION(ARB_derivative_control);
    ADD_GLSL_EXTENSION(LW_extended_pointer_atomics);
    ADD_GLSL_EXTENSION(KHR_shader_subgroup_basic);
    ADD_GLSL_EXTENSION(KHR_shader_subgroup_vote);
    ADD_GLSL_EXTENSION(KHR_shader_subgroup_arithmetic);
    ADD_GLSL_EXTENSION(KHR_shader_subgroup_ballot);
    ADD_GLSL_EXTENSION(KHR_shader_subgroup_shuffle);
    ADD_GLSL_EXTENSION(KHR_shader_subgroup_shuffle_relative);
    ADD_GLSL_EXTENSION(KHR_shader_subgroup_clustered);
    ADD_GLSL_EXTENSION(KHR_shader_subgroup_quad);
    ADD_GLSL_EXTENSION(LW_shader_subgroup_partitioned);
    ADD_GLSL_EXTENSION(ARB_gpu_shader_int64);
    ADD_GLSL_EXTENSION(LW_separate_texture_types);
    default:
        assert(0);
        return "ILWALID_EXTENSION";
    }
}


// Check if the shader extension is supported on the platform in question.
// Use the external function lwShaderExtensionSupported() because the enum
// definition hides the global variables indicating extension support in the
// class methods.
bool lwShaderExtension::supported() const
{
    return lwShaderExtensionSupported(m_extension);
}

// Macros to generate a case statement returning appropriate OPTION or
// #extension directives for a specific extension enum.
#define CHECK_GLSL_EXTENSION(_x) \
    case lwShaderExtension::_x: return true
#define CHECK_ASSEMBLY_EXTENSION(_x) \
    case lwShaderExtension::_x: return false
#define CHECK_ASSEMBLY_OR_GLSL_EXTENSION(_x) \
    case lwShaderExtension::_x: return true

bool lwShaderExtensionSupported(lwShaderExtension ext)
{
    switch (ext) {
    CHECK_ASSEMBLY_EXTENSION(LW_vertex_program2);
    CHECK_ASSEMBLY_EXTENSION(LW_vertex_program3);
    CHECK_ASSEMBLY_EXTENSION(LW_fragment_program);
    CHECK_ASSEMBLY_EXTENSION(LW_fragment_program2);
    CHECK_GLSL_EXTENSION(EXT_gpu_shader4);
    CHECK_GLSL_EXTENSION(LW_geometry_shader4);
    CHECK_GLSL_EXTENSION(LW_gpu_shader5);
    CHECK_GLSL_EXTENSION(ARB_tessellation_shader);
    CHECK_ASSEMBLY_OR_GLSL_EXTENSION(LW_shader_atomic_float);
    CHECK_ASSEMBLY_OR_GLSL_EXTENSION(LW_shader_atomic_float64);
    CHECK_ASSEMBLY_OR_GLSL_EXTENSION(LW_shader_atomic_fp16_vector);
    CHECK_ASSEMBLY_OR_GLSL_EXTENSION(LW_shader_atomic_int64);
    CHECK_ASSEMBLY_EXTENSION(LW_gpu_program5_mem_extended);
    CHECK_GLSL_EXTENSION(EXT_shader_image_load_formatted);
    CHECK_GLSL_EXTENSION(EXT_shader_image_load_store);
    CHECK_GLSL_EXTENSION(ARB_shader_image_load_store);
    CHECK_GLSL_EXTENSION(ARB_shader_storage_buffer_object);
    CHECK_ASSEMBLY_EXTENSION(LW_shader_storage_buffer);
    CHECK_GLSL_EXTENSION(ARB_texture_lwbe_map_array);
    CHECK_GLSL_EXTENSION(EXT_geometry_shader);
    CHECK_GLSL_EXTENSION(OES_geometry_shader);
    CHECK_ASSEMBLY_OR_GLSL_EXTENSION(LW_geometry_shader_passthrough);
    CHECK_ASSEMBLY_OR_GLSL_EXTENSION(LW_shader_thread_group);
    CHECK_ASSEMBLY_OR_GLSL_EXTENSION(LW_shader_thread_shuffle);
    CHECK_ASSEMBLY_OR_GLSL_EXTENSION(ARB_viewport_array);
    CHECK_ASSEMBLY_OR_GLSL_EXTENSION(LW_viewport_array2);
    CHECK_GLSL_EXTENSION(EXT_tessellation_shader);
    CHECK_GLSL_EXTENSION(OES_tessellation_shader);
    CHECK_GLSL_EXTENSION(EXT_shader_integer_mix);
    CHECK_GLSL_EXTENSION(EXT_geometry_point_size);
    CHECK_GLSL_EXTENSION(OES_geometry_point_size);
    CHECK_GLSL_EXTENSION(EXT_tessellation_point_size);
    CHECK_GLSL_EXTENSION(OES_tessellation_point_size);
    CHECK_GLSL_EXTENSION(EXT_shader_io_blocks);
    CHECK_GLSL_EXTENSION(OES_shader_io_blocks);
    CHECK_GLSL_EXTENSION(ARB_gpu_shader_fp64);
    CHECK_ASSEMBLY_EXTENSION(LW_gpu_program_fp64);
    CHECK_GLSL_EXTENSION(OES_shader_image_atomic);
    CHECK_GLSL_EXTENSION(EXT_texture_buffer);
    CHECK_GLSL_EXTENSION(EXT_texture_lwbe_map_array);
    CHECK_GLSL_EXTENSION(OES_texture_lwbe_map_array);
    CHECK_GLSL_EXTENSION(EXT_gpu_shader5);
    CHECK_GLSL_EXTENSION(OES_gpu_shader5);
    CHECK_GLSL_EXTENSION(ANDROID_extension_pack_es31a);
    CHECK_GLSL_EXTENSION(OES_texture_storage_multisample_2d_array);
    CHECK_GLSL_EXTENSION(LW_image_formats);
    CHECK_GLSL_EXTENSION(LW_viewport_array);
    CHECK_GLSL_EXTENSION(LW_bindless_texture);
    CHECK_GLSL_EXTENSION(OES_texture_buffer);
    CHECK_GLSL_EXTENSION(ARB_bindless_texture);
    CHECK_GLSL_EXTENSION(LW_desktop_lowp_mediump);
    CHECK_GLSL_EXTENSION(KHR_shader_subgroup_basic);
    CHECK_GLSL_EXTENSION(KHR_shader_subgroup_vote);
    CHECK_GLSL_EXTENSION(KHR_shader_subgroup_arithmetic);
    CHECK_GLSL_EXTENSION(KHR_shader_subgroup_ballot);
    CHECK_GLSL_EXTENSION(KHR_shader_subgroup_shuffle);
    CHECK_GLSL_EXTENSION(KHR_shader_subgroup_shuffle_relative);
    CHECK_GLSL_EXTENSION(KHR_shader_subgroup_clustered);
    CHECK_GLSL_EXTENSION(KHR_shader_subgroup_quad);
    CHECK_GLSL_EXTENSION(LW_shader_subgroup_partitioned);
    CHECK_GLSL_EXTENSION(ARB_gpu_shader_int64);

    case lwShaderExtension::NoExtension:
        return true;

    CHECK_GLSL_EXTENSION(ARB_compatibility);
    default:
        assert(0);
        return false;
    }
}

////////////////////////////////////////////////////////////////////////

// Build up and return a string holding the initial source that should be
// prepended to shader code of the specified shader type.
lwString lwShaderType::shaderCodePrefix() const
{
    lwStringBuf sb;

    // Generate an appropriate #version or "!!" assembly directive according
    // to the program class.
    if (languageIsGLSLAny()) {

        // Generate a #version directive for all ES shaders, as well as all
        // desktop versions but except 1.00 and 1.10.
        if (languageIsGLSLES() || m_version > 110) {

            sb << "#version " << m_version;

            // Generate an appropriate "compatibility", or "es" directive, as
            // required.  For desktop shaders, we don't generate "core", which
            // is the default for #version 140+.
            if (languageIsGLSLES()) {
                if (m_version >= 300) sb << " es";
            }
            if (hasExtension(lwShaderExtension::ARB_compatibility)) {
                if (m_version == 140) {
                    sb << "\n#extension GL_ARB_compatibility : enable\n";
                } else {
                    sb << " compatibility";
                }
            }

            sb << "\n";
        }

    } else {
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }

    // After the base header, add in #extension/OPTION directives for
    // all enabled extensions.
    for (lwShaderExtension ext = lwShaderExtension::FirstGeneralExtension;
         ext != lwShaderExtension::EndOfGeneralExtensions; ++ext) 
    {
        if (m_extensions.includes(ext)) {
            sb << ext.string(m_programClass);
        }
    }

    return sb.str();
}

// Returns true if the shader type is supported on the current system.  Note
// that we intentionally don't return early because this function is used to
// probe for unsupported/invalid versions or parameter combinations.
bool lwShaderType::isSupported(void) const
{
    bool result = true;

    // Check for support for the base program type based on the class and
    // version.
    switch (m_programClass) {

    case LWProgramClassGLSLCore:
        switch (m_version) {
        case 100:
        case 110:
        case 120:
        case 130:
        case 140:
        case 150:
        case 330:
        case 400:
        case 410:
        case 420:
        case 430:
        case 440:
        case 450:
            break;
        default:
            assert(0); 
            result = false; 
            break;
        }
        break;

    case LWProgramClassGLSLES:
        // ES shading language versions are supported for both real ES
        // implementations as well as ARB_ES*_compatibility on desktop.
        switch (m_version) {
        case 100:
        case 300:
        case 310:
        case 320:
            break;
        default:
            assert(0);
            result = false;
            break;
        }
        break;

    case LWProgramClassGLSLExt:
    case LWProgramClassAssembly:
    case LWProgramClassAssemblyLW:
        // Only support GLSL
        assert(0); 
        result = false;
        break;

    case LWProgramClassGLSLCore_NonGL:
        // For non-GL APIs, we support all known versions of GLSL.
        switch (m_version) {
        case 100:
        case 110:
        case 120:
        case 130:
        case 140:
        case 150:
        case 330:
        case 400:
        case 410:
        case 420:
        case 430:
        case 440:
        case 450:
            break;
        default:
            assert(0);
            result = false;
            break;
        }
        break;

    case LWProgramClassGLSLES_NonGL:
        // For non-GL APIs, we support all known versions of GLSL ES.
        switch (m_version) {
        case 100:
        case 300:
        case 310:
        case 320:
            break;
        default:
            assert(0);
            result = false;
            break;
        }
        break;

    default:
        assert(0);
        result = false;
        break;
    }

    // Iterate over all the extensions to determine if 
    for (lwShaderExtension ext = lwShaderExtension::FirstGeneralExtension;
         ext != lwShaderExtension::EndOfGeneralExtensions; ++ext) 
    {
        if (m_extensions.includes(ext) && !ext.supported()) {
            result = false;
        }
    }

    return result;
}
////////////////////////////////////////////////////////////////////////

lwShader::lwShaderParams::lwShaderParams(lwShaderType type)
{
    switch (type.targetStage()) {
    case TESS_CONTROL_STAGE:
        tcs.patchSize = 0;
        break;
    case TESS_EVALUATION_STAGE:
        tes.mode = GL_ILWALID_ENUM;
        tes.spacing = GL_ILWALID_ENUM;
        tes.vertexOrder = GL_ILWALID_ENUM;
        tes.pointMode = false;
        break;
    case GEOMETRY_STAGE:
        gs.inputType = GL_ILWALID_ENUM;
        gs.outputType = GL_ILWALID_ENUM;
        gs.verticesOut = 0;
        gs.ilwocations = 0;
        break;
    case COMPUTE_STAGE:
        cs.width = 0;
        cs.height = 0;
        cs.depth = 0;
        cs.sharedMemorySize = 0;
        break;
    default:
        break;
    }
}

lwShader::lwShaderObject::lwShaderObject(lwShaderType type, lwString code) : 
    lwShaderType(type), 
    m_code(code),
    m_params(type)
{
}

void lwShader::init(lwShaderType shaderType, lwString string)
{
    m_objectHandle = shaderMachine.shaderObjects().alloc(new  lwShaderObject(shaderType, string));
}

lwShader::lwShaderObject * lwShader::getObject()
{
    return shaderMachine.shaderObjects()[m_objectHandle];
}

const lwShader::lwShaderObject * lwShader::getObject() const
{
    return shaderMachine.shaderObjects()[m_objectHandle];
}

lwString lwShader::source(void)
{
    lwString shaderCode;
    shaderCode = codePrefix();
    shaderCode += code();
    return shaderCode;
}

void lwShader::setGSInputType(GLenum inputType)
{
    if (targetStage() != GEOMETRY_STAGE) return;
    getObject()->m_params.gs.inputType = inputType;
}

void lwShader::setGSOutputType(GLenum outputType)
{
    if (targetStage() != GEOMETRY_STAGE) return;
    getObject()->m_params.gs.outputType = outputType;
}

void lwShader::setGSVerticesOut(int verticesOut)
{
    if (targetStage() != GEOMETRY_STAGE) return;
    getObject()->m_params.gs.verticesOut = verticesOut;
}

void lwShader::setGSIlwocations(int ilwocations)
{
    if (targetStage() != GEOMETRY_STAGE) return;
    getObject()->m_params.gs.ilwocations = ilwocations;

}

void lwShader::setGSParameters(GLenum inputType, GLenum outputType, 
                               int verticesOut, int ilwocations /*= 0*/)
{
    setGSInputType(inputType);
    setGSOutputType(outputType);
    setGSVerticesOut(verticesOut);
    setGSIlwocations(ilwocations);
}

void lwShader::setTCSOutPatchSize(GLint size)
{
    if (targetStage() != TESS_CONTROL_STAGE) return;
    getObject()->m_params.tcs.patchSize = size;
}

void lwShader::setTCSParameters(GLint size)
{
    setTCSOutPatchSize(size);
}

void lwShader::setTESMode(GLenum mode)
{
    if (targetStage() != TESS_EVALUATION_STAGE) return;
    getObject()->m_params.tes.mode = mode;
}

void lwShader::setTESSpacing(GLenum spacing)
{
    if (targetStage() != TESS_EVALUATION_STAGE) return;
    getObject()->m_params.tes.spacing = spacing;
}

void lwShader::setTESVertexOrder(GLenum order)
{
    if (targetStage() != TESS_EVALUATION_STAGE) return;
    getObject()->m_params.tes.vertexOrder = order;
}

void lwShader::setTESPointMode(GLboolean pointMode)
{
    if (targetStage() != TESS_EVALUATION_STAGE) return;
    getObject()->m_params.tes.pointMode = (pointMode != GL_FALSE);
}

void lwShader::setCSGroupSize(GLuint width, GLuint height /*= 0*/, GLuint depth /*= 0*/)
{
    lwShaderObject *object = getObject();

    if (targetStage() != COMPUTE_STAGE) return;
    object->m_params.cs.width = width;
    object->m_params.cs.height = height;
    object->m_params.cs.depth = depth;
}

void lwShader::setCSSharedMemory(GLuint bytes)
{
    if (targetStage() != COMPUTE_STAGE) return;
    getObject()->m_params.cs.sharedMemorySize = bytes;
}

void lwShader::setTESParameters(GLenum mode /*= GL_QUADS*/, GLenum spacing /*= GL_EQUAL*/, 
                                GLenum order /*= GL_CCW*/, GLboolean pointMode /*= GL_FALSE*/)
{
    setTESMode(mode);
    setTESSpacing(spacing);
    setTESVertexOrder(order);
    setTESPointMode(pointMode);
}

lwString lwShader::codePrefix()
{
    LWShaderStage stage = targetStage();
    lwString code;
    lwShaderObject *object = getObject();

    ct_assert((LWNshaderStage)VERTEX_STAGE == LWN_SHADER_STAGE_VERTEX);
    ct_assert((LWNshaderStage)FRAGMENT_STAGE == LWN_SHADER_STAGE_FRAGMENT);
    ct_assert((LWNshaderStage)GEOMETRY_STAGE == LWN_SHADER_STAGE_GEOMETRY);
    ct_assert((LWNshaderStage)TESS_CONTROL_STAGE == LWN_SHADER_STAGE_TESS_CONTROL);
    ct_assert((LWNshaderStage)TESS_EVALUATION_STAGE == LWN_SHADER_STAGE_TESS_EVALUATION);

    bool compileStageAsFp16 = ((lwnCompileAsFp16Mask & (1U << stage)) != 0);

    if (compileStageAsFp16) {
        addExtension(lwShaderExtension::LW_desktop_lowp_mediump);
    }

    code = type().shaderCodePrefix();

    if (type().version() >= 150) {
        if (stage == TESS_CONTROL_STAGE) {
            lwShaderParams::TessControl &tcs = object->m_params.tcs;
            if (tcs.patchSize) {
                code += "layout(vertices=";
                code += tcs.patchSize;
                code += ") out;\n";
            }
        } else if (stage == TESS_EVALUATION_STAGE) {
            lwShaderParams::TessEvaluation &tes = object->m_params.tes;
            switch (tes.mode) {
            case GL_ILWALID_ENUM:
                break;
            case GL_TRIANGLES:
                code += "layout(triangles) in;\n"; break;
            case GL_QUADS:
                code += "layout(quads) in;\n"; break;
            case GL_ISOLINES:
                code += "layout(isolines) in;\n"; break;
            default:
                code += "layout(unknown) in;\n"; break;
            }
            switch (tes.spacing) {
            case GL_ILWALID_ENUM:
                break;
            case GL_EQUAL:
                code += "layout(equal_spacing) in;\n"; break;
            case GL_FRACTIONAL_EVEN:
                code += "layout(fractional_even_spacing) in;\n"; break;
            case GL_FRACTIONAL_ODD:
                code += "layout(fractional_odd_spacing) in;\n"; break;
            default:
                code += "layout(unknown_spacing) in;\n"; break;
            }
            switch (tes.vertexOrder) {
            case GL_ILWALID_ENUM:
                break;
            case GL_CCW:
                code += "layout(ccw) in;\n"; break;
            case GL_CW:
                code += "layout(cw) in;\n"; break;
            default:
                code += "layout(unknown_order) in;\n"; break;
            }
            if (tes.pointMode) {
                code += "layout(point_mode) in;\n";
            }
        } else if (stage == GEOMETRY_STAGE) {
            lwShaderParams::Geometry &gs = object->m_params.gs;
            switch (gs.inputType) {
            case GL_ILWALID_ENUM:
                break;
            case GL_POINTS:
                code += "layout(points) in;\n"; break;
            case GL_LINES:
                code += "layout(lines) in;\n"; break;
            case GL_TRIANGLES:
                code += "layout(triangles) in;\n"; break;
            case GL_LINES_ADJACENCY:
                code += "layout(lines_adjacency) in;\n"; break;
            case GL_TRIANGLES_ADJACENCY:
                code += "layout(triangles_adjacency) in;\n"; break;
            case GL_PATCHES:
                code += "layout(patches) in;\n"; break;
            default:
                code += "layout(unknown_topology) in;\n"; break;
            }
            if (gs.ilwocations) {
                code += "layout(ilwocations=";
                code += gs.ilwocations;
                code += ") in;\n";
            }
            switch (gs.outputType) {
            case GL_ILWALID_ENUM:
                break;
            case GL_POINTS:
                code += "layout(points) out;\n"; break;
            case GL_LINE_STRIP:
                code += "layout(line_strip) out;\n"; break;
            case GL_TRIANGLE_STRIP:
                code += "layout(triangle_strip) out;\n"; break;
            default:
                code += "layout(unknown_topology) out;\n"; break;

            }
            if (gs.verticesOut) {
                code += "layout(max_vertices=";
                code += gs.verticesOut;
                code += ") out;\n";
            }
        } else if (stage == COMPUTE_STAGE) {
            lwShaderParams::Compute &cs = object->m_params.cs;
            if (cs.width || cs.height || cs.depth) {
                code += "layout(local_size_x=";
                code += cs.width;
                if (cs.height || cs.depth) {
                    code += ", local_size_y=";
                    code += cs.height;
                }
                if (cs.depth) {
                    code += ", local_size_z=";
                    code += cs.depth;
                }
                code += ") in;\n";
            }
            // Shared memory size inferred from variable usage in GLSL.
        }
    }

    // Adds default precision modifier.  Make sure this comes after the # extensions.
    if (compileStageAsFp16) {
        code += "precision mediump float;\n";
    }

    return code;
}

