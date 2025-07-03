/*
 * Copyright (c) 2009 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __CPPSHADERS_H__
#define __CPPSHADERS_H__

//
// cppshaders.h
//
// Infrastructure for managing shaders using C++ objects.
//

#include "shaders.h"
#include "cppstring.h"              // string type used for shader source

typedef uint64_t lwShaderHandle;

// lwShaderExtension:  Enumerant class used to handle all shading language
// extensions supported by this package.  When these are set in the shading
// language type, the proper #extension directive is injected automatically
// into the shader code.
class lwShaderExtension {
public:
    enum Extension {

        FirstGeneralExtension = 0,      // marker for iterating over all regular extensions

        LW_vertex_program2 =            // assembly (LW_vertex_program2_option)
            FirstGeneralExtension, 
        LW_vertex_program3,             // assembly
        LW_fragment_program,            // assembly (LW_fragment_program_option)
        LW_fragment_program2,           // assembly
        EXT_gpu_shader4,                // GLSL
        LW_geometry_shader4,            // GLSL
        LW_gpu_shader5,                 // GLSL
        ARB_tessellation_shader,        // GLSL
        LW_shader_atomic_float,         // assembly and GLSL
        LW_shader_atomic_float64,       // assembly and GLSL
        LW_gpu_program5_mem_extended,   // assembly
        LW_shader_atomic_fp16_vector,   // assembly and GLSL
        EXT_shader_image_load_formatted,// GLSL only
        EXT_shader_image_load_store,    // GLSL only (in LW_gpu_program5 for assembly)
        ARB_shader_image_load_store,    // GLSL only
        ARB_shader_storage_buffer_object,      // GLSL only
        LW_shader_storage_buffer,       // assembly
        ARB_texture_lwbe_map_array,     // GLSL
        EXT_geometry_shader,            // GLSL ES >= 3.1
        OES_geometry_shader,            // GLSL ES >= 3.1
        LW_geometry_shader_passthrough, // assembly and GLSL
        LW_shader_thread_group,         // assembly and GLSL
        LW_shader_thread_shuffle,       // assembly and GLSL
        ARB_viewport_array,             // assembly and GLSL
        LW_viewport_array2,             // assembly and GLSL
        EXT_tessellation_shader,        // GLSL ES >= 3.1
        OES_tessellation_shader,        // GLSL ES >= 3.1
        EXT_shader_integer_mix,         // GLSL
        EXT_geometry_point_size,        // GLSL ES >= 3.1
        OES_geometry_point_size,        // GLSL ES >= 3.1
        EXT_tessellation_point_size,    // GLSL ES >= 3.1
        OES_tessellation_point_size,    // GLSL ES >= 3.1
        EXT_shader_io_blocks,           // GLSL ES >= 3.1
        OES_shader_io_blocks,           // GLSL ES >= 3.1
        LW_shader_atomic_int64,         // assembly and GLSL
        ARB_gpu_shader_fp64,            // GLSL
        LW_gpu_program_fp64,            // assembly
        OES_shader_image_atomic,        // GLSL ES >= 3.1
        EXT_texture_buffer,             // GLSL ES >= 3.1
        EXT_texture_lwbe_map_array,     // GLSL ES >= 3.1
        OES_texture_lwbe_map_array,     // GLSL ES >= 3.1
        EXT_gpu_shader5,                // GLSL ES >= 3.1
        OES_gpu_shader5,                // GLSL ES >= 3.1
        ANDROID_extension_pack_es31a,   // GLSL ES >= 3.1
        OES_texture_storage_multisample_2d_array,   // GLSL ES >= 3.1
        LW_image_formats,               // GLSL ES >= 3.1
        LW_viewport_array,              // GLSL ES >= 3.1
        LW_bindless_texture,            // GLSL
        OES_texture_buffer,             // GLSL ES >= 3.1
        ARB_bindless_texture,           // GLSL
        LW_desktop_lowp_mediump,        // GLSL, LWN only
        ARB_shader_ballot,              // GLSL
        ARB_lwll_distance,              // GLSL
        ARB_arrays_of_arrays,           // GLSL
        EXT_post_depth_coverage,        // GLSL
        LW_fragment_shader_interlock,   // GLSL
        ARB_derivative_control,         // GLSL
        LW_extended_pointer_atomics,    // GLSL
        KHR_shader_subgroup_basic,      // GLSL
        KHR_shader_subgroup_vote,       // GLSL
        KHR_shader_subgroup_arithmetic, // GLSL
        KHR_shader_subgroup_ballot,     // GLSL
        KHR_shader_subgroup_shuffle,    // GLSL
        KHR_shader_subgroup_shuffle_relative, // GLSL
        KHR_shader_subgroup_clustered,  // GLSL
        KHR_shader_subgroup_quad,       // GLSL
        LW_shader_subgroup_partitioned, // GLSL
        ARB_gpu_shader_int64,           // GLSL
        LW_separate_texture_types,      // GLSL, LWN only

        EndOfGeneralExtensions,         // marker for iterating over all regular extensions

        // ARB_compatibility is treated as an extension, but it's "special" as
        // it usually shows up in the "#version" directive.
        ARB_compatibility = EndOfGeneralExtensions,

        NoExtension,                    // marker indicating a non-existing extension

        ExtensionCount,
    };
private:
    Extension m_extension;
public:
    lwShaderExtension(Extension ext = NoExtension) : m_extension(ext) 
    {
        assert(m_extension >= 0 && m_extension < ExtensionCount);
    }

    // Used to iterate over all general extensions.
    lwShaderExtension& operator++()
    {
        m_extension = static_cast<Extension>(m_extension + 1);
        assert(m_extension >= 0 && m_extension <= ExtensionCount);
        return *this;
    }

    // int() operator used to identify a bit number for the list below.
    operator int() const        { return int(m_extension); }

    // string:  Returns text needed to be included in shader code to enable
    // the extension.  <isGLSL> indicates whether the shader is GLSL or
    // assembly.  Some extensions may support both; others only one.  This
    // function will assert if called for a shader type that doesn't support
    // it.
    const char *string(LWProgramClass programClass) const;

    // supported:  Returns <true> if and only if the extension is supported.
    bool supported() const;
};

// lwShaderExtensionSupported:  We need to provide a separate external
// function to check for extension support because the enums above often use
// the same names as the lwogtest globals indicating whether an extension is
// supported.  The enums in the class end up hiding the globals in class
// methods.
bool lwShaderExtensionSupported(lwShaderExtension ext);


// lwShaderExtensionSet:  Class indicated a set of extensions supported for a
// particular shader type.  Implemented as a bitset using the
// lwShaderExtension type as an index.
class lwShaderExtensionSet {
private:
    static const int listWordCount = ((lwShaderExtension::ExtensionCount + 31) / 32);
    unsigned int m_words[listWordCount];

    static int listWord(lwShaderExtension ext) {
        return int(ext) / 32;
    }
    static unsigned int listBit(lwShaderExtension ext) {
        return 1U << (int(ext) % 32);
    }

public:
    void add(lwShaderExtension ext) {
        if (ext == lwShaderExtension::NoExtension) return;
        m_words[listWord(ext)] |= listBit(ext);
    }
    void remove(lwShaderExtension ext) {
        if (ext == lwShaderExtension::NoExtension) return;
        m_words[listWord(ext)] &= ~listBit(ext);
    }
    bool includes(lwShaderExtension ext) const
    {
        if (ext == lwShaderExtension::NoExtension) return false;
        if (m_words[listWord(ext)] & listBit(ext)) {
            return true;
        } else {
            return false;
        }
    }
    lwShaderExtensionSet() {
        for (int i = 0; i < listWordCount; i++) m_words[i] = 0;
    }
    lwShaderExtensionSet(lwShaderExtension ext1,
                         lwShaderExtension ext2 = lwShaderExtension::NoExtension,
                         lwShaderExtension ext3 = lwShaderExtension::NoExtension,
                         lwShaderExtension ext4 = lwShaderExtension::NoExtension,
                         lwShaderExtension ext5 = lwShaderExtension::NoExtension)
    {
        for (int i = 0; i < listWordCount; i++) m_words[i] = 0;
        add(ext1);
        add(ext2);
        add(ext3);
        add(ext4);
        add(ext5);
    }
};


// lwShaderType:  Class implementing generalized support for shader types,
// where the program class (e.g., GLSL or assembly, different variants),
// shader stage (e.g., vertex), shader version (e.g., 430), and a list of
// extensions can all be specified independently.
class lwShaderType {
private:
    LWShaderStage           m_shaderStage;
    int                     m_version;
    LWProgramClass          m_programClass;
    lwShaderExtensionSet    m_extensions;

    void init(LWShaderStage shaderStage, int version, 
              LWProgramClass programClass = LWProgramClassGLSLCore,
              lwShaderExtensionSet extensions = lwShaderExtensionSet())
    {
        // Initialize class members.
        m_shaderStage = shaderStage;
        m_version = version;
        m_programClass = programClass;
        m_extensions = extensions;
    }

public:
    // Accessors to get properties of the program type.
    LWShaderStage targetStage() const           { return m_shaderStage; }
    int version() const                         { return m_version; }
    LWProgramClass programClass() const         { return m_programClass; }
    lwShaderExtensionSet extensions() const     { return m_extensions; }

    // Accessors to set properties of the program type.
    void setTargetStage(LWShaderStage stage)            { m_shaderStage = stage; }
    void setVersion(int version)                        { m_version = version; }
    void setProgramClass(LWProgramClass programClass)   { m_programClass = programClass; }
    void setExtensions(lwShaderExtensionSet extensions) { m_extensions = extensions; }

    // Accessors computing derived information based on the program class.
    bool apiIsGLSLCore(void) const              { return m_programClass == LWProgramClassGLSLCore ||
                                                         m_programClass == LWProgramClassGLSLES; }
    bool apiIsGLSLExtension(void) const         { return m_programClass == LWProgramClassGLSLExt; }
    bool apiIsGLSLAny(void) const               { return apiIsGLSLCore() || apiIsGLSLExtension(); }
    bool apiIsNotGL(void) const                 { return m_programClass == LWProgramClassGLSLCore_NonGL ||
                                                         m_programClass == LWProgramClassGLSLES_NonGL; }

    bool languageIsGLSLDesktop(void) const      { return m_programClass == LWProgramClassGLSLCore ||
                                                         m_programClass == LWProgramClassGLSLExt ||
                                                         m_programClass == LWProgramClassGLSLCore_NonGL; }
    bool languageIsGLSLES(void) const           { return m_programClass == LWProgramClassGLSLES ||
                                                         m_programClass == LWProgramClassGLSLES_NonGL; }
    bool languageIsGLSLAny(void) const          { return languageIsGLSLDesktop() || languageIsGLSLES(); }

    // Accessors to determine a particular stage.
    bool isVertex(void) const                   { return targetStage() == VERTEX_STAGE; }
    bool isTessControl(void) const              { return targetStage() == TESS_CONTROL_STAGE; }
    bool isTessEvaluation(void) const           { return targetStage() == TESS_EVALUATION_STAGE; }
    bool isGeometry(void) const                 { return targetStage() == GEOMETRY_STAGE; }
    bool isFragment(void) const                 { return targetStage() == FRAGMENT_STAGE; }
    bool isCompute(void) const                  { return targetStage() == COMPUTE_STAGE; }

    // Determines the enum used to pass into the OpenGL APIs for the
    // particular shader type, a function of the stage and also program class.
    GLenum targetEnum() const;

    // Determines if the program type includes support for a particular
    // extension.
    bool hasExtension(lwShaderExtension ext) const
    {
        return m_extensions.includes(ext);
    }

    // Add or remove support for a particular extension.
    void addExtension(lwShaderExtension ext) {
        m_extensions.add(ext);
    }
    void removeExtension(lwShaderExtension ext) {
        m_extensions.remove(ext);
    }

    // Determines if the program type is supported in the current environment.
    bool isSupported(void) const;

    // Returns a string object containing a shader code that should be
    // injected at the beginning of a shader of the specified type.
    lwString shaderCodePrefix() const;

    // The default constructor is provided to allow for shader type locals
    // without specifying all the parameters up front.
    lwShaderType() {}

    // The primary constructors build the shader type from fully independent
    // parameters.  We have variants that take an extension set as an argument
    // as well as a list of one or more extension defines.
    lwShaderType(LWShaderStage shaderStage, int version, 
                 LWProgramClass programClass = LWProgramClassGLSLCore,
                 lwShaderExtensionSet extensions = lwShaderExtensionSet())
    {
        init(shaderStage, version, programClass, extensions);
    }
    lwShaderType(LWShaderStage shaderStage, int version,
                 LWProgramClass programClass,
                 lwShaderExtension ext1,
                 lwShaderExtension ext2 = lwShaderExtension::NoExtension,
                 lwShaderExtension ext3 = lwShaderExtension::NoExtension,
                 lwShaderExtension ext4 = lwShaderExtension::NoExtension,
                 lwShaderExtension ext5 = lwShaderExtension::NoExtension)
    {
        init(shaderStage, version, programClass, lwShaderExtensionSet(ext1,ext2,ext3,ext4,ext5));
    }

#if !defined(CPPSHADERS_NON_GL_API)
    // We also provide a secondary constructor taking an LWProgramType enum,
    // which maps each to the appropriate combination.  The initial version of
    // this library used of LWProgramType natively, and required a new enum
    // for each supported configuration.
    lwShaderType(LWProgramType);
#endif

};

//
// lwShader is a class used to encapsulate either a GLSL shader object or an
// LW/ARB_vertex_program-style assembly program.
//
class lwShader {

    // Helper union holding shader type-specific parameters.  Depending on the
    // program type, these parameters will be used to generate header code
    // when the shader is compiled and/or make GLSL program parameter calls
    // prior to linking.  These parameters are used this way only if set
    // explicitly through lwShader class methods below; invalid default values
    // for these parameters are chosen, which will not result in any generated
    // code or program parameter calls.
    union lwShaderParams {
        struct TessControl {
            int     patchSize;
        } tcs;
        struct TessEvaluation {
            GLenum  mode;
            GLenum  spacing;
            GLenum  vertexOrder;
            bool    pointMode;
        } tes;
        struct Geometry {
            GLenum  inputType;
            GLenum  outputType;
            int     verticesOut;
            int     ilwocations;
        } gs;
        struct Compute {
            GLuint  width, height, depth;
            GLuint  sharedMemorySize;
        } cs;
        lwShaderParams(lwShaderType type);
    };

public:
    // Helper class holding the dynamically-allocated data for the shader,
    // including information on the API resources the object uses.  The shader
    // class itself only contains a pointer to this object.
    class lwShaderObject : public lwShaderType {
    public:
        lwShaderType    m_type;             // type of the shader
        lwString        m_code;             // shader source code
        lwShaderParams  m_params;           // shader-type specific parameters

        lwShaderObject(lwShaderType type, lwString code = lwString(/*empty*/));
    };

private:
    // The only data in an lwShader object is a handle to its
    // dynamically-allocated data.
    lwShaderHandle m_objectHandle;

    // Get dynamically allocated data (if still active) for the object corresponding
    // to <m_objectHandle>.
    lwShaderObject *getObject();
    const lwShaderObject *getObject() const;

    // Generate a prefix string appropriate to this shader, to be used when
    // the shader is compiled.  Includes a program type-specific prefix and
    // any strings appropriate to the type-specific parameters stored in the
    // shader object.
    lwString codePrefix();

public:
    // Helper method to build a shader object of type <type> from <string>;
    // called by the various constructors.
    void init(lwShaderType type, lwString string);

public:
    // We include constructors of various forms -- default, no code, an lwString, a C
    // string, or an array of C strings.
    lwShader() { m_objectHandle = 0; }
    lwShader(lwShaderType type) {
        init(type, lwString(/*empty*/));
    }
    lwShader(lwShaderType type, lwString string)
    {
        init(type, string);
    }
    lwShader(lwShaderType type, const char *const string)
    {
        init(type, lwString(string));
    }
    lwShader(lwShaderType type, int nStrings, const char * const * strings)
    {
        init(type, lwString(nStrings, strings));
    }

    // The "==" operator is used to determine if two objects have the same
    // dynamic allocation.
    bool operator == (lwShader other) { return m_objectHandle == other.m_objectHandle; }

    // Queries for various properties of the shader allocation.
    const void *handle() const          { return getObject(); }
    bool exists() const                 { return getObject() != NULL; }
    lwShaderType type() const           { return lwShaderType(*getObject()); }
    lwString &code()                    { return getObject()->m_code; }
    const lwString &code() const        { return getObject()->m_code; }

    // Queries for various properties of the shader type.
    LWProgramClass programClass() const     { return getObject() ? getObject()->programClass() : LWProgramClassUnknown; }
    LWShaderStage targetStage() const       { return getObject() ? getObject()->targetStage() : ILWALID_SHADER_STAGE; }
    bool apiIsGLSLCore(void) const          { return getObject() ? getObject()->apiIsGLSLCore() : false; }
    bool apiIsGLSLExtension(void) const     { return getObject() ? getObject()->apiIsGLSLExtension() : false; }
    bool apiIsGLSLAny(void) const           { return getObject() ? getObject()->apiIsGLSLAny() : false; }
    bool apiIsNotGL(void) const             { return getObject() ? getObject()->apiIsNotGL() : false; }
    bool languageIsGLSLDesktop(void) const  { return getObject() ? getObject()->languageIsGLSLDesktop() : false; }
    bool languageIsGLSLES(void) const       { return getObject() ? getObject()->languageIsGLSLES() : false; }

    // The operator "<<" can be used to append <data> to the shader code.
    template <typename T> lwShader & operator << (T data) {
        assert(getObject());
        code().append(data); 
        return *this;
    }

    // Methods to override shader type properties after creating the object.
    // We can't override the program class (assembly/GLSL) or target stage
    // after creating the object because we generate shader objects in the
    // lwShaderObject constructor.  We can modify the extension set or version
    // at any point because these are used to inject code headers when the
    // compile() method is ilwoked.
    void addExtension(lwShaderExtension ext) {
        if (getObject()) {
            getObject()->addExtension(ext);
        }
    }
    void removeExtension(lwShaderExtension ext) {
        if (getObject()) {
            getObject()->removeExtension(ext);
        }
    }
    void setVersion(int version) {
        if (getObject()) {
            getObject()->setVersion(version);
        }
    }

    // The source() method extracts the source code from the shader, including
    // any prefixes/suffixes.
    lwString source(void);

    // Methods to set geometry shader-specific parameters.  These methods
    // have no effect if called for any other shader type.
    void setGSInputType(GLenum inputType);
    void setGSOutputType(GLenum outputType);
    void setGSVerticesOut(int verticesOut);
    void setGSIlwocations(int ilwocations);
    void setGSParameters(GLenum inputType, GLenum outputType, 
                         int verticesOut, int ilwocations = 0);

    // Methods to set tessellation control shader-specific parameters.  These
    // methods have no effect if called for any other shader type.
    void setTCSOutPatchSize(GLint size);
    void setTCSParameters(GLint size);

    // Methods to set tessellation evaluation shader-specific parameters.
    // These methods have no effect if called for any other shader type.
    void setTESMode(GLenum mode);
    void setTESSpacing(GLenum spacing);
    void setTESVertexOrder(GLenum order);
    void setTESPointMode(GLboolean pointMode);
    void setTESParameters(GLenum mode = GL_QUADS, GLenum spacing = GL_EQUAL,
                          GLenum order = GL_CCW, GLboolean pointMode = GL_FALSE);

    // Methods to set compute shader-specific parameters.  These methods have
    // no effect if called for any other shader type.
    void setCSGroupSize(GLuint width, GLuint height = 0, GLuint depth = 0);
    void setCSSharedMemory(GLuint bytes);

};


// lwShaderWithStageClass:  Template class used to stamp out derived classes
// for various shader stage (e.g., vertex) and class (e.g., GLSL core)
// combinations.
template <LWShaderStage Stage, LWProgramClass Class> class lwShaderWithStageClass : public lwShader
{
public:
    lwShaderWithStageClass() : lwShader() {}
    lwShaderWithStageClass(int version) :
        lwShader(lwShaderType(Stage, version, Class)) {}
};

// Specific shader types for GLSL shaders of each stage.
typedef class lwShaderWithStageClass <VERTEX_STAGE, LWProgramClassGLSLCore> 
    lwGLSLVertexShader;
typedef class lwShaderWithStageClass <TESS_CONTROL_STAGE, LWProgramClassGLSLCore> 
    lwGLSLTessControlShader;
typedef class lwShaderWithStageClass <TESS_EVALUATION_STAGE, LWProgramClassGLSLCore> 
    lwGLSLTessEvaluationShader;
typedef class lwShaderWithStageClass <GEOMETRY_STAGE, LWProgramClassGLSLCore> 
    lwGLSLGeometryShader;
typedef class lwShaderWithStageClass <FRAGMENT_STAGE, LWProgramClassGLSLCore> 
    lwGLSLFragmentShader;
typedef class lwShaderWithStageClass <COMPUTE_STAGE, LWProgramClassGLSLCore> 
    lwGLSLComputeShader;

// Specific shader types for GLSL shaders of each stage (non-OpenGL API).
typedef class lwShaderWithStageClass <VERTEX_STAGE, LWProgramClassGLSLCore_NonGL> 
    lwGLSLVertexShader_NonGL;
typedef class lwShaderWithStageClass <TESS_CONTROL_STAGE, LWProgramClassGLSLCore_NonGL> 
    lwGLSLTessControlShader_NonGL;
typedef class lwShaderWithStageClass <TESS_EVALUATION_STAGE, LWProgramClassGLSLCore_NonGL> 
    lwGLSLTessEvaluationShader_NonGL;
typedef class lwShaderWithStageClass <GEOMETRY_STAGE, LWProgramClassGLSLCore_NonGL> 
    lwGLSLGeometryShader_NonGL;
typedef class lwShaderWithStageClass <FRAGMENT_STAGE, LWProgramClassGLSLCore_NonGL> 
    lwGLSLFragmentShader_NonGL;
typedef class lwShaderWithStageClass <COMPUTE_STAGE, LWProgramClassGLSLCore_NonGL> 
    lwGLSLComputeShader_NonGL;

// Specific shader types for GLSL ES shaders of each stage (non-OpenGL API).
typedef class lwShaderWithStageClass <VERTEX_STAGE, LWProgramClassGLSLES_NonGL> 
    lwESVertexShader_NonGL;
typedef class lwShaderWithStageClass <TESS_CONTROL_STAGE, LWProgramClassGLSLES_NonGL>
    lwESTessControlShader_NonGL;
typedef class lwShaderWithStageClass <TESS_EVALUATION_STAGE, LWProgramClassGLSLES_NonGL>
    lwESTessEvaluationShader_NonGL;
typedef class lwShaderWithStageClass <GEOMETRY_STAGE, LWProgramClassGLSLES_NonGL>
    lwESGeometryShader_NonGL;
typedef class lwShaderWithStageClass <FRAGMENT_STAGE, LWProgramClassGLSLES_NonGL> 
    lwESFragmentShader_NonGL;
typedef class lwShaderWithStageClass <COMPUTE_STAGE, LWProgramClassGLSLES_NonGL>
    lwESComputeShader_NonGL;


// shaderCleanup:  Unbinds and deletes all shader and program objects being
// tracked.
extern void shaderCleanup(void);


#endif // #ifndef __CPPSHADERS_H__
