#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;
using namespace lwn::dt;

class LWNShaderSpecialization
{
public:
    union ArrayUnion {
        int32_t  i[16];
        uint32_t u[16];
        float    f[16];
        double   d[16];
    };

    enum ArgTypeEnum {
        ARG_TYPE_INT = 0,
        ARG_TYPE_DOUBLE = 1,
        ARG_TYPE_FLOAT = 2,
        ARG_TYPE_UINT = 3
    };

    static const int total_tests = 75;
    static const int cellsX = 32;
    static const int cellsY = 32;

    struct Vertex {
        dt::vec2 position;
    };
    LWNTEST_CppMethods();
    bool RunTest(int test_number, Program *pgm) const;
    void SetData(GLSLCspecializationUniform * uniform, const char * name, int numElements,
                 ArgTypeEnum type, int numArgs, ... ) const;
};

lwString LWNShaderSpecialization::getDescription() const {
    lwStringBuf sb;
    sb << "Tests for testing the shader specialization feature for LWN.";
    sb << "Each test is designed to exercise a sepcific combination of bindings and";
    sb << "specialization parameters.";
    sb << "Pass - Green (0.0, 1.0, 0.0).";
    sb << "Test Fail - Red (1.0, 0.0, 0.0).";
    sb << "Compilation error - Dark red (0.5, 0.0, 0.0).";
    return sb.str();
}

int LWNShaderSpecialization::isSupported() const {
    return lwogCheckLWNAPIVersion(38, 3);
}

void LWNShaderSpecialization::SetData(GLSLCspecializationUniform * uniform, const char * name, int numElements,
                    ArgTypeEnum type, int numArgs, ... ) const
{

    assert(numArgs % numElements == 0);

    ArrayUnion * arryUnion = (ArrayUnion *)uniform->values;
    int numComponents = numArgs/numElements;
    int elementSize = 0;

    switch(type) {
        case ARG_TYPE_DOUBLE:
            elementSize = numComponents * sizeof(double);
            break;
        case ARG_TYPE_UINT:
        case ARG_TYPE_FLOAT:
        case ARG_TYPE_INT:
            elementSize = numComponents * sizeof(uint32_t);
            break;
    };

    uniform->uniformName = name;
    uniform->numElements = numElements;
    uniform->elementSize = elementSize;

    va_list arguments;

    memset(arryUnion, 0, sizeof(ArrayUnion));

    va_start ( arguments, numArgs );
    for (int i = 0; i < numArgs; ++i) {
        switch(type) {
            case ARG_TYPE_INT:
                arryUnion->i[i] = va_arg(arguments, int32_t);
                break;
            case ARG_TYPE_FLOAT:
                arryUnion->f[i] = (float)va_arg(arguments, double);
                break;
            case ARG_TYPE_DOUBLE:
                arryUnion->d[i] = va_arg(arguments, double);
                break;
            case ARG_TYPE_UINT:
                arryUnion->u[i] = va_arg(arguments, uint32_t);
                break;
            default:
                assert(!"Invalid argument type.");
                break;
        };
    }
    va_end( arguments );

    return;
}

bool LWNShaderSpecialization::RunTest(int test_number, Program *pgm) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    FragmentShader fs(440);
    VertexShader vs(440);

    // Up to 5 uniforms can be specialized per test.
    GLSLCspecializationUniform uniform[5];

    ArrayUnion arrys[5];

    for (int i = 0; i < 5; ++i) {
        uniform[i].values = (void*)(&arrys[i]);
    }

    switch (test_number)
    {
        case 0:
            /* Test wherein an array of boolean values is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";
            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "    int   iCount;\n"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    bvec4 bv_metric[4];\n"
                "};\n"
                "void main() {\n"
                " if (bv_metric[0].x == true && bv_metric[1].x == false && bv_metric[2].x == true && bv_metric[3].x == true) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


                SetData(&uniform[0], "bv_metric", 4, ARG_TYPE_INT, 16, 1, 1, 1, 1,
                                                                       0, 1, 1, 1,
                                                                       1, 1, 1, 1,
                                                                       1, 1, 1, 1);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);

            break;
        case 1:
            /* Test wherein an array of double values is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block1 {\n"
                "    double  fArray[8];\n"
                "    int   iCount;\n"
                "};\n"
                "void main() {\n"
                " if (fArray[4] + fArray[5] + fArray[6] + fArray[7]  + float(iCount) == 11.0) {\n"
                "     fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "     fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "fArray", 8, ARG_TYPE_DOUBLE, 8,
                        0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0);

                SetData(&uniform[1], "iCount", 1, ARG_TYPE_INT, 1, 1);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
                g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 2:
            /* Test wherein an array of float values is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";
            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block1 {\n"
                "    float  fArray[8];\n"
                "    int   iCount;\n"
                "};\n"
                "void main() {\n"
                " if (fArray[4] + fArray[5] + fArray[6] + fArray[7]  + float(iCount) == 11.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


                SetData(&uniform[0], "fArray", 8, ARG_TYPE_FLOAT, 8,
                        0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0);
                SetData(&uniform[1], "iCount", 1, ARG_TYPE_INT, 1, 1);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
                g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 3:
            /* Test wherein an array of int values is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";
            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "    int   iCount;\n"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    int iArr[4];\n"
                "};\n"
                "void main() {\n"
                " if (float(iArr[0] + iArr[1] + iArr[2] + iArr[3]) + float(iCount) == 11.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "iArr", 4, ARG_TYPE_INT, 16,
                        1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0);
                SetData(&uniform[1], "iCount", 1, ARG_TYPE_INT, 1, 1);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
                g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 4:
            /* Test wherein an array of vec4 and an integer is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";
            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 1) uniform Block {\n"
                "    int   iCount;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block1 {\n"
                "    vec4 v_metric[4];\n"
                "};\n"
                "void main() {\n"
                " if (v_metric[0].x + v_metric[1].x + v_metric[2].x + v_metric[2].x  + float(iCount) == 5.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";
                SetData(&uniform[0], "v_metric", 4, ARG_TYPE_FLOAT, 16,
                        1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f);

                SetData(&uniform[1], "iCount", 1, ARG_TYPE_INT, 1, 1);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
                g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
    case 5:
            /* Test wherein an array of matrices, array of double and an integer is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(column_major, binding=0) uniform Block {\n"
                "    mat2 m2[2];"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    double  fArray[8];\n"
                "    int   iCount;\n"
                "};\n"
                "void main() {\n"
                " if (m2[0][1][0] + m2[1][0][1] + fArray[4] + fArray[5] + fArray[6] + fArray[7]  + float(iCount) == 16.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "m2", 2, ARG_TYPE_FLOAT, 16,
                        0.0f, 5.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 5.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f, 0.0f);

            SetData(&uniform[1], "fArray", 8, ARG_TYPE_DOUBLE, 8,
                    0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0);

            SetData(&uniform[2], "iCount", 1, ARG_TYPE_INT, 1, 1);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            break;
        case 6:
            /* Test wherein an array of matrices is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";
            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(row_major) uniform Block {\n"
                "    mat2 m2[2];"
                "};\n"
                "void main() {\n"
                " if (m2[0][1][0] + m2[1][0][1] == 5.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "m2", 2, ARG_TYPE_FLOAT, 16,
                        0.0f, 5.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 5.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f, 0.0f);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 7:
            /* Test wherein an array of matrices, array of double and an integer is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(row_major, binding=0) uniform Block {\n"
                "    mat2 m2[2];"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    double  fArray[8];\n"
                "    int   iCount;\n"
                "};\n"
                "void main() {\n"
                " if (m2[0][1][0] + m2[1][0][1] + fArray[4] + fArray[5] + fArray[6] + fArray[7]  + float(iCount) == 16.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "m2", 2, ARG_TYPE_FLOAT, 16,
                        0.0f, 5.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 5.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f, 0.0f);

            SetData(&uniform[1], "fArray", 8, ARG_TYPE_DOUBLE, 8,
                        0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0);

            SetData(&uniform[2], "iCount", 1, ARG_TYPE_INT, 1, 1);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            break;
        case 8:
            /* Test wherein a matrix mat2 is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";
            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(row_major) uniform Block {\n"
                "    mat2 m2;"
                "};\n"
                "void main() {\n"
                " if (m2[1][0] + m2[0][1] == 6.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "m2", 1, ARG_TYPE_FLOAT, 4,
                        0.0f, 5.0f,
                        1.0f, 0.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;

        case 9:
            /* Test wherein elements of 2 arrays of structures are specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"

                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   vec4 v4_metric;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S s1[2];"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    S2 s2[2];"
                "};\n"

                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if (s1[0].a + s1[0].b + s2[0].v4_metric.x + s2[0].v4_metric.y + s2[0].v4_metric.z == 80.0) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   vec4 v4_metric;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S s1[2];"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    S2 s2[2];"
                "};\n"
                "void main() {\n"
                " if (s1[0].a + s1[0].b + s2[0].v4_metric.x + s2[0].v4_metric.y + s2[0].v4_metric.z == 80.0) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = ocolor;\n"
                " }\n"
                "}\n";



            SetData(&uniform[0], "s1[0].a", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[1], "s1[0].b", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[2], "s2[0].v4_metric", 1, ARG_TYPE_FLOAT, 4,
                        10.0, 20.0, 30.0, 40.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            break;
        case 10:
            /* Test wherein arrays of vec4, integer are specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 1) uniform Block {\n"
                "    int   iCount;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block1 {\n"
                "    vec4 v_metric[4];\n"
                "};\n"
                "void main() {\n"
                " if (v_metric[0].x + v_metric[1].y + v_metric[2].z + v_metric[3].a  + float(iCount) == 11.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "v_metric", 4, ARG_TYPE_FLOAT, 16,
                    1.0f, 2.0f, 3.0f, 4.0f,
                    1.0f, 2.0f, 3.0f, 4.0f,
                    1.0f, 2.0f, 3.0f, 4.0f,
                    1.0f, 2.0f, 3.0f, 4.0f);
            SetData(&uniform[1], "iCount", 1, ARG_TYPE_INT, 1, 1);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 11:
            /* Test wherein arrays of vec4, integer are specialised. Swizzling is used on the elements of the vector */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 1) uniform Block {\n"
                "    int   iCount;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block1 {\n"
                "    vec4 v_metric[4];\n"
                "};\n"
                "void main() {\n"
                "vec2 v2;\n"
                "v2 = v_metric[0].xy + v_metric[1].yz + v_metric[2].xy + v_metric[1].xy;\n"
                " if (v2.x + v2.y + float(iCount) == 15.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "v_metric", 4, ARG_TYPE_FLOAT, 16,
                        1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f);
            SetData(&uniform[1], "iCount", 1, ARG_TYPE_INT, 1, 1);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 12:
            /* Test wherein a matrix mat2 within a UBO with a column_major layout is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, column_major) uniform Block {\n"
                "    mat2 m2;"
                "};\n"
                "void main() {\n"
                " if (m2[0][0] + m2[1][0] == 4.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "m2", 1, ARG_TYPE_FLOAT, 4,
                        1.0f, 2.0f,
                        3.0f, 4.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
      case 13:
            /* Test wherein a matrix mat3x2 within a UBO having a column_major layout is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(column_major) uniform Block {\n"
                "    mat3x2 m2;"
                "};\n"
                "void main() {\n"
                " if (m2[0][0] + m2[1][0] <= 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "m2", 1, ARG_TYPE_FLOAT, 6,
                        1.0f, 2.0f,
                        3.0f, 4.0,
                        3.0f, 4.0);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 14:
            /* Test wherein a matrix mat2 within a UBO having a row_major layout is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(row_major) uniform Block {\n"
                "    mat2 m2;"
                "};\n"
                "void main() {\n"
                " if (m2[1][0] + m2[0][1] == 6.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "m2", 1, ARG_TYPE_FLOAT, 4,
                        0.0f, 5.0f,
                        1.0f, 0.0f);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);

            break;
        case 15:
            /* Test wherein a matrix mat2 within a UBO having a row_major layout is specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(row_major) uniform Block {\n"
                "    mat3x2 m2;"
                "};\n"
                "void main() {\n"
                " if (m2[0][0] + m2[1][0] == 3.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "m2", 1, ARG_TYPE_FLOAT, 8,
                            1.0f, 2.0f, 3.0f, 0.0f,
                            4.0f, 5.0, 6.0, 0.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
         case 16:
            /* Test wherein a matrix within a UBO having a row_major layout and another matrix within a UBO having column_major layout are specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(column_major) uniform Block {\n"
                "    mat2 m2;"
                "};\n"

                "layout(row_major) uniform Block1 {\n"
                "    mat2 m2_1;"
                "};\n"
                "void main() {\n"
                " if (m2[0][0] + m2_1[1][0] <= 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "m2", 1, ARG_TYPE_FLOAT, 4,
                                 1.0f, 2.0f,
                                 3.0f, 4.0);
            SetData(&uniform[1], "m2_1", 1, ARG_TYPE_FLOAT, 4,
                                 1.0f, 2.0f,
                                 3.0f, 4.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 17:
            /* Test wherein a matrix within a UBO having a row_major layout and another matrix within a UBO having column_major layout
               , elements in other layouts (shared, std140) are specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(column_major) uniform BlockA {\n"
                "    mat2 m2;"
                "};\n"

                "layout(row_major) uniform BlockB {\n"
                "    mat2 m2_1;"
                "};\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 scale;\n"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 scale1;\n"
                "};\n"
                "void main() {\n"
                " if (m2[0][0] + m2_1[1][0] <= 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else if(f_scalar + f_scalar1 <= 1.0) {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "m2", 1, ARG_TYPE_FLOAT, 4,
                                 1.0f, 2.0f,
                                 3.0f, 4.0);

            SetData(&uniform[1], "m2_1", 1, ARG_TYPE_FLOAT, 4,
                                 1.0f, 2.0f,
                                 3.0f, 4.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 18:
            /* Test wherein a matrix within a UBO having a row_major layout and another matrix within a UBO having column_major layout
               , elements in other layouts (packed, std140) are specialised */

            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(column_major) uniform BlockA {\n"
                "    mat2 m2;"
                "};\n"

                "layout(row_major) uniform BlockB {\n"
                "    mat2 m2_1;"
                "};\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 scale;\n"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 scale1;\n"
                "};\n"
                "void main() {\n"
                " if (m2[0][0] + m2_1[1][0] <= 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else if(f_scalar + f_scalar1 <= 1.0) {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "m2", 1, ARG_TYPE_FLOAT, 4,
                         1.0f, 2.0f,
                         3.0f, 4.0f);

            SetData(&uniform[1], "m2_1", 1, ARG_TYPE_FLOAT, 4,
                         1.0f, 2.0f,
                         3.0f, 4.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 19:
            /* Test wherein a matrix within a UBO having a row_major layout and another matrix within a UBO having column_major layout
               , elements in other layouts (packed, std140) in different bindings  are specialised */

            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(column_major, binding=0) uniform BlockA {\n"
                "    mat2 m2;"
                "};\n"

                "layout(row_major, binding=1) uniform BlockB {\n"
                "    mat2 m2_1;"
                "};\n"
                "layout(packed, binding=2) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 scale;\n"
                "};\n"
                "layout(std140, binding=3) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 scale1;\n"
                "};\n"
                "void main() {\n"
                " if (m2[0][0] + m2_1[1][0] <= 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else if(f_scalar + f_scalar1 <= 1.0) {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "m2", 1, ARG_TYPE_FLOAT, 4,
                         1.0f, 2.0f,
                         3.0f, 4.0);
            SetData(&uniform[1], "m2_1", 1, ARG_TYPE_FLOAT, 4,
                         1.0f, 2.0f,
                         3.0f, 4.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 20:
            /* Test wherein elements in different layouts (packed, std140, shared) and different bindings  are specialised */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";
            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 scale;\n"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 scale1;\n"
                "};\n"
                "layout(std140, binding = 2) uniform Block2 {\n"
                "    float f_scalar2;"
                "    vec4 scale2;\n"
                "};\n"
                "void main() {\n"
                " if (f_scalar + f_scalar1 == 3.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 1.0f);
            SetData(&uniform[1], "f_scalar1", 1, ARG_TYPE_FLOAT, 1, 2.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 21:
            /* Test wherein  a float, array of vec4 in a UBO (Block) and shared between fragment and vertex shader are specialised
               The other elements in UBO (Block1) are not shared with vertex shader and also specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 v_metric[4];\n"
                "};\n"

                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                " if (f_scalar +  v_metric[2].x == 11.0) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 v_metric[4];\n"
                "};\n"
                "layout(shared, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 v_metric1[4];\n"
                "};\n"
                "void main() {\n"
                " if (f_scalar + f_scalar1 + v_metric[1].x + v_metric1[2].x + v_metric[2].x == 23.0) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[1], "f_scalar1", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[2], "v_metric", 4, ARG_TYPE_FLOAT, 16,
                          1.0, 2.0, 3.0, 4.0,
                          1.0, 2.0, 3.0, 4.0,
                          1.0, 2.0, 3.0, 4.0,
                          1.0, 2.0, 3.0, 4.0);
            SetData(&uniform[3], "v_metric1", 4, ARG_TYPE_FLOAT, 16,
                          1.0, 2.0, 3.0, 4.0,
                          1.0, 2.0, 3.0, 4.0,
                          1.0, 2.0, 3.0, 4.0,
                          1.0, 2.0, 3.0, 4.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            break;
        case 22:
            /* Test wherein matrices in 2 UBOs having column_major, row_major are shared between
                vertex and fragment shader also specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "layout(column_major) uniform Block {\n"
                "    mat2 m2;"
                "};\n"

                "layout(row_major) uniform Block1 {\n"
                "    mat2 m2_1;"
                "};\n"

                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"



                " if ( (m2_1[0][0] == 5.0)  && (m2_1[0][1] == 7.0) && (m2_1[1][0] == 6.0) && (m2_1[1][1] == 8.0) && (m2[0][0] == 1.0)  && (m2[0][1] == 2.0) && (m2[1][0] == 3.0) && (m2[1][1] == 4.0)) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(0.0, 0.0, 1.0, 1.0);\n"
                " }\n"
                "}\n";
           fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                 "layout(column_major) uniform Block {\n"
                "    mat2 m2;"
                "};\n"

                "layout(row_major) uniform Block1 {\n"
                "    mat2 m2_1;"
                "};\n"

                "void main() {\n"



                " if ( (m2_1[0][0] == 5.0)  && (m2_1[0][1] == 7.0) && (m2_1[1][0] == 6.0) && (m2_1[1][1] == 8.0) && (m2[0][0] == 1.0)  && (m2[0][1] == 2.0) && (m2[1][0] == 3.0) && (m2[1][1] == 4.0)) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(0.0, 0.0, 1.0, 0.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "m2", 1, ARG_TYPE_FLOAT, 4,
                         1.0f, 2.0,
                         3.0f, 4.0);

            SetData(&uniform[1], "m2_1", 1, ARG_TYPE_FLOAT, 4,
                         5.0, 6.0,
                         7.0, 8.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 23:
           /* Test wherein float, array of vec4 in 2 UBOs  are shared between
                vertex and fragment shader and all also specialised*/
           vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 v_metric[4];\n"
                "};\n"
                "layout(shared, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 v_metric1[4];\n"
                "};\n"
                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if (f_scalar + f_scalar1  == 20.0) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 v_metric[4];\n"
                "};\n"
                "layout(shared, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 v_metric1[4];\n"
                "};\n"
                "void main() {\n"
                " if (f_scalar + f_scalar1 + v_metric[1].x + v_metric1[2].x + v_metric[2].x == 23.0) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);
                SetData(&uniform[1], "f_scalar1", 1, ARG_TYPE_FLOAT, 1, 10.0f);
                SetData(&uniform[2], "v_metric", 4, ARG_TYPE_FLOAT, 16,
                          1.0, 2.0, 3.0, 4.0,
                          1.0, 2.0, 3.0, 4.0,
                          1.0, 2.0, 3.0, 4.0,
                          1.0, 2.0, 3.0, 4.0);
                SetData(&uniform[3], "v_metric1", 4, ARG_TYPE_FLOAT, 16,
                          1.0, 2.0, 3.0, 4.0,
                          1.0, 2.0, 3.0, 4.0,
                          1.0, 2.0, 3.0, 4.0,
                          1.0, 2.0, 3.0, 4.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            break;
        case 24:
            /* Test wherein structures  in 2 UBOs  are shared between
                vertex and fragment shader and all member elements of the structures also specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   int c;\n"
                "   int d;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S s1[2];"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    S2 s2[2];"
                "};\n"
                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if (s1[0].a == 10.0 &&  s1[0].b == 20.0 &&  s2[0].c == 30 &&  s2[0].d == 40) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   int c;\n"
                "   int d;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S s1[2];"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    S2 s2[2];"
                "};\n"
                "void main() {\n"
                " if (s1[0].a == 10.0 &&  s1[0].b == 20.0 &&  s2[0].c == 30 &&  s2[0].d == 40) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s1[0].a", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[1], "s1[0].b", 1, ARG_TYPE_FLOAT, 1, 20.0f);
            SetData(&uniform[2], "s2[0].c", 1, ARG_TYPE_INT, 1, 30);
            SetData(&uniform[3], "s2[0].d", 1, ARG_TYPE_INT, 1, 40);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            break;

        case 25:
            /* Test wherein matrices (m2, m2_1)  in 2 UBOs  are shared between
                vertex and fragment shader and all the matrices also specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "layout(column_major) uniform Block {\n"
                "    mat2 m2;"
                "};\n"

                "layout(row_major) uniform Block1 {\n"
                "    mat2 m2_1;"
                "};\n"

                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if (m2[0][0] + m2_1[1][0] <= 10.0) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                 "layout(column_major) uniform Block {\n"
                "    mat2 m2;"
                "};\n"

                "layout(row_major) uniform Block1 {\n"
                "    mat2 m2_1;"
                "};\n"

                "void main() {\n"
                " if (m2[0][0] + m2_1[1][0] <= 10.0) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "m2", 1, ARG_TYPE_FLOAT, 4,
                     1.0f, 2.0f,
                     3.0f, 4.0);

            SetData(&uniform[1], "m2_1", 1, ARG_TYPE_FLOAT, 4,
                    1.0f, 2.0f,
                    3.0f, 4.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;

        case 26:
            /* Test wherein structures  in 2 UBOs  are shared between
                vertex and fragment shader and all member elements of the structures also specialised*/
            vs <<
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec3 color;\n"
            "out vec4 ocolor;\n"
            "struct S\n"
            "{\n"
            "   int   a;\n"
            "   float b;\n"
            "};\n"
            "struct S2\n"
            "{\n"
            "   vec4 v4_metric;\n"
            "};\n"
            "layout(std140, binding = 0) uniform Block {\n"
            "    S s1[2];"
            "};\n"
            "layout(packed, binding = 1) uniform Block1 {\n"
            "    S2 s2[2];"
            "};\n"
            "void main() {\n"
            " gl_Position = vec4(position, 1.0);\n"
            " if (s1[0].a + s1[0].b + s2[0].v4_metric.x + s2[0].v4_metric.y + s2[0].v4_metric.z == 80.0) {\n"
            "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
            " } else {\n"
            "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            " }\n"
            "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   vec4 v4_metric;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S s1[2];"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    S2 s2[2];"
                "};\n"
                "void main() {\n"
                " if (s1[0].a + s1[0].b + s2[0].v4_metric.x + s2[0].v4_metric.y + s2[0].v4_metric.z == 80.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";
            SetData(&uniform[0], "s1[0].a", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[1], "s1[0].b", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[2], "s2[0].v4_metric", 1, ARG_TYPE_FLOAT, 4,
                    10.0, 20.0, 30.0, 40.0 );

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            break;
        case 27:
            /* Test wherein float , array of vectors (vec4)  in 2 UBOs  are shared between
                vertex and fragment shader and all member elements of the UBOs also specialised
                Swizzle operation is performed on elements of the array of vectors */

            vs  <<
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec3 color;\n"
            "out vec4 ocolor;\n"
            "layout(shared, binding = 0) uniform Block {\n"
            "    float f_scalar;"
            "    vec4 v_metric[4];\n"
            "};\n"
            "layout(shared, binding = 1) uniform Block1 {\n"
            "    float f_scalar1;"
            "    vec4 v_metric1[4];\n"
            "};\n"
            "void main() {\n"
            " gl_Position = vec4(position, 1.0);\n"
            " if (f_scalar + f_scalar1  == 20.0) {\n"
            "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
            " }\n"
            "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 v_metric[4];\n"
                "};\n"
                "layout(shared, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 v_metric1[4];\n"
                "};\n"
                "void main() {\n"
                "vec2 v2;\n"
                "v2 = v_metric[1].xy + v_metric1[2].yz + v_metric[2].zw; \n"
                " if (f_scalar + v2.x + v2.y == 25.0) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[1], "f_scalar1", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[2], "v_metric", 4, ARG_TYPE_FLOAT, 16,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0);
            SetData(&uniform[3], "v_metric1", 4, ARG_TYPE_FLOAT, 16,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            break;
        case 28:
            /* Test wherein float , array of vectors (vec4)  in 2 UBOs  are not shared between
                vertex and fragment shader and all member elements of the UBOs also specialised*/
             vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 v_metric[4];\n"
                "};\n"

                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if (f_scalar + v_metric[0].x + v_metric[1].x + v_metric[2].x + v_metric[3].x  == 14.0) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"

                "layout(shared, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 v_metric1[4];\n"
                "};\n"
                "void main() {\n"
                " if (f_scalar1 + v_metric1[0].x + v_metric1[1].x + v_metric1[2].x + v_metric1[3].x  == 14.0) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = ocolor;\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[1], "f_scalar1", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[2], "v_metric", 4, ARG_TYPE_FLOAT, 16,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0);
            SetData(&uniform[3], "v_metric1", 4, ARG_TYPE_FLOAT, 16,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            break;

        case 29:
            /* Test wherein matrix (m2_1) is  shared between vertex and fragment shader
               Matrix m2 is not shared with vertex shader and all member elements of the UBOs also specialised
            */
            vs <<
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec3 color;\n"
            "out vec4 ocolor;\n"

            "layout(row_major) uniform Block1 {\n"
            "    mat2 m2_1;"
            "};\n"

            "void main() {\n"
            " gl_Position = vec4(position, 1.0);\n"
            " if (m2_1[0][0] + m2_1[1][0] <= 10.0) {\n"
            "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
            " } else {\n"
            "      ocolor = vec4(0.0, 0.0, 1.0, 1.0);\n"
            " }\n"
            "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                 "layout(column_major) uniform Block {\n"
                "    mat2 m2;"
                "};\n"

                "layout(row_major) uniform Block1 {\n"
                "    mat2 m2_1;"
                "};\n"

                "void main() {\n"
                " if (m2[0][0] + m2_1[1][0] <= 10.0) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(0.0, 0.0, 1.0, 0.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "m2", 1, ARG_TYPE_FLOAT, 4,
                    1.0f, 2.0f,
                    3.0f, 4.0);
            SetData(&uniform[1], "m2_1", 1, ARG_TYPE_FLOAT, 4,
                    1.0f, 2.0f,
                    3.0f, 4.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;

        case 30:
            /* Test wherein 1 structures  in  an UBO is  shared between vertex and fragment shader
               The other structure in UBO is not shared with vertex shader
               All member elements of the structures are specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   vec4 v4_metric;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S s1[2];"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    S2 s2[2];"
                "};\n"
                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if (s1[0].a + s1[0].b + s2[0].v4_metric.x + s2[0].v4_metric.y + s2[0].v4_metric.z == 80.0) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"

                "struct S2\n"
                "{\n"
                "   vec4 v4_metric;\n"
                "};\n"

                "layout(packed, binding = 1) uniform Block1 {\n"
                "    S2 s2[2];"
                "};\n"
                "void main() {\n"
                " if (s2[0].v4_metric.x + s2[0].v4_metric.y + s2[0].v4_metric.z == 60.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s1[0].a", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[1], "s1[0].b", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[2], "s2[0].v4_metric", 1, ARG_TYPE_FLOAT, 4,                                10.0, 20.0, 30.0, 40.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            break;
        case 31:
            /* Test wherein array of vectors (vec4), float variables in an UBO is  shared between vertex and fragment shader
               The other UBO (Block1) with members as array of vectors (vec4), float variable  is not shared with vertex shader.
               Swizzle operation if performed on the individual elements of the array of vectors
               All member elements of the 2 UBOs are specialised*/
            vs <<
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec3 color;\n"
            "out vec4 ocolor;\n"
            "layout(shared, binding = 0) uniform Block {\n"
            "    float f_scalar;"
            "    vec4 v_metric[4];\n"
            "};\n"

            "void main() {\n"
            " vec2 v2;\n"
            " gl_Position = vec4(position, 1.0);\n"
            " v2 = v_metric[1].xy + v_metric[2].yz + v_metric[2].zw; \n"
            " if (v2.x + v2.y + f_scalar == 25.0) {\n"
            "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
            " }  else { \n"
            "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            " }\n"
            "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 v_metric[4];\n"
                "};\n"
                "layout(shared, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 v_metric1[4];\n"
                "};\n"
                "void main() {\n"
                " vec2 v2;\n"
                " v2 = v_metric[1].xy + v_metric1[2].yz + v_metric[2].zw; \n"
                " if (f_scalar + v2.x + v2.y == 25.0) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = ocolor;\n"
                " }\n"
                "}\n";



            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[1], "f_scalar1", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[2], "v_metric", 4, ARG_TYPE_FLOAT, 16,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0);
            SetData(&uniform[3], "v_metric1", 4, ARG_TYPE_FLOAT, 16,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0,
                  1.0, 2.0, 3.0, 4.0);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
        break;
        case 32:
            /* Test wherein vector (vec4), float variables in 2 different UBO with layouts (shared, packed)are specialised*/

            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 scale;\n"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 scale1;\n"
                "};\n"
                "void main() {\n"
                " if (f_scalar <= 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 2.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 33:
            /* Test wherein vector (vec4), float variables in 3 different UBO with layouts (shared, packed, std140) are specialised*/

            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 scale;\n"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 scale1;\n"
                "};\n"
                "void main() {\n"
                " if (f_scalar <= 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 2.0f);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 34:
            /* Test wherein vector (vec4), float variables in 2 different UBO with layouts (shared, packed) are specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 scale;\n"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 scale1;\n"
                "};\n"
                "void main() {\n"
                " if (f_scalar <= 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 2.0f);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 35:
            /* Test wherein vector (vec4), float variables in 2 different UBO with layouts (shared, packed) are specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 scale;\n"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 scale1;\n"
                "    int iCount;\n"
                "};\n"
                "void main() {\n"
                " if (f_scalar + float(iCount) == 3.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 2.0f);
            SetData(&uniform[1], "iCount", 1, ARG_TYPE_INT, 1, 1);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;

        case 36:
            /* Test wherein float variable in single UBO with  shared layouts is specialised*/

            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "};\n"
                "void main() {\n"
                " if (f_scalar == 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 37:
            /* Test wherein float variable in single UBO with  shared layouts is specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "};\n"
                "void main() {\n"
                " if (f_scalar == 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;

        case 38:
            /* Test wherein float variable in single UBO with  shared layouts is specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 1) uniform Block {\n"
                "    float f_scalar;"
                "};\n"
                "void main() {\n"
                " if (f_scalar == 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;

        case 39:
            /* Test wherein float variable in UBOs with different bindings is shared between fragment and vertex shader and the float is specialised*/

            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "layout(shared, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "};\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                " if (f_scalar == 10.0) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 3) uniform Block {\n"
                "    float f_scalar;"
                "};\n"
                "void main() {\n"
                " if (f_scalar == 10.0) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;

        case 40:
            /* Test wherein float variable in UBOs with same bindings (binding=0) is shared between fragment and vertex shader and the float is specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "layout(shared, binding = 1) uniform BlockA {\n"
                "    float f_scalar;"
                "};\n"
                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if (f_scalar == 10.0) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(shared, binding = 1) uniform BlockA {\n"
                "    float f_scalar;"
                "};\n"
                "void main() {\n"
                " if (f_scalar == 10.0) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 41:
            /* Test wherein float variable in UBO with layout as "std140" is specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "};\n"
                "void main() {\n"
                " if (f_scalar == 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.f);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;

        case 42:
            /* Test wherein float variable, array of vectors (vec4) in UBO with layout as "std140" are specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4  iv4_metric;"
                "};\n"
                "void main() {\n"
                " if (f_scalar + iv4_metric.x == 20.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);

            SetData(&uniform[1], "iv4_metric", 1, ARG_TYPE_FLOAT, 4, 10.0, 0.0f, 0.0f, 0.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 43:
            /* Test wherein different datatypes (float,vec4,int,bool) in 2 UBOs with layout as "std140", "packed" are specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4  iv4_metric;"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    int iCount;"
                "    bool b_status;"
                "};\n"
                "void main() {\n"
                " if (f_scalar + iv4_metric.x + float(iCount) == 30.0 && b_status == true) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);

            SetData(&uniform[1], "iv4_metric", 1, ARG_TYPE_FLOAT, 4,
                    10.0, 20.0, 30.0, 40.0);
            SetData(&uniform[2], "iCount", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[3], "b_status", 1, ARG_TYPE_INT, 1, 1);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            break;
        case 44:
            /* Test wherein different datatypes (float,vec4,int,bool, double) in 2 UBOs with layout as "std140", "packed" are specialised*/

            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4  iv4_metric;"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    int iCount;"
                "    bool b_status;"
                "    double d_count;"
                "};\n"
                "void main() {\n"
                " if (f_scalar + iv4_metric.y + float(iCount) + d_count == 50.0 && b_status == true) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);

            SetData(&uniform[1], "iv4_metric", 1, ARG_TYPE_FLOAT, 4,
                    10.0, 20.0, 30.0, 40.0);
            SetData(&uniform[2], "iCount", 1, ARG_TYPE_INT, 1, 10);

            SetData(&uniform[3], "b_status", 1, ARG_TYPE_INT, 1, 1);

            SetData(&uniform[4], "d_count", 1, ARG_TYPE_DOUBLE, 1, 10.0);


            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[4]);
            break;
        case 45:
            /* Test wherein different datatypes (float,vec4,int,bool, double) in 2 UBOs with layout as "std140", "packed" are specialised*/

            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4  iv4_metric;"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    int iCount;"
                "    bool b_status;"
                "};\n"
                "void main() {\n"
                " if (f_scalar + iv4_metric.x + float(iCount) == 30.0 && b_status == true) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[1], "iv4_metric", 1, ARG_TYPE_FLOAT, 4, 10.0f, 0.0f, 0.0f, 0.0f);
            SetData(&uniform[2], "iCount", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[3], "b_status", 1, ARG_TYPE_INT, 1, 1);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            break;
        case 46:
            /* Test wherein different datatypes (float,vec4,int) in 2 UBOs with layout as "std140", "packed" are specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4  iv4_metric;"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    int iCount;"
                "};\n"
                "void main() {\n"
                " if (f_scalar + iv4_metric.x + float(iCount) == 30.0 ) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[1], "iv4_metric", 1, ARG_TYPE_FLOAT, 4, 10.0, 0.0f, 0.0f, 0.0f);

            SetData(&uniform[2], "iCount", 1, ARG_TYPE_INT, 1, 10);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            break;
        case 47:
            /* Test wherein different datatypes (float,vec4,int) in 2 UBOs with layout as "std140", "packed" are specialised
               Sizzle operations are perfromed on the elements of the vector vec4*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4  iv4_metric;"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    int iCount;"
                "};\n"
                "void main() {\n"
                " vec2 v2;\n"
                " v2 = iv4_metric.xy + iv4_metric.yz + iv4_metric.zw + iv4_metric.zx ; \n"
                " if (v2.x + v2.y + float(iCount) == 200.0 ) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, iCount, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[1], "iv4_metric", 1, ARG_TYPE_FLOAT, 4, 10.0f, 20.0f, 30.0f, 40.0f);

            SetData(&uniform[2], "iCount", 1, ARG_TYPE_INT, 1, 10);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            break;
        case 48:
            /* Test wherein different datatypes (float,vec4,int) in 2 UBOs with layout as "std140", "packed" are specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4 scale;\n"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    float f_scalar1;"
                "    vec4 scale1;\n"
                "};\n"
                "void main() {\n"
                " if (f_scalar == 10.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 49:
            /* Test wherein different datatypes (float,vec4,vec3) in 2 UBOs with layout as "std140", "packed" are specialised*/

            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    float f_scalar;"
                "    vec4  iv4_metric;"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    vec3 fv3_metric;"
                "};\n"
                "void main() {\n"
                " vec2 v2;\n"
                " v2 = iv4_metric.xy + fv3_metric.yz + iv4_metric.zw + fv3_metric.xy ; \n"
                " if (v2.x  + v2.y == 180.0 ) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[1], "iv4_metric", 1, ARG_TYPE_FLOAT, 4, 10.0f, 20.0f, 30.0f, 40.0f);
            SetData(&uniform[2], "fv3_metric", 1, ARG_TYPE_FLOAT, 3, 10.0f, 20.0f, 30.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            break;
        case 50:
            /* Test wherein structure having the members as (vec3, uvec4, mat3x2) in a single UBOs are specialised*/

            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "struct S\n"
                "{\n"
                "   vec3   v3;\n"
                "   uvec4  uv4;\n"
                "   mat3x2 m32; \n "
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S s1;\n"
                "};\n"
                "void main() {\n"
                " if ((s1.v3.x + s1.v3.y + s1.v3.z == 60.0) &&  "
                "  (s1.v3.x + s1.v3.y + s1.v3.z + s1.uv4.x + s1.uv4.y + s1.uv4.z + s1.uv4.w == 280) &&  "
                "  (s1.m32[0][0]==10.0 && s1.m32[0][1]==20.0)) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(0.0, 0.0, 1.0, 0.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s1.v3", 1, ARG_TYPE_FLOAT, 3, 10.0f, 20.0f, 30.0f);
            SetData(&uniform[1], "s1.uv4", 1, ARG_TYPE_UINT, 4, 40, 50, 60, 70);
            SetData(&uniform[2], "s1.m32", 1, ARG_TYPE_FLOAT, 12,
                                     10.0, 20.0, 0.0f, 0.0f,
                                     30.0, 40.0, 0.0f, 0.0f,
                                     50.0, 60.0, 0.0f, 0.0f);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            break;

        case 51:
            /* Test wherein 2 structure in 2 UBOs are used in fragment shader are specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   vec4 v4_metric;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S s1;"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    S2 s2;"
                "};\n"
                "void main() {\n"
                " if (s1.a + s1.b + s2.v4_metric.x + s2.v4_metric.y + s2.v4_metric.z == 80.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";


            SetData(&uniform[0], "s1.a", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[1], "s1.b", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[2], "s2.v4_metric", 1, ARG_TYPE_FLOAT, 4, 10.0f, 20.0f, 30.0f, 40.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            break;
        case 52:
            /* Test wherein 2 structure in 2 UBOs are shared between fragment and vertex shader are specialised*/

            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"

                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   vec4 v4_metric;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S s1;"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    S2 s2;"
                "};\n"

                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if (s1.a + s1.b + s2.v4_metric.x + s2.v4_metric.y + s2.v4_metric.z == 80.0) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   vec4 v4_metric;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S s1;"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    S2 s2;"
                "};\n"
                "void main() {\n"
                " if (s1.a + s1.b + s2.v4_metric.x + s2.v4_metric.y + s2.v4_metric.z == 80.0) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = ocolor;\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s1.a", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[1], "s1.b", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[2], "s2.v4_metric", 1, ARG_TYPE_FLOAT, 4, 10.0f, 20.0f, 30.0f, 40.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            break;
        case 53:
            /* Test wherein 2 structure in 2 UBOs are used in fragment shader are specialised*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   vec4 v4_metric;\n"
                "   int   iCount;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S s1;"
                "};\n"
                "layout(packed, binding = 1) uniform Block1 {\n"
                "    S2 s2;"
                "};\n"
                "void main() {\n"
                " if (s1.a + s1.b + s2.v4_metric.x + s2.v4_metric.y + s2.v4_metric.z + s2.iCount == 81.00) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s1.a", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[1], "s1.b", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[2], "s2.v4_metric", 1, ARG_TYPE_FLOAT, 4, 10.0f, 20.0f, 30.0f, 40.0f);
            SetData(&uniform[3], "s2.iCount", 1, ARG_TYPE_INT, 1, 1);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            break;

        case 54:
            /* Test wherein 2 structure in 2 UBOs are used in fragment shader are specialised*/

            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";

            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S s1;"
                "};\n"
                "void main() {\n"
                " if (s1.a + s1.b == 20.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s1.a", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[1], "s1.b", 1, ARG_TYPE_FLOAT, 1, 10.0f);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 55:
            /* Test case 55 : Cross-stage optimisation : The vertex shader attribute ocolor usage is guarded by a condition that is never true */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = color;\n"
                "}\n";
            fs <<
                "in vec3 ocolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "    int   iCount;\n"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    bvec4 bv_metric[4];\n"
                "};\n"
                "void main() {\n"
                " if (bv_metric[0].x == true && bv_metric[1].x == false && bv_metric[2].x == true && bv_metric[3].x == true) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "bv_metric", 4, ARG_TYPE_INT, 16, 1, 1, 1, 1,
                                                                       0, 1, 1, 1,
                                                                       1, 1, 1, 1,
                                                                       1, 1, 1, 1);
                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 56:
            /* Test case 56 : Cross-stage optimisation : The vertex shader attribute ocolor, bgcolor usage is guarded by a condition that is never true */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "out vec4 bgcolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor  = color;\n"
                "  bgcolor = vec4(color,1);\n"
                "}\n";
            fs <<
                "in vec3 ocolor;\n"
                "in vec4 bgcolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "    int   iCount;\n"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    bvec4 bv_metric[4];\n"
                "};\n"
                "void main() {\n"
                " if (bv_metric[0].x == true && bv_metric[1].x == false && bv_metric[2].x == true && bv_metric[3].x == true) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "bv_metric", 4, ARG_TYPE_INT, 16, 1, 1, 1, 1,
                                                                       0, 1, 1, 1,
                                                                       1, 1, 1, 1,
                                                                       1, 1, 1, 1);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 57:
            /* Test case 57 : Cross-stage optimisation :  The vertex shader attribute bgcolor is used and ocolor usage is guarded by a condition that is never true */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "out vec4 bgcolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor  = color;\n"
                "  bgcolor = vec4(0.625, 0.75, 0.8125, 0.9375);\n"
                "}\n";
            fs <<
                "in vec3 ocolor;\n"
                "in vec4 bgcolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "    int   iCount;\n"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    bvec4 bv_metric[4];\n"
                "};\n"
                "void main() {\n"
                " if (bv_metric[0].x == true && bv_metric[1].x == false && bv_metric[2].x == true && bv_metric[3].x == true && \
                      bgcolor.x == 0.625 && bgcolor.y == 0.75 && bgcolor.z == 0.8125 && bgcolor.w == 0.9375) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "bv_metric", 4, ARG_TYPE_INT, 16, 1, 1, 1, 1,
                                                                       0, 1, 1, 1,
                                                                       1, 1, 1, 1,
                                                                       1, 1, 1, 1);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 58:
            /* Test case 58 : Cross-stage optimisation : Vertex shader attribute ocolor usage is guarded by a condition that is never true */
            vs <<
                 "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "out vec4 bgcolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor  = vec3(1, 0, 0);\n" // Red color
                "  bgcolor = vec4(0.25, 0.375, 0.4375, 0);\n"
                "}\n";
            fs <<
                "in vec3 ocolor;\n"
                "in vec4 bgcolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "    int   iCount;\n"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    bvec4 bv_metric[4];\n"
                "};\n"
                "void main() {\n"
                " if (bv_metric[0].x == true && bv_metric[1].x == false && bv_metric[2].x == true && bv_metric[3].x == true) {\n" //This condition is satisfied
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(ocolor.xyz, 1.0);\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "bv_metric", 4, ARG_TYPE_INT, 16, 1, 1, 1, 1,
                                                                       0, 1, 1, 1,
                                                                       1, 1, 1, 1,
                                                                       1, 1, 1, 1);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 59:
            /* Test case 59 : Cross-stage optimisation :  Vertex shader attribute bgcolor is used
                             Other vertex shader attribute ocolor usage is guarded by a condition that is never true
                             Type: Some vertex shader attributes are unused*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec3 ocolor;\n"
                "out vec3 bgcolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor  = vec3(0.5, 0.25, 0.75);\n"
                "  bgcolor = vec3(0.75, 0.8125, 0.9375);\n" // Green color
                "}\n";
            fs <<
                "in vec3 ocolor;\n"
                "in vec3 bgcolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "    int   iCount;\n"
                 "   float f_scalar;"
                "};\n"
                "void main() {\n"
                " if (iCount == 5 && f_scalar == 1.0) {\n"
                "    if (bgcolor.x == 0.75 && bgcolor.y == 0.8125 && bgcolor.z == 0.9375) {\n"
                "      fcolor = vec4(0, 1, 0, 1);\n"
                "    } else {\n"
                "      fcolor = vec4(1, 0, 0, 1);\n"
                "    } \n"
                " } else {\n"
                "      fcolor = vec4(1, 0, 0, 1);\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 1.0f);

                SetData(&uniform[1], "iCount", 1, ARG_TYPE_INT, 1, 5);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
                g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 60:
            /* Test case 60 : Cross-stage optimisation :Vertex shader attribute ocolor, bgcolor usage is guarded by a condition that is never true
                             All vertex shader attributes (ocolor, bgcolor) are unused in fragment shader
                             Type: All vertex shader attributes are unused*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec2 ocolor;\n"
                "out vec2 bgcolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor  = vec2(0.5, 0.9375);\n" // Red color
                "  bgcolor = vec2(0.75, 0.8125);\n"
                "}\n";
            fs <<
                "in vec2 ocolor;\n"
                "in vec2 bgcolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "    int   iCount;\n"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    bvec4 bv_metric[4];\n"
                "};\n"
                "void main() {\n"
                " if (bv_metric[0].x == true && bv_metric[1].x == false && bv_metric[2].x == true && bv_metric[3].x == false) {\n"
                "      fcolor = vec4(0, 1, 0, 1);\n"
                " } else {\n"
                "    if (bgcolor.x == 0.75 && bgcolor.y == 0.8125 && ocolor.x == 0.5 && ocolor.y == 0.9375) {\n"
                "      fcolor = vec4(1, 0, 0, 1);\n"
                "    } else {\n"
                "      fcolor = vec4(0.8, 0, 0, 1);\n"
                "    }\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "bv_metric", 4, ARG_TYPE_INT, 16, 1, 1, 1, 1,
                                                                       0, 1, 1, 1,
                                                                       1, 1, 1, 1,
                                                                       0, 1, 1, 1);


                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            break;
        case 61:
            /* Test case 61 : Cross-stage optimisation :  Vertex shader attribute "ocolor" components .xz are used and .yw are unused , bgcolor usage is unused
                             Type: Some vertex shader attribute components are used and some are unused*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor = vec4(0.75, 0.8125, 0.9375, 0.5);\n"
                "}\n";
            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"

                "layout(packed, binding = 0) uniform Block {\n"
                "   int   iCount;\n"
                "   float f_scalar;"
                "};\n"

                "void main() {\n"
                " if (iCount == 5 && f_scalar == 1.0) {\n"
                "    if (ocolor.x == 0.75 && ocolor.z == 0.9375) {\n"
                "      fcolor = vec4(0, 1, 0, 1);\n"
                "    }\n"
                " } else {\n"
                "    if (ocolor.y == 0.8125 && ocolor.w == 0.5) {\n"
                "      fcolor = vec4(1, 0, 0, 1);\n"
                "    } else {\n"
                "      fcolor = vec4(0.8, 0, 0, 1);\n"
                "    }\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 1.0f);

                SetData(&uniform[1], "iCount", 1, ARG_TYPE_INT, 1, 5);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
                g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 62:
            /* Test case 62 : Cross-stage optimisation : Vertex shader attribute "ocolor" components .xyz are used and .w is unused
                             Vertex shader attribute  "bgcolor" components .w is used and .xyz are unused
                             Type: Some vertex shader attribute components are used and some are unused*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "out vec4 bgcolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor  = vec4(0.25, 0.375, 0.4375, 0.5);\n"
                "  bgcolor = vec4(0.625, 0.75, 0.8125, 0.9375);\n"
                "}\n";
            fs <<
                "in vec4 ocolor;\n"
                "in vec4 bgcolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "   int   iCount;\n"
                "   float f_scalar;"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    bvec4 bv_metric[4];\n"
                "};\n"
                "void main() {\n"
                " if (iCount == 5 && f_scalar == 1.0) {\n"
                "    if (ocolor.x == 0.25 && ocolor.y == 0.375 && ocolor.z == 0.4375 && bgcolor.w == 0.9375) {\n"
                "      fcolor = vec4(0, 1, 0, 1);\n"
                "    } else {\n"
                "      fcolor = vec4(0.8, 0, 0, 1);\n"
                "    }\n"
                " } else {\n"
                "      fcolor = vec4(1, 0, 0, 1);\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 1.0f);

                SetData(&uniform[1], "iCount", 1, ARG_TYPE_INT, 1, 5);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
                g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 63:
            /* Test case 63 : Cross-stage optimisation : Vertex shader attribute "ocolor" components .xyzw are used
                             Vertex shader attribute  "bgcolor" components .xyzw are unused
                             Type: Some vertex shader attribute components are used and some are unused*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "out vec4 bgcolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor  = vec4(0.25, 0.375, 0.4375, 0.5);\n"
                "  bgcolor = vec4(0.625, 0.75, 0.8125, 0.9375);\n"
                "}\n";
            fs <<
                "in vec4 ocolor;\n"
                "in vec4 bgcolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "   int   iCount;\n"
                "   float f_scalar;"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    bvec4 bv_metric[4];\n"
                "};\n"
                "void main() {\n"
                " if (iCount == 5 && f_scalar == 1.0) {\n"
                "    if (ocolor.x == 0.25 && ocolor.y == 0.375 && ocolor.z == 0.4375 && ocolor.w == 0.5) {\n"
                "      fcolor = vec4(0, 1, 0, 1);\n"
                "    } else {\n"
                "      fcolor = vec4(0.8, 0, 0, 1);\n"
                "    }\n"
                " } else {\n"
                "      fcolor = vec4(1, 0, 0, 1);\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 1.0f);

                SetData(&uniform[1], "iCount", 1, ARG_TYPE_INT, 1, 5);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
                g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
        case 64:
            /* Test case 64 : Cross-stage optimisation :Vertex shader attribute "ocolor" components .xyzw are used
                             Vertex shader attribute  "bgcolor" components .xyzw are unused
                             Type: Some vertex shader attribute components are used and some are unused*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "out vec4 bgcolor;\n"
                "void main() {\n"
                "  gl_Position = vec4(position, 1.0);\n"
                "  ocolor  = vec4(0.25, 0.375, 0.4375, 0.5);\n"
                "  bgcolor = vec4(0.625, 0.75, 0.8125, 0.9375);\n"
                "}\n";
            fs <<
                "in vec4 ocolor;\n"
                "in vec4 bgcolor;\n"
                "out vec4 fcolor;\n"
                "layout(packed, binding = 0) uniform Block {\n"
                "   int   iCount;\n"
                "   float f_scalar;"
                "};\n"
                "layout(std140, binding = 1) uniform Block1 {\n"
                "    bvec4 bv_metric[4];\n"
                "};\n"
                "void main() {\n"
                " if (iCount == 5 && f_scalar == 1.0) {\n"
                "    if (ocolor.x == 0.25 && ocolor.y == 0.375 && ocolor.z == 0.4375 && ocolor.w == 0.5) {\n"
                "      fcolor = vec4(0, 1, 0, 1);\n"
                "    } else {\n"
                "      fcolor = vec4(0.8, 0, 0, 1);\n"
                "    }\n"
                " } else {\n"
                "    if (bgcolor.x == 0.625 && bgcolor.y == 0.75 && bgcolor.z == 0.8125 && bgcolor.w == 0.9375) {\n"
                "      fcolor = vec4(1, 0, 0, 1);\n"
                "    } else {\n"
                "      fcolor = vec4(0.9, 0, 0, 1);\n"
                "    }\n"
                " }\n"
                "}\n";

                SetData(&uniform[0], "f_scalar", 1, ARG_TYPE_FLOAT, 1, 1.0f);

                SetData(&uniform[1], "iCount", 1, ARG_TYPE_INT, 1, 5);

                g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
                g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
    case 65:
        /* Test case 65 : Bug : 200199667
        Uniform block Block1 has 2 members :
        1. float array fArray[8]
        2. int iArray[8]
        Float array fArray is specialised.
        fArray is dynamically indexed in the fragment shaders using a varying i.
        The final output color is set to green based on the value of fArray[i]
        Test  Output:
        1. Warning "Warning: Potentially failed to specialize a uniform array from UBO Block2."
        2. Output color should be green.
        */
        vs <<
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec3 color;\n"
            "flat out int i;\n"
            "void main() {\n"
            "  gl_Position = vec4(position, 1.0);\n"
            "  i       = 2; \n"
            "}\n";
        fs <<
            "flat in int i;\n"
            "out vec4 fcolor;\n"
            "layout(binding = 0) uniform Block2 {\n"
            "    float  fArray1[8];\n"
            "    int    iArray1[4];\n"
            "};\n"

            "void main() {\n"
            " if (fArray1[i] == 7.0) {\n"
            "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
            " } else {\n"
            "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            " }\n"
            "}\n";
        //Specialization of "fArray1" will be  a NOP here since we are dynamically indexing into the array "fArray1"
        SetData(&uniform[0], "fArray1", 8, ARG_TYPE_FLOAT, 8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
        break;
    case 66:
        /* Test case 66 : Bug : 200199667
        Uniform block Block3 has 2 members :
        1. float array fArray2[8]
        2. int iArray2[8]
        Int array iArray is only specialised.
        iArray is dynamically indexed in the fragment shaders using a varying i.
        The final output color is set to green based on the value of iArray[i]
        Test  Output:
        1. Warning "Warning: Potentially failed to specialize a uniform array from UBO Block2."
        2. Output color should be green.
        */
        vs <<
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec3 color;\n"
            "flat out int i;\n"
            "void main() {\n"
            "  gl_Position = vec4(position, 1.0);\n"
            "  i       = 2; \n"
            "}\n";
        fs <<
            "in vec4 ocolor;\n"
            "in vec4 bgcolor;\n"
            "flat in int i;\n"
            "out vec4 fcolor;\n"
            "layout(binding = 0) uniform Block3 {\n"
            "    float  fArray2[4];\n"
            "    int    iArray2[8];\n"
            "};\n"

            "void main() {\n"
            " if (iArray2[i] == 10 && fArray2[i] == 8.0) {\n"
            "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
            " } else {\n"
            "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            " }\n"
            "}\n";
        //Specialization of "iArray2" will be  a NOP here since we are dynamically indexing into the array "iArray2"
        SetData(&uniform[0], "iArray2", 8, ARG_TYPE_INT, 8, 1, 2, 3, 4, 5, 6, 7, 8);
        g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
        break;
    case 67:
        /* Test case 67 : Bug : 200199667
        Uniform block Block1 has 2 members :
        1. float array fArray[8]
        2. int iCount
        Float array fArray is specialised.
        fArray is dynamically indexed in the fragment shaders using a varying i.
        The final output color is set to green based on the value of fArray[i]
        Test  Output:
        1. Warning "Warning: Potentially failed to specialize a uniform array from UBO Block1."
        2. Output color should be green.
        */
        vs <<
            "layout(location=0) in vec3 position;\n"
            "layout(location=1) in vec3 color;\n"
            "flat out int i;\n"
            "void main() {\n"
            "  gl_Position = vec4(position, 1.0);\n"
            "  i       = 1; \n"
            "}\n";
        fs <<
            "flat in int i;\n"
            "out vec4 fcolor;\n"
            "layout(binding = 0) uniform Block1 {\n"
            "    float  fArray[8];\n"
            "    int   iCount;\n"
            "};\n"

            "void main() {\n"
            " if (fArray[i] == 8.0 && iCount == 10) {\n"
            "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
            " } else {\n"
            "      fcolor = vec4(0.0, 0.0, 1.0, 1.0);\n"
            " }\n"
            "}\n";
        //Specialization of "fArray" will be  a NOP here since we are dynamically indexing into the array "fArray"
        SetData(&uniform[0], "fArray", 8, ARG_TYPE_FLOAT, 8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
        break;
    case 68:
            /* Test with a UBO containing a structure within a structure (2-level).
               Here the values of the structure containing nested structures are specialiazed values*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   struct S s1;\n"
                "   int c;\n"
                "   int d;\n"
                "};\n"
                
                "layout(packed, binding = 1) uniform Block {\n"
                "    S2 s2;"
                "};\n"
                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if (s2.s1.a == 10.0 &&  s2.s1.b == 20.0 &&  s2.c == 30 &&  s2.d == 40) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "struct S\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "  struct S s1;\n"
                "   int c;\n"
                "   int d;\n"
                "};\n"
                "layout(packed, binding = 1) uniform Block {\n"
                "    S2 s2;"
                "};\n"
                
                "void main() {\n"
                " if (s2.s1.a == 10.0 &&  s2.s1.b == 20.0 &&  s2.c == 30 &&  s2.d == 40) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s2.s1.a", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[1], "s2.s1.b", 1, ARG_TYPE_FLOAT, 1, 20.0f);
            SetData(&uniform[2], "s2.c", 1, ARG_TYPE_INT, 1, 30);
            SetData(&uniform[3], "s2.d", 1, ARG_TYPE_INT, 1, 40);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            break;
    case 69:
            /* Test with a UBO containing a structure within a structure (3-level).
               Here the values of the structure containing nested structures are specialiazed values*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "struct S1\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   struct S1 s1;\n"
                "   int c;\n"
                "   int d;\n"
                "};\n"
                "struct S3\n"
                "{\n"
                "   struct S2 s2;\n"
                "   int e;\n"
                "   int f;\n"
                "};\n"
                
                "layout(packed, binding = 1) uniform Block {\n"
                "    S3 s3;"
                "};\n"
                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if (s3.s2.s1.a == 10.0 &&  s3.s2.s1.b == 20.0 &&  s3.s2.c == 30 &&  s3.s2.d == 40) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"
                "struct S1\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   struct S1 s1;\n"
                "   int c;\n"
                "   int d;\n"
                "};\n"
                "struct S3\n"
                "{\n"
                "   struct S2 s2;\n"
                "   int e;\n"
                "   int f;\n"
                "};\n"
                
                "layout(packed, binding = 1) uniform Block {\n"
                "    S3 s3;"
                "};\n"
                
                "void main() {\n"
                " if (s3.s2.s1.a == 10.0 &&  s3.s2.s1.b == 20.0 &&  s3.s2.c == 30 &&  s3.s2.d == 40) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s3.s2.s1.a", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[1], "s3.s2.s1.b", 1, ARG_TYPE_FLOAT, 1, 20.0f);
            SetData(&uniform[2], "s3.s2.c", 1, ARG_TYPE_INT, 1, 30);
            SetData(&uniform[3], "s3.s2.d", 1, ARG_TYPE_INT, 1, 40);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            break;
    case 70:
            /* Test with a UBO containing a structure within a structure (2-level).
               Here the structure S2 contains an array of nested structures
               Here the values of the structure containing nested structures are specialiazed values*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
               "struct S1\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   struct S1 s1[2];\n"
                "   vec4 v4_metric;\n"
                "   int c;\n"
                "   int d;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S2 s2[2];"
                "};\n"
               
                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if (s2[0].s1[0].a + s2[0].s1[0].b + s2[0].v4_metric.x + s2[0].v4_metric.y + s2[0].v4_metric.z == 80.0) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"

                "struct S1\n"
                "{\n"
                "   int   a;\n"
                "   float b;\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "   struct S1 s1[2];\n"
                "   vec4 v4_metric;\n"
                "   int c;\n"
                "   int d;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S2 s2[2];"
                "};\n"
                "void main() {\n"
                " if (s2[0].s1[0].a + s2[0].s1[0].b + s2[0].v4_metric.x + s2[0].v4_metric.y + s2[0].v4_metric.z == 80.0) {\n"
                "      fcolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s2[0].s1[0].a", 1, ARG_TYPE_INT, 1, 10);
            SetData(&uniform[1], "s2[0].s1[0].b", 1, ARG_TYPE_FLOAT, 1, 10.0f);
            SetData(&uniform[2], "s2[0].v4_metric", 1, ARG_TYPE_FLOAT, 4, 10.0, 20.0, 30.0, 40.0);

            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            break;
    case 71:
            /* Complex data structures : Test with a UBO containing a structure withan array of integers.
               Here the structure S2 contains an array of nested structures
               Here the values of the structure containing nested structures are specialiazed values*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "struct S1\n"
                "{\n"
                "    int   iCount;\n"
                "    int iArr[4];\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S1 s1[2];"
                "};\n"
               
                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if (s1[1].iArr[0] + s1[1].iArr[1] + s1[1].iCount == 4) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";
            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"

                "struct S1\n"
                "{\n"
                "   int   iCount;\n"
                "   int iArr[4];\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S1 s1[2];"
                "};\n"
               
                "void main() {\n"
                " if (s1[1].iArr[0] + s1[1].iArr[1] + s1[1].iCount == 4) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s1[1].iCount", 1, ARG_TYPE_INT, 1, 1);
            SetData(&uniform[1], "s1[1].iArr", 4, ARG_TYPE_INT, 16,
                        1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0);
           
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            break;
    case 72:
            /* Complex data structures :Test with a UBO containing a structure withan array of vectors .
               Here the structure S2 contains an array of nested structures
               Here the values of the structure containing nested structures are specialiazed values*/
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "struct S1\n"
                "{\n"
                "   int   iCount;\n"
                "   int iArr[4];\n"
                "   vec4 v_metric[4];\n"
                 "  bvec4 bv_metric[4];\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S1 s1[2];"
                "};\n"
               
                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if ((s1[1].iArr[0] + s1[1].iArr[1] + s1[1].iCount == 4) && \
                     (s1[1].bv_metric[0].x == true && s1[1].bv_metric[1].x == false && \
                     s1[1].bv_metric[2].x == true && s1[1].bv_metric[3].x == true) && \
                     (s1[1].v_metric[0].x + s1[1].v_metric[1].x  + s1[1].v_metric[2].x + s1[1].v_metric[2].x  == 4.0)) "
                "{\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";
            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"

                "struct S1\n"
                "{\n"
                "   int   iCount;\n"
                "   int iArr[4];\n"
                "   vec4 v_metric[4];\n"
                 "  bvec4 bv_metric[4];\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S1 s1[2];"
                "};\n"
               
                "void main() {\n"
                " if ((s1[1].iArr[0] + s1[1].iArr[1] + s1[1].iCount == 4) && \
                     (s1[1].bv_metric[0].x == true && s1[1].bv_metric[1].x == false && \
                     s1[1].bv_metric[2].x == true && s1[1].bv_metric[3].x == true) && \
                     (s1[1].v_metric[0].x + s1[1].v_metric[1].x  + s1[1].v_metric[2].x + s1[1].v_metric[2].x  == 4.0)) "
                "{\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s1[1].iCount", 1, ARG_TYPE_INT, 1, 1);
            SetData(&uniform[1], "s1[1].iArr", 4, ARG_TYPE_INT, 16,
                        1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0);
            SetData(&uniform[2], "s1[1].bv_metric", 4, ARG_TYPE_INT, 16, 
                                                                    1, 1, 1, 1,
                                                                    0, 1, 1, 1,
                                                                    1, 1, 1, 1,
                                                                    1, 1, 1, 1);
            SetData(&uniform[3], "s1[1].v_metric", 4, ARG_TYPE_FLOAT, 16,
                                                            1.0f, 2.0f, 3.0f, 4.0f,
                                                            1.0f, 2.0f, 3.0f, 4.0f,
                                                            1.0f, 2.0f, 3.0f, 4.0f,
                                                            1.0f, 2.0f, 3.0f, 4.0f);
           
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            break;
    case 73:
            /* Complex data structure : Test with a UBO containing a structure with array of integers, 
               vectors, boolean vectors and matrices (column_major).
               Here the values of the members belonging to the structure are specialiazed */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "struct S1\n"
                "{\n"
                "   int   iCount;\n"
                "   int iArr[4];\n"
                "   vec4 v_metric[4];\n"
                 "  bvec4 bv_metric[4];\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "    mat2 m2;\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S1 s1[2];\n"
                "};\n"
                
                "layout(packed, binding = 1,column_major)  uniform Block1 {\n"
               "    S2 s2[2];\n"
                "};\n"
               
                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if ((s1[1].iArr[0] + s1[1].iArr[1] + s1[1].iCount == 4) && \
                     (s1[1].bv_metric[0].x == true && s1[1].bv_metric[1].x == false && \
                     s1[1].bv_metric[2].x == true && s1[1].bv_metric[3].x == true) && \
                     (s1[1].v_metric[0].x + s1[1].v_metric[1].x  + s1[1].v_metric[2].x + s1[1].v_metric[2].x  == 4.0) && \
                     (s2[1].m2[0][0] + s2[1].m2[1][0] == 4.0)) \n"
                "{\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";
            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"

               "struct S1\n"
                "{\n"
                "   int   iCount;\n"
                "   int iArr[4];\n"
                "   vec4 v_metric[4];\n"
                 "  bvec4 bv_metric[4];\n"
                "};\n"
                "struct S2\n"
                "{\n"
                "    mat2 m2;"
                "};\n"
                
                "layout(std140, binding = 0) uniform Block {\n"
                "    S1 s1[2];"
                "};\n"
                
                "layout(packed, binding = 1,column_major) uniform Block1 {\n"
                "    S2 s2[2];"
                "};\n"
               
                "void main() {\n"
                " if ((s1[1].iArr[0] + s1[1].iArr[1] + s1[1].iCount == 4) && \
                     (s1[1].bv_metric[0].x == true && s1[1].bv_metric[1].x == false && \
                     s1[1].bv_metric[2].x == true && s1[1].bv_metric[3].x == true) && \
                     (s1[1].v_metric[0].x + s1[1].v_metric[1].x  + s1[1].v_metric[2].x + s1[1].v_metric[2].x  == 4.0) && \
                     (s2[1].m2[0][0] + s2[1].m2[1][0] == 4.0)) \n"
               "{\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s1[1].iCount", 1, ARG_TYPE_INT, 1, 1);
            SetData(&uniform[1], "s1[1].iArr", 4, ARG_TYPE_INT, 16,
                        1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0);
            SetData(&uniform[2], "s1[1].bv_metric", 4, ARG_TYPE_INT, 16, 
                                                                    1, 1, 1, 1,
                                                                    0, 1, 1, 1,
                                                                    1, 1, 1, 1,
                                                                    1, 1, 1, 1);
            SetData(&uniform[3], "s1[1].v_metric", 4, ARG_TYPE_FLOAT, 16,
                                                            1.0f, 2.0f, 3.0f, 4.0f,
                                                            1.0f, 2.0f, 3.0f, 4.0f,
                                                            1.0f, 2.0f, 3.0f, 4.0f,
                                                            1.0f, 2.0f, 3.0f, 4.0f);
            SetData(&uniform[4], "s2[1].m2", 1, ARG_TYPE_FLOAT, 4,
                        1.0f, 2.0f,
                        3.0f, 4.0);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[4]);
            break;
    case 74:
            /* Complex data structure : Test with a UBO containing a structure with array of integers, 
               vectors, boolean vectors and matrices (row_major).
               Here the values of the members belonging to the structure are specialiazed */
            vs <<
                "layout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\n"
                "out vec4 ocolor;\n"
                "struct S1\n"
                "{\n"
                "   int   iCount;\n"
                "   int iArr[4];\n"
                "   vec4 v_metric[4];\n"
                 "  bvec4 bv_metric[4];\n"
                "};\n"
                "struct S3\n"
                "{\n"
                "    mat2 m3[2];\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S1 s1[2];\n"
                "};\n"
                
                
                "layout(packed, binding = 2, row_major)  uniform Block1 {\n"
                "    S3 s3[2];\n"
                "};\n"
               
                "void main() {\n"
                " gl_Position = vec4(position, 1.0);\n"
                " if ((s1[1].iArr[0] + s1[1].iArr[1] + s1[1].iCount == 4) && \
                     (s1[1].bv_metric[0].x == true && s1[1].bv_metric[1].x == false && \
                     s1[1].bv_metric[2].x == true && s1[1].bv_metric[3].x == true) && \
                     (s1[1].v_metric[0].x + s1[1].v_metric[1].x  + s1[1].v_metric[2].x + s1[1].v_metric[2].x  == 4.0) && \
                     (s3[1].m3[0][1][0] + s3[1].m3[1][0][1] == 5.0)) {\n"
                "      ocolor = vec4(0.0, 1.0, 0.0, 1.0);\n"
                " } else {\n"
                "      ocolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";
            fs <<
                "in vec4 ocolor;\n"
                "out vec4 fcolor;\n"

               "struct S1\n"
                "{\n"
                "   int   iCount;\n"
                "   int iArr[4];\n"
                "   vec4 v_metric[4];\n"
                 "  bvec4 bv_metric[4];\n"
                "};\n"
                "struct S3\n"
                "{\n"
                "    mat2 m3[2];\n"
                "};\n"
                "layout(std140, binding = 0) uniform Block {\n"
                "    S1 s1[2];"
                "};\n"
                
                "layout(packed, binding = 2, row_major)  uniform Block1 {\n"
                "    S3 s3[2];\n"
                "};\n"
               
                "void main() {\n"
                " if ((s1[1].iArr[0] + s1[1].iArr[1] + s1[1].iCount == 4) && \
                     (s1[1].bv_metric[0].x == true && s1[1].bv_metric[1].x == false && \
                     s1[1].bv_metric[2].x == true && s1[1].bv_metric[3].x == true) && \
                     (s1[1].v_metric[0].x + s1[1].v_metric[1].x  + s1[1].v_metric[2].x + s1[1].v_metric[2].x  == 4.0) && \
                     (s3[1].m3[0][1][0] + s3[1].m3[1][0][1] == 5.0)) {\n"
                "      fcolor = ocolor;\n"
                " } else {\n"
                "      fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
                " }\n"
                "}\n";

            SetData(&uniform[0], "s1[1].iCount", 1, ARG_TYPE_INT, 1, 1);
            SetData(&uniform[1], "s1[1].iArr", 4, ARG_TYPE_INT, 16,
                        1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0);
            SetData(&uniform[2], "s1[1].bv_metric", 4, ARG_TYPE_INT, 16, 
                                                                    1, 1, 1, 1,
                                                                    0, 1, 1, 1,
                                                                    1, 1, 1, 1,
                                                                    1, 1, 1, 1);
            SetData(&uniform[3], "s1[1].v_metric", 4, ARG_TYPE_FLOAT, 16,
                                                            1.0f, 2.0f, 3.0f, 4.0f,
                                                            1.0f, 2.0f, 3.0f, 4.0f,
                                                            1.0f, 2.0f, 3.0f, 4.0f,
                                                            1.0f, 2.0f, 3.0f, 4.0f);
            
            SetData(&uniform[4], "s3[1].m3", 2, ARG_TYPE_FLOAT, 16,
                        0.0f, 5.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 5.0f, 0.0f, 0.0f,
                        1.0f, 0.0f, 0.0f, 0.0f);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[0]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[1]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[2]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[3]);
            g_glslcHelper->AddSpecializationUniform(0, &uniform[4]);
            
            break;
    }

    LWNboolean compileSuccess = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);

    g_glslcHelper->ClearSpecializationUniformArrays();

    if (!compileSuccess) {
        printf("Shader compilation failure on test %d\n", test_number);
        printf("Infolog: \n%s\n", g_glslcHelper->GetInfoLog());
        return false;
    }

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    return true;
}

void LWNShaderSpecialization::doGraphics() const {
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    int vboSize = 1024*1024;

    int vpWidth = lwrrentWindowWidth / cellsX;
    int vpHeight = lwrrentWindowHeight / cellsY;
    int row = 0;
    int col = 0;


    MemoryPoolAllocator allocator(device, NULL, vboSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexArrayState vertex = stream.CreateVertexArrayState();
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *vbo = allocator.allocBuffer(&bb, BUFFER_ALIGN_VERTEX_BIT, vboSize);
    BufferAddress vboAddr = vbo->GetAddress();

    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, vboSize);

    Vertex *vboMap = (Vertex *) vbo->Map();

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 1.0);

    int vnum = 0;
    static LWNsizeiptr coherentPoolSize = 0x100000UL; // 1MB pool size
    MemoryPoolAllocator coherent_allocator(device, NULL, coherentPoolSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    for (int test_num = 0; test_num < total_tests; test_num++)
    {
        Program *pgm = device->CreateProgram();
        if (test_num == 65 || test_num == 66 || test_num == 67) {
            if (test_num == 65) {
                typedef struct {
                    float  fArray1[8];
                    int    iArray1[4];
                } UniformBlock;
                UniformBlock blockData;
                for (int i = 0; i < 8; i++) {
                    blockData.fArray1[i] = 7.0;
                }
                for (int i = 0; i < 4; i++) {
                    blockData.iArray1[i] = 8;
                }
                Buffer *ubo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, &blockData, sizeof(blockData), BUFFER_ALIGN_UNIFORM_BIT, false);
                BufferAddress uboAddr = ubo->GetAddress();
                queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr, sizeof(blockData));
            } else if (test_num == 66) {
                typedef struct {
                    float  fArray2[4];
                    int    iArray2[8];
                } UniformBlock;
                UniformBlock blockData;

                for (int i = 0; i < 4; i++) {
                    blockData.fArray2[i] = 8.0;
                }
                for (int i = 0; i < 8; i++) {
                    blockData.iArray2[i] = 10;
                }
                Buffer *ubo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, &blockData, sizeof(blockData), BUFFER_ALIGN_UNIFORM_BIT, false);
                BufferAddress uboAddr = ubo->GetAddress();
                queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr, sizeof(blockData));
            } else if (test_num == 67) {
                typedef struct {
                    float  fArray[8];
                    int iCount;
                } UniformBlock;
                UniformBlock blockData;
                blockData.iCount = 10;
                for (int i = 0; i < 8; i++) {
                    blockData.fArray[i] = 8.0;
                }
                Buffer *ubo = AllocAndFillBuffer(device, queue, queueCB, coherent_allocator, &blockData, sizeof(blockData), BUFFER_ALIGN_UNIFORM_BIT, false);
                BufferAddress uboAddr = ubo->GetAddress();
                queueCB.BindUniformBuffer(ShaderStage::FRAGMENT, 0, uboAddr, sizeof(blockData));
            }
        }
        bool compileSuccess = RunTest(test_num, pgm);

        row = test_num / cellsX;
        col = test_num % cellsX;

        queueCB.SetViewportScissor(col * vpWidth + 2, (row + 1) * vpHeight + 2, vpWidth - 4, vpHeight - 4);

        for (int v = 0; v < 4; ++v) {
            vboMap[vnum + v].position[0] = (v & 2) ? +1.0 : -1.0;
            vboMap[vnum + v].position[1] = (v & 1) ? +1.0 : -1.0;
        }

        if (compileSuccess) {
            queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, vnum, 4);
        } else {
            queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
        }

        vnum += 4;
    }

    queueCB.submit();

    queue->Finish();
}

OGTEST_CppTest(LWNShaderSpecialization, lwn_shader_specialization,  );
