static const char *vert_ColTex =
R"glsl_str(
#version 450 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec4 aColor;
layout(location = 2) in vec2 aTexCoord;

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec2 vTexCoord;

void main()
{
    gl_Position = vec4(aPosition,1.0);
    vColor = aColor;
    vTexCoord = aTexCoord;
}
)glsl_str";

static const char *frag_Col =
R"glsl_str(
#version 450 core
layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vTexCoord;

layout(location = 0) out vec4 oFrag;

void main()
{
    oFrag = vColor;
}
)glsl_str";

static const char *frag_ColTex =
R"glsl_str(
#version 450 core
layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vTexCoord;

layout(location = 0) out vec4 oFrag;
layout(binding = 0) uniform sampler2D tex2D;

void main()
{
    oFrag = texture(tex2D, vTexCoord);
}
)glsl_str";

static const char *frag_Bitmap =
R"glsl_str(
#version 450 core
layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vTexCoord;

layout(location = 0) out vec4 oFrag;
layout(binding = 0) uniform sampler2D tex2D;

void main()
{
    float valid = texture(tex2D, vTexCoord).r;

    if (valid == 0.0) {
        discard;
    }
    oFrag = vColor;
}
)glsl_str";
