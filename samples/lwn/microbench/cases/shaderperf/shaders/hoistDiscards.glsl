#version 450

#define HOISTDISCARDS 0
#if HOISTDISCARDS
#pragma hoistDiscards true
#else
#pragma hoistDiscards false
#endif

// 8 varyings to test
in IO {
    vec4 varying1;
    float varying2;
    float varying3;
    vec4 varying4;
    float varying5;
    float varying6;
    flat int varying7;
    vec2 vPos;
    vec2 fTexCoord;
};

#define SAMPLERINDEX 0
layout(binding=SAMPLERINDEX) uniform sampler2D tex;

out vec4 color;

void main()
{
    color = vec4(0.1, 0.1, 0.1, 0.1);

#define LOOPS 5
    for (int i = 0; i < LOOPS; i++) {
        color += (varying1 * 0.7);
        color *= (varying2);
        color /= (varying3);
        color *= (varying4);
        color.y += (varying5 * 0.25);
        color.z -= (varying6 * 0.3);
        color.x += varying7;
    }

    // discard pixels for half the screen.
    // viewport is fullscreen, but we
    // call setscissor (w/4, h/4)
    if (vPos.x <= -0.75) {
        discard;
    }

    color += 0.5 * texture(tex, fTexCoord);
}