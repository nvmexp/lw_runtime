#ifndef __g_hoistDiscards_h_
#define __g_hoistDiscards_h_

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!! AUTOMATICALLY GENERATED - DO NOT EDIT !!!
// !!! Please refer to README.txt on how to  !!!
// !!! generate such files                   !!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

static const char *shader_hoistDiscards =
    "#version 450\n"
    "\n"
    "#define HOISTDISCARDS 0\n"
    "#if HOISTDISCARDS\n"
    "#pragma hoistDiscards true\n"
    "#else\n"
    "#pragma hoistDiscards false\n"
    "#endif\n"
    "\n"
    "// 8 varyings to test\n"
    "in IO {\n"
    "    vec4 varying1;\n"
    "    float varying2;\n"
    "    float varying3;\n"
    "    vec4 varying4;\n"
    "    float varying5;\n"
    "    float varying6;\n"
    "    flat int varying7;\n"
    "    vec2 vPos;\n"
    "    vec2 fTexCoord;\n"
    "};\n"
    "\n"
    "#define SAMPLERINDEX 0\n"
    "layout(binding=SAMPLERINDEX) uniform sampler2D tex;\n"
    "\n"
    "out vec4 color;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    color = vec4(0.1, 0.1, 0.1, 0.1);\n"
    "\n"
    "#define LOOPS 5\n"
    "    for (int i = 0; i < LOOPS; i++) {\n"
    "        color += (varying1 * 0.7);\n"
    "        color *= (varying2);\n"
    "        color /= (varying3);\n"
    "        color *= (varying4);\n"
    "        color.y += (varying5 * 0.25);\n"
    "        color.z -= (varying6 * 0.3);\n"
    "        color.x += varying7;\n"
    "    }\n"
    "\n"
    "    // discard pixels for half the screen.\n"
    "    // viewport is fullscreen, but we\n"
    "    // call setscissor (w/4, h/4)\n"
    "    if (vPos.x <= -0.75) {\n"
    "        discard;\n"
    "    }\n"
    "\n"
    "    color += 0.5 * texture(tex, fTexCoord);\n"
    "}\n";

#endif /* __g_hoistDiscards_h_ */
