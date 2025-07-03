#ifndef __g_algebraic_expression_h_
#define __g_algebraic_expression_h_

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!! AUTOMATICALLY GENERATED - DO NOT EDIT !!!
// !!! Please refer to README.txt on how to  !!!
// !!! generate such files                   !!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

static const char *shader_algebraic_expression =
    "#version 450\n"
    "\n"
    "#define IS_PREPROCESSED 0\n"
    "#define UNI_SPEC 0.00 // will be changed to non-zero value when IS_PREPROCESSED == 1\n"
    "\n"
    "uniform block {\n"
    "    float specConst;\n"
    "    vec4 val1;\n"
    "    float val2;\n"
    "    float val3;\n"
    "};\n"
    "\n"
    "out vec4 color;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    color = vec4(0.4, 0.4, 0.4, 0.4);\n"
    "\n"
    "#define LOOPS 24\n"
    "    int i = 0;\n"
    "    while(true) {\n"
    "        if (i > LOOPS ) {\n"
    "            break;\n"
    "        }\n"
    "        /*\n"
    "        if specConst1/UNI_SPEC == 2:\n"
    "            color += -val1;\n"
    "            color +=  val1;\n"
    "            color += -val1\n"
    "            color.r *= val2 * 0.25;\n"
    "            color.g *= 0.333 * val1;\n"
    "            color.b -= 0.0;\n"
    "        */\n"
    "#if IS_PREPROCESSED\n"
    "        color += -val1 *  UNI_SPEC + val1;\n"
    "        color += -val1 * -UNI_SPEC - val1;\n"
    "        color +=  val1 * -UNI_SPEC + val1;\n"
    "        color.r *= val2 * (UNI_SPEC / UNI_SPEC) * 0.25;\n"
    "        color.g *= 0.333 * val2 / UNI_SPEC * UNI_SPEC;\n"
    "        color.b -= UNI_SPEC * val3 - val3 - val3;\n"
    "#else\n"
    "        color += -val1 *  specConst + val1;\n"
    "        color += -val1 * -specConst - val1;\n"
    "        color +=  val1 * -specConst + val1;\n"
    "        color.r *= val2 * (specConst / specConst) * 0.25;\n"
    "        color.g *= 0.333 * val2 / specConst * specConst;\n"
    "        color.b -= specConst * val3 - val3 - val3;\n"
    "#endif\n"
    "        i++;\n"
    "    }\n"
    "}\n";

#endif /* __g_algebraic_expression_h_ */
