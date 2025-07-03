#version 450

#define IS_PREPROCESSED 0
#define UNI_SPEC 0.00 // will be changed to non-zero value when IS_PREPROCESSED == 1

uniform block {
    float specConst;
    vec4 val1;
    float val2;
    float val3;
};

out vec4 color;

void main()
{
    color = vec4(0.4, 0.4, 0.4, 0.4);

#define LOOPS 24
    int i = 0;
    while(true) {
        if (i > LOOPS ) {
            break;
        }
        /*
        if specConst1/UNI_SPEC == 2:
            color += -val1;
            color +=  val1;
            color += -val1
            color.r *= val2 * 0.25;
            color.g *= 0.333 * val1;
            color.b -= 0.0;
        */
#if IS_PREPROCESSED
        color += -val1 *  UNI_SPEC + val1;
        color += -val1 * -UNI_SPEC - val1;
        color +=  val1 * -UNI_SPEC + val1;
        color.r *= val2 * (UNI_SPEC / UNI_SPEC) * 0.25;
        color.g *= 0.333 * val2 / UNI_SPEC * UNI_SPEC;
        color.b -= UNI_SPEC * val3 - val3 - val3;
#else
        color += -val1 *  specConst + val1;
        color += -val1 * -specConst - val1;
        color +=  val1 * -specConst + val1;
        color.r *= val2 * (specConst / specConst) * 0.25;
        color.g *= 0.333 * val2 / specConst * specConst;
        color.b -= specConst * val3 - val3 - val3;
#endif
        i++;
    }
}