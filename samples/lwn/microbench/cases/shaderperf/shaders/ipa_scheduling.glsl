#version 450

#define ARRSIZE 128
#define UNI_SPEC 0
#if UNI_SPEC == 1
#define COND false
#else
#define COND true
#endif

uniform block {
    bool specConst;
    uint loops;
};

// 8 varyings to test
in IO {
    vec4 varying1;
    float varying2;
    float varying3;
    vec4 varying4;
    float varying5;
    float varying6;
    flat int varying7;
};

out vec4 color;

void main()
{
    color = vec4(0.4, 0.4, 0.4, 0.4);

    if (specConst && COND) {
        color += (varying1 * 0.9);
        color *= (varying2);
        color /= (varying3);
        color *= (varying4);
        color.y += (varying5 * 1.2);
        color.z -= (varying6 * 0.3);
        color.x += varying7;
    }

    float arr2[ARRSIZE];
    for(uint i = 0 ; i < ARRSIZE ; i++) {
        arr2[i] = i;
    }

    // Insert a bunch of work
    vec4 vals[2] = { vec4(0.3, 0.8, 0.7, 0.2),
                     vec4(1.4, 3.2, -2.7, 2.1) };
    vec4 vals2[4] = { vec4(4.2, 1.8, 0.2, 1.1),
                      vec4(10.4, 0.8, 0.9, 0.2),
                      vec4(-0.4, 0.9, 0.7, -0.5),
                      vec4(-0.3, 1.2, 0.05, 0.3) };
    uint i = 0;

    while(true) {
        if (i >= loops) {
            break;
        }
        color += vals[i % 2] + vals2[i % 4];
        color /= vals2[i % 4] / arr2[i % ARRSIZE];
        i++;
    }

    // Use all the varyings again
    color += (varying1 * 0.9);
    color *= (varying2 + 0.12);
    color /= (varying3);
    color *= (varying4) + 0.05;
    color.z += (varying5 * 1.7);
    color.y -= (varying6 * 0.5);
    color.x += varying7;
}