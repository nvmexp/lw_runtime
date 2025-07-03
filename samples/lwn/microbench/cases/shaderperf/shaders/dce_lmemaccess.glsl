#version 450

#define IS_PREPROCESSED 0

// Possible values:
// UNI_COUNT:  [0-4]
// UNI_INDEX:  [0-1]
// UNI_BRANCH: [0-1]

#if IS_PREPROCESSED
#define UNI_COUNT 0
#define UNI_INDEX 0
#define UNI_BRANCH 0
#else
#define UNI_COUNT specConstCount
#define UNI_INDEX specConstIndex
#define UNI_BRANCH specConstBranch
#endif

uniform block
{
    int specConstCount;
    int specConstIndex;
    int specConstBranch;
};

out vec4 color;

vec3 foo( vec4 lhs, vec4 rhs ) {
    vec4 ret = vec4( 0 );
    ret.x = ((lhs.x < 0.5) ? (lhs.x * rhs.x * 2) : (1 - 2*(1-lhs.x)*(1-rhs.x)));
    return ret.xyz;
}

vec4 hoge() {
    vec4 ret[ 2 ] = vec4[]( vec4( 0 ), vec4( 0 ) );
    vec4 ret2[ 2 ] = vec4[]( vec4(0.3), vec4(0.3) );

    #define LOOPS 32
    int loops = 0;

    while (true) {
        if (loops >= LOOPS) {
            break;
        }
        loops++;
        if( UNI_COUNT >= 1 ) {
            ret[ UNI_INDEX ].xyz = foo( vec4( 0 ), vec4( 0 ) );
        }

        if( UNI_COUNT >= 2 ) {
            vec4 lhs = vec4( 0 );
            vec4 rhs = vec4( 0 );
            if( UNI_BRANCH == 1 ) lhs = ret2[ 0 ];

            ret2[ UNI_INDEX ].xyz += foo( lhs, rhs );
        }

        if( UNI_COUNT >= 3 ) {
            vec4 lhs = vec4( 0 );
            vec4 rhs = vec4( 0 );
            if( UNI_BRANCH == 1 ) lhs = ret[ 0 ];

            ret[ UNI_INDEX ].xyz += foo( lhs, rhs );
        }
    }

    return ret[ UNI_INDEX ] + 0.5 * ret2[ UNI_INDEX ];
}

void main() {
    vec4 hogehoge = vec4( 1.0 );
    hogehoge *= hoge();
    color = vec4( 0 );
}
