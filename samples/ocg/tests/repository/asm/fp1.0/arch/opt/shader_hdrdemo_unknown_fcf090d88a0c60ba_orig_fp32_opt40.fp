!!FP2.0
DECLARE C0={0.1, 0.3, 0.5, 0.9};
DECLARE C1={0.25, 0.22, 0.27, 0.33};
TEX R1, f[TEX1], TEX1, 2D;
MADR R1.w, R1, C0.z, C0.y;
MULR R1.w, R1, C0.w;
EX2R R1.w, R1.w;
TEX R0, f[TEX0], TEX0, 2D;
MULR R2.xyz, R1, R1.w;
MADR R0.w, R0, C0.z, C0.y;
MULR R0.w, R0, C0.w;
EX2R R0.w, R0.w;
MULR R1.xyz, R0, R0.w;
MULR R1.xyz, R1, C0.w;
TEX R0, f[TEX1], TEX2, 2D;
MADR R1.xyz, R2, C1, R1;
MADR R0.w, R0, C0.w, C0.x;
MULR R0.w, R0, C0.z;
EX2R R0.w, R0.w;
MULR R0.xyz, R0, R0.w;
MADR R0.xyz, R0, C1, R1;
MOVR R1.xy, f[TEX2];
DP2AR R0.w, C1.x, R1, R1.xyxy;
ADDR R0.w, -R0, C0.w;
MULR R1.w, R0, R0;
MULR R0.w, R0, R1;
MULR R0.xyz, R0, R0.w;
MOVR R0.w, C0.z;
LG2R R0.x, |R0.x|;
LG2R R0.z, |R0.z|;
LG2R R0.y, |R0.y|;
MULR R0.xyz, R0, C0.w;
EX2R R0.x, R0.x;
EX2R R0.z, R0.z;
EX2R R0.y, R0.y;
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
