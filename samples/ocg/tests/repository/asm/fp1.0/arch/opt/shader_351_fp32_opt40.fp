!!FP2.0
MOVR R1.xyz, f[TEX1];
MOVR R0.w, f[TEX1];
RCPR R0.w, R0.w;
MULR R1.xyz, R0.w, R1;
TEX R0, f[TEX0], TEX0, 2D;
DP3R R2.x, R1, R0;
DIVR R1.xyz, f[TEX2], f[TEX2].w;
DP3R R2.y, R1, R0;
TEX R1, f[TEX3], TEX3, 2D;
MADR R1.xyz, {1.987654, 0, 0, 0}.x, R1, {1.987654, 0, 0, 0}.y;
TEX R0, R2, TEX2, 2D;
DP3R R0.w, f[TEX4], f[TEX4];
MOVR R2.yzw, f[TEX4].wxyz;
LG2R R0.w, |R0.w|;
MULR R2.x, R0.w, {1.304100, 0, 0, 0}.x;
EX2R R0.w, R2.x;
MULR R2.xyz, R0.w, R2.yzwy;
DP3R R1.x, R2, R1;
SGER R0.w, R1.x, {0, 0, 0, 0}.x;
ADDR R1.z, R1.x, -{0.561372, 0, 0, 0}.x;
MULR R0.w, R0, R1.z;
ADDR R0.w, {0.368316, 0, 0, 0}.x, R0;
ADDR R1.z, -R0.w, {0.741652, 0, 0, 0}.x;
MULR R0.w, R1.z, R1.z;
MULR R0.w, R0, R0;
MULR R0.w, R1.z, R0;
MULR R0.xyz, R0, R0.w;
MOVR R0.w, R1;
END

# Passes = 17 

# Registers = 3 

# Textures = 5 
