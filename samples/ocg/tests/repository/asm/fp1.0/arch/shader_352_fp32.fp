!!FP1.0
MOVR R1.xyz, f[TEX3];
MOVR R2.xyz, f[TEX2];
TEX R0, f[TEX0], TEX1, 2D;
MULR R1.xyz, R0.y, R1;
MADR R0.xyz, R2.xyzz, R0.x, R1.xyzz;
MOVR R1.xyz, f[TEX4];
MADR R0.xyz, R1, R0.z, R0;
MOVR R1.xyz, f[TEX1];
DP3R R1.y, R0, R1;
MOVR R2.xyz, f[TEX1];
DP3R R1.x, R0, R0;
RCPR R1.w, R1.x;
DP3R R2.x, R2, R2;
MULR R0.w, R1, R1.y;
MOVR R1.xy, f[TEX1];
ADDR R1.w, R0, R0;
MOVR R2.yzw, f[TEX1].wxyz;
MADR R1.xy, R1.w, R0, -R1;
TEX R1, R1, TEX7, LWBE;
LG2R R0.w, |R2.x|;
MULR R0.w, R0, {0.5, 0, 0, 0}.x; 
EX2R R1.w, -R0.w;
MULR R2.yzw, R1.w, R2;
DP3R R0.x, R2.yzwy, R0;
SGER R0.z, R0.x, {0.855620, 0.770058, 0.693052, 0.623747}.x;
ADDR R0.w, R0.x, -{0.561372, 0.505235, 0.454712, 0.409240}.x;
LG2R R0.x, |R2.x|;
MULR R0.x, R0, {0.5, 0, 0, 0}.x; 
MULR R0.w, R0.z, -R0;
ADDR R1.w, {0.241652, 0.217487, 0.195738, 0.176165}.x, R0;
EX2R R0.y, -R0.x;
ADDR R0.w, -R1, {0.958548, 0.142693, 0.128424, 0.115582}.x;
MULR R0.z, R0.w, R0.w;
MOVR R0, R0.z;
MULR R1.w, R0.y, R2.x;
MULR R0.z, R0, R0;
MULR R0.w, R0, R0.z;
MADR R1.xyz, R1, R0.w, {0.104023, 0.093621, 0.084259, 0.075833};
MOVR R0.w, -{0.0068250, 0.061425, 0.055282, 0.049754}.x;
ADDR R1.w, R1, R0;
ADDR R0.w, R0, {0.044779, 0.040301, 0.036271, 0.032644}.x;
RCPR R0.w, R0.w;
MULR_SAT R1.w, R1, R0;
MOVR R0, R1;
MOVR o[COLR], R0; 
END

# Passes = 28 

# Registers = 3 

# Textures = 5 
