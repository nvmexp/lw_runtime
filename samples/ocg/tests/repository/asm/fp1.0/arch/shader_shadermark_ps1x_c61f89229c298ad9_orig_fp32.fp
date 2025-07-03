!!FP1.0
MOVR R0.xyz, f[TEX1];
MOVR R2.xyz, f[TEX3];
TEX R1, f[TEX0], TEX0, 2D;
MADR R3.xyz, R1, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
MOVR R1.xyz, f[TEX2];
DP3R R0.x, R0, R3;
MADR R3.xyz, R3, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3R R0.y, R1, R3;
TEX R1, f[TEX4], TEX1, 2D;
MADR R3.xyz, R3, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3R R0.z, R2, R3;
DP3R R0.w, R0, R0;
MADR R2.xyz, R1, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3R R0.z, R0, R2;
MULR R0.z, R0, {2, 2, 2, 2}.x;
TEX R2, R0, TEX4, 2D;
MULR R2.xyz, R2, {2, 2, 2, 2}.x;
MADR R1.zw, -R1.xxxy, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
MULR R1.xy, R0, R0.z;
MADR R0.xy, -R1.zwzz, R0.w, R1;
MOVR R0.zw, f[TEX5].xxxy;
TEX R1, f[TEX0], TEX5, 2D;
ADDR R0.xy, R0, R0.zwzz;
TEX R0, R0, TEX2, 2D;
MULR R0.xyz, R1, R0;
TEX R1, f[TEX0], TEX3, 2D;
MADR R0.xyz, R1, R2, R0;
MOVR R1.xyz, R0;
MOVR R0, R1;
MOVR o[COLR], R0; 
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
