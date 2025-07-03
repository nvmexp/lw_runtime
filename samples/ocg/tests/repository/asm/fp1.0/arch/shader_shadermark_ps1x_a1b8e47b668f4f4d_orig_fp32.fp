!!FP1.0
MOVR R0.xyz, f[TEX0];
MOVR R3.xyz, f[TEX1];
MOVR R2.xyz, f[TEX3];
TEX R1, f[TEX4], TEX1, 2D;
MADR R1.xyz, R1, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3R R0.x, R0, R1;
MADR R4.xyz, R1, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
MOVR R1.xyz, f[TEX1];
DP3R R0.y, R3, R4;
MADR R3.xyz, R4, {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3R R0.z, R2, R3;
DP3R R0.x, R0, R1;
MULR R0.x, R0, R0;
TEX R0, R0.x, TEX5, 2D;
MOVR R0.w, R0.y;
MULR R0.xyz, {0.1, 0.5, 0.3, 1.0}, R0;
MOVR o[COLR], R0; 
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
