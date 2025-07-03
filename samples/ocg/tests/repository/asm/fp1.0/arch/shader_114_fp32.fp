!!FP1.0
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, f[TEX2], TEX2, 2D;
TEX R3, f[TEX3], TEX3, 2D;
TEX R5, f[TEX5], TEX5, 2D;
DP3R_SAT R2.xyz, R2, R3;
TEX R4, f[TEX4], TEX4, 2D;
MULR R2.xyz, R2, R4;
MADR R1.xyz, R2, f[COL0], R1;
MULR R0.xyz, R0, R1;
MULR R0.xyz, R0, R5;
MULR R0.xyz, R0, {2, 0, 0, 0}.x; 
MOVR R0.w, f[COL0].w;
MOVR o[COLR], R0; 
END

# Passes = 10 

# Registers = 6 

# Textures = 6 
