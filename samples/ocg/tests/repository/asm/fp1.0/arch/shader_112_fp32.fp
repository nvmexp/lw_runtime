!!FP1.0 
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, f[TEX2], TEX2, 2D;
TEX R3, f[TEX3], TEX3, 2D;
TEX R4, f[TEX4], TEX4, 2D;
TEX R5, f[TEX5], TEX5, 2D;
DP3R_SAT R3.xyz, R3, R4;
MULR R3.xyz, R3, R3;
MULR R3.xyz, R3, R0.w;
MULR R3.xyz, R3, R5;
MULR R0.xyz, R0, R2;
MADR R0.xyz, R3, f[COL0], R0;
MULR H0.xyz, R0, R1;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVR H0.w, f[COL0].w;
MOVH o[COLH], H0; 
END

# Passes = 12 

# Registers = 6 

# Textures = 6 
