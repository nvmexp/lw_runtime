!!FP2.0 
TEX R1, f[TEX3], TEX3, 2D;
MOVR R0.w, f[COL0].w;
TEX R2, f[TEX2], TEX2, 2D;
DP3R_SAT R2.xyz, R2, R1;
TEX R0.xyz, f[TEX4], TEX4, 2D;
MULR R2.xyz, R2, R0;
TEX R1, f[TEX1], TEX1, 2D;
MADR R1.xyz, R2, f[COL0], R1;
TEX R0.xyz, f[TEX0], TEX0, 2D;
MULR R0.xyz, R0, R1;
TEX R1.xyz, f[TEX5], TEX5, 2D;
MULR_m2 R0.xyz, R0, R1;
END

# Passes = 6 

# Registers = 3 

# Textures = 6 
