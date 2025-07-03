!!FP2.0 
TEX R3, f[TEX3], TEX3, 2D;
TEX R1, f[TEX4], TEX4, 2D;
DP3R_SAT R3.xyz, R3, R1;
MULR R3.xyz, R3, R3;
TEX R0, f[TEX0], TEX0, 2D;
MULR R3.xyz, R3, R0.w;
TEX R1, f[TEX5], TEX5, 2D;
MULR R3.xyz, R3, R1;
TEX R2, f[TEX2], TEX2, 2D;
MULR R1.xyz, R0, R2;
TEX H1, f[TEX1], TEX1, 2D;
MADR R1.xyz, R3, f[COL0], R1;
MOVR H0, f[COL0];
MULR_m2 H0.xyz, R1, H1;
END

# Passes = 8 

# Registers = 4 

# Textures = 6 
