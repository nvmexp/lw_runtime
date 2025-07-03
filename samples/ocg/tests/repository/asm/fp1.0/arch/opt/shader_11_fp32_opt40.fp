!!FP2.0 
TEX R0, f[TEX1], TEX1, 2D;
NRMH R1.xyz, f[TEX0];
DP3R_SAT R1, R0, R1;
TEX R3, f[TEX2], TEX2, 2D;
NRMH R2.xyz, f[TEX0];
DP3R_SAT R2, R0, R2;
TEX H0, R2, TEX8, 1D;
TXP R2, f[TEX3], TEX3, 2D;
MULR R3, R3, R2;
TEX H1, f[TEX5], TEX5, 2D;
MULR H2, R3, R1;
TEX H3, f[TEX4], TEX4, 2D;
MADR H0, H1, H0, H3;
MULB H0.xyz, H2_B, H0_B;
END

# Passes = 6 

# Registers = 4 

# Textures = 6 
