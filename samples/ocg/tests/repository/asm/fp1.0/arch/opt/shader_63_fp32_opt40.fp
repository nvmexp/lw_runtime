!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX R2, f[TEX0], TEX0, 2D;
DP3R R1, R2, C0;
TEX R0, f[TEX1], TEX1, 2D;
MULR R0, R0, R1;
TEX R3, f[TEX2], TEX2, 2D;
DP3R R1, R2, C1;
MADR R0, R3, R1, R0;
TEX R3, f[TEX3], TEX3, 2D;
DP3R R1, R2, C2;
MADR R0, R3, R1, R0;
MULR_m2 H0, R0, f[COL0];
END

# Passes = 7 

# Registers = 4 

# Textures = 4 
