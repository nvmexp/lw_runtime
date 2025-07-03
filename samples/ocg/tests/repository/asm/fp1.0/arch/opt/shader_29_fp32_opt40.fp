!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={1, 2, 3, 4};
TEX R0, f[TEX0], TEX0, 2D;
DP3R_SAT R3, R0, C0;
TEX R1, f[TEX1], TEX1, 2D;
MULR R1, R1, R3;
DP3R_SAT R3, R0, C1;
TEX R2, f[TEX2], TEX2, 2D;
MADR R1, R3, R2, R1;
TEX R2, f[TEX3], TEX3, 2D;
DP3R_SAT R3, R0, C2;
MADR R1, R3, R2, R1;
MULR H0, R1, C3;
END

# Passes = 7 

# Registers = 4 

# Textures = 4 
