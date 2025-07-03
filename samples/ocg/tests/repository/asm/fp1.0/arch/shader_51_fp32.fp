!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={1, 2, 3, 4};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, f[TEX2], TEX2, 2D;
TEX R3, f[TEX3], TEX3, 2D;
DP3R_SAT R5, R0, C0;
MULR R4, R1, R5;
DP3R_SAT R5, R0, C1;
MADR R4, R5, R2, R4;
DP3R_SAT R5, R0, C2;
MADR R4, R5, R3, R4;
MULR R0, R4, C3;
MOVR o[COLR], R0; 
END

# Passes = 10 

# Registers = 6 

# Textures = 4 
