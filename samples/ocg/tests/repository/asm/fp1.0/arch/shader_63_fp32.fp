!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX R2, f[TEX0], TEX0, 2D;
TEX R3, f[TEX1], TEX1, 2D;
TEX R4, f[TEX2], TEX2, 2D;
TEX R5, f[TEX3], TEX3, 2D;
DP3R R1, R2, C0;
MULR R0, R3, R1;
DP3R R1, R2, C1;
MADR R0, R4, R1, R0;
DP3R R1, R2, C2;
MADR R0, R5, R1, R0;
MULR H0, R0, f[COL0];
MULH H0, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 10 

# Registers = 6 

# Textures = 4 
