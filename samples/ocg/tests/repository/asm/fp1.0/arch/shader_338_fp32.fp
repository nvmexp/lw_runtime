!!FP1.0
DECLARE C0= {0.21, 0.36, 0.54, 0.45};
DECLARE C1= {0.54, 0.65, 0.54, 0.34};
DECLARE C2= {0.6, 0.5, 0.7, 0.9};
DECLARE C3= {0.3, 0.2, 0.5, 0.6};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, f[TEX2], TEX2, 2D;
TEX R3, f[TEX3], TEX3, 2D;
DP3R_SAT R4, R0, C0;
MULR R0, R1, R4;
DP3R_SAT R1, R0, C1;
MADR R0, R1, R2, R0;
DP3R_SAT R1, R0, C2;
MADR R0, R1, R3, R0;
MULR H0, R0, C3;
MOVH o[COLH], H0; 
END

# Passes = 10 

# Registers = 5 

# Textures = 4 
