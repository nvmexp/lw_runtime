!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
MADR R5, f[COL1], {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3R R3, R0, R5;
ADDR R2, R3, C0;
MULR R3, f[COL0], R2;
MULR H0.xyz, R3, R1;
MOVR H0.w, R1;
MOVH o[COLH], H0; 
END

# Passes = 7 

# Registers = 6 

# Textures = 2 
