!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX R2, f[TEX0], TEX0, 2D;
TEX R3, f[TEX1], TEX1, 2D;
MADR R1, f[COL1], {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3R R1, R2, R1;
ADDR R0, R1, C0;
MULR R1, f[COL0], R0;
MULR R0.xyz, R1, R3;
MOVR R0.w, R3;
MOVR o[COLR], R0; 
END

# Passes = 7 

# Registers = 4 

# Textures = 2 
