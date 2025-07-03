!!FP1.0
TEX R4, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, f[TEX2], TEX2, 2D;
TEX R3, f[TEX3], TEX3, 2D;
ADDR R0, f[COL0], R3;
MADR R0, R4, R0, f[COL1];
MADR R0, R1, R2, R0;
MOVR o[COLR], R0; 
END

# Passes = 7 

# Registers = 5 

# Textures = 4 
