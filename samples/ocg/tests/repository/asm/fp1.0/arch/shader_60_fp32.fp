!!FP1.0 
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, f[TEX2], TEX2, 2D;
MADR R3, R1.w, -R0, R0;
MADR R0, R1.w, R1, R3;
MULR R0, f[COL0], R0;
MULR R0, R0, {2, 0, 0, 0}.x; 
ADDR R0.xyz, R0, R2;
MOVR o[COLR], R0; 
END

# Passes = 6 

# Registers = 4 

# Textures = 3 
