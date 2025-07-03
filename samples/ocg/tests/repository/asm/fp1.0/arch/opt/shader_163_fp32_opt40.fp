!!FP2.0 
TEX R0, f[TEX3], TEX3, 2D;
ADDR R0, f[COL0], R0;
TEX R1, f[TEX0], TEX0, 2D;
MULR R0, R1, R0;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, f[TEX2], TEX2, 2D;
MADR R0, R1, R2, R0;
END

# Passes = 4 

# Registers = 3 

# Textures = 4 
