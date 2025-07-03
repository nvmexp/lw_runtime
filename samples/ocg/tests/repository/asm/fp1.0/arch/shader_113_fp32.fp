!!FP1.0
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, f[TEX2], TEX2, 2D;
TEX R3, f[TEX3], TEX3, 2D;
MULR R0, R0, R1;
MULR R0, R0, f[COL0];
MULR R0, R0, R2;
MULR R0, R0, R3;
MULR R0, R0, {2, 0, 0, 0}.x; 
MOVR o[COLR], R0; 
END

# Passes = 7 

# Registers = 4 

# Textures = 4 
