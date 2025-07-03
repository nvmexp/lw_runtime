!!FP2.0
TEX R0, f[TEX0], TEX0, 2D;
MULR R0, R0, f[COL0];
TEX R1, f[TEX1], TEX1, 2D;
MULR R0, R0, R1;
TEX R1, f[TEX2], TEX2, 2D;
MULR R0, R0, R1;
TEX R1, f[TEX3], TEX3, 2D;
MULR_m2 R0, R0, R1;
END

# Passes = 4 

# Registers = 3 

# Textures = 4 
