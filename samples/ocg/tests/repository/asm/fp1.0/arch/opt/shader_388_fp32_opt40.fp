!!FP2.0
TEX R0, f[TEX0], TEX0, 2D;
MULR R0, R0, f[COL0];
MADR H0, R0, {0.6, 0.6, 0.6, 0.5}.x, R31;
END

# Passes = 2 

# Registers = 32 

# Textures = 1 
