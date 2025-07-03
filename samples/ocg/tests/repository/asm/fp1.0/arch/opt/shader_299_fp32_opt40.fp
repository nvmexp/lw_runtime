!!FP2.0
DECLARE C0={0.5, 0.6, 0.7, 0.8};
TEX R0, f[TEX0], TEX0, 2D;
MULR R0, R0, f[COL0];
MADR H0, R0, C0, R31;
END

# Passes = 2 

# Registers = 32 

# Textures = 1 
