!!FP2.0
TEX R0, f[TEX0], TEX0, 2D;
MULR R0.w, R0, {1.987654, 0, 0, 0}.x;
MULR R0, R0, f[COL0];
MADR H0, R0, {1.304100, 1.173690, 1.056321, 1.0}, R1;
END

# Passes = 3 

# Registers = 2 

# Textures = 1 
