!!FP2.0
TEX R1, f[TEX0], TEX0, 2D;
MULR R1.xyz, R1, f[COL0];
TEX R0, f[TEX2], TEX1, 2D;
MULR R0.w, R1, f[COL0];
MULR R0.xyz, R0, R1;
MADR R0.xyz, R0, {1.987654, 0, 0, 0}.x, {1.987654, 0, 0, 0}.y;
END

# Passes = 3 

# Registers = 2 

# Textures = 2 
