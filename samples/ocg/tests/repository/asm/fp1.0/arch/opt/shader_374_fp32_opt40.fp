!!FP2.0
TEX R1, f[TEX0], TEX0, 2D;
MULR R1, R1, f[COL0];
TEX R0, f[TEX2], TEX1, 2D;
MULR R1.xyz, R0, R1;
MULR H0, R1, {0.4, 0.4, 0.4, 1};
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
