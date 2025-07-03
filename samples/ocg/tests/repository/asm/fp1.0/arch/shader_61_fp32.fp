!!FP1.0 
TEX R0, f[TEX0], TEX0, 2D;
MULR R0.xyz, R0, f[COL0];
MULR R0.xyz, R0, {2, 0, 0, 0}.x; 
MULR R0.w, R0, f[COL0];
MOVR o[COLR], R0; 
END

# Passes = 2 

# Registers = 1 

# Textures = 1 
