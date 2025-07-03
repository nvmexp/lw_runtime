!!FP1.0
TEX R0, f[TEX0], TEX0, 2D;
MULR R0.w, R0, {0.5, 0.000000, 0.000000, 0.000000}.x;
MULR R0, R0, f[COL0];
MULR R0.xyz, R0, {0.6, 0.000000, 0.000000, 0.000000}.x;
MOVR H0, R0;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 1 
