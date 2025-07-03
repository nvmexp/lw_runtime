!!FP1.0
TEX R0, f[TEX2], TEX1, 2D;
TEX R1, f[TEX0], TEX0, 2D;
MULR R1.xyz, R1, f[COL0];
MULR R1.w, R1, f[COL0];
MULR R1.xyz, R0, R1;
MULR H0.xyz, R1, {0.4, 0.000000, 0.000000, 0.000000}.x;
MOVR H0.w, R1;
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 2 
