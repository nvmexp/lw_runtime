!!FP1.0
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
DP3R R2.x, f[TEX2], R1;
DP3R R2.y, f[TEX3], R1;
TEX R2, R2, TEX3, 2D;
MULR R2, R0, R2;
MULR H0, R2, f[COL0];
MULH H0, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 6 

# Registers = 3 

# Textures = 4 
