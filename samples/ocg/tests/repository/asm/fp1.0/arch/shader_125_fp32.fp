!!FP1.0 
TEX R2, f[TEX2], TEX2, 2D;
DP3R_SAT R3.x, R2, f[TEX4];
TEX R0, f[TEX3], TEX3, 2D;
MULR R3, R3.x, R0;
DP3R_SAT R0, f[TEX1], f[TEX1];
ADDR R0.x, {1, 1, 1, 1}, -R0.x;
MULR R0, R0.x, R3;
MULR H0, R0, f[COL0];
MULH H0, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 7 

# Registers = 4 

# Textures = 4 
