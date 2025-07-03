!!FP1.0 
TEX R0, f[TEX0], TEX0, 2D;
DP3R R1.x, f[TEX1], R0;
DP3R R1.y, f[TEX2], R0;
TEX R1, R1, TEX2, 2D;
MOVR o[COLR], R0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 3 
