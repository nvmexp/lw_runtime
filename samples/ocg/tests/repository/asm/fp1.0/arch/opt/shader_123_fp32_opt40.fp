!!FP2.0 
TEX R1, f[TEX0], TEX0, 2D;
DP3R R0.x, f[TEX1], R1;
DP3R R0.y, f[TEX2], R1;
TEX H0, R0, TEX2, 2D;
END

# Passes = 4 

# Registers = 2 

# Textures = 3 
