!!FP1.0 
TEX H1, f[TEX0], TEX0, 2D;
DP3H H0.x, f[TEX1], H1;
DP3H H0.y, f[TEX2], H1;
TEX H0, H0, TEX2, 2D;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 3 
