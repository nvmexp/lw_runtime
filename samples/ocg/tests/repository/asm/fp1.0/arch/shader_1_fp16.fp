!!FP1.0
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
MULH H0, H0, H1;
MOVH o[COLH], H0; 
END

# Passes = 2 

# Registers = 2 

# Textures = 2 
