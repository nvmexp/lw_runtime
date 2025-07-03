!!FP1.0 
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, R1.wxxx, TEX2, 2D;
MADR R0, R2, -R0, R0;
MADR H0, R2, R1, R0;
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 3 

# Textures = 2 
