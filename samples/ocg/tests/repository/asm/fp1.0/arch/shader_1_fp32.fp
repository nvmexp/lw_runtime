!!FP1.0
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
MULR H0, R0, R1;
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 2 

# Textures = 2 
