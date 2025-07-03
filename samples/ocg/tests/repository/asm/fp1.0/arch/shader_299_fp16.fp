!!FP1.0
DECLARE C0={0.5, 0.6, 0.7, 0.8};
TEX H0, f[TEX0], TEX0, 2D;
MULH H0, H0, f[COL0];
MULH H0, H0, C0;
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 2 

# Textures = 1 
