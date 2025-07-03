!!FP1.0
TEX H0, f[TEX0], TEX0, 2D;
MULH H0.w, H0, {0.5, 0.000000, 0.000000, 0.000000}.x;
MULH H0, H0, f[COL0];
MULH H0.xyz, H0, {0.6, 0.000000, 0.000000, 0.000000}.x;
MOVH H0, H0;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 1 
