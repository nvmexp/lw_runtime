!!FP1.0
TEX H0, f[TEX2], TEX1, 2D;
TEX H1, f[TEX0], TEX0, 2D;
MULH H1.xyz, H1, f[COL0];
MULH H1.w, H1, f[COL0];
MULH H1.xyz, H0, H1;
MULH H0.xyz, H1, {0.4, 0.000000, 0.000000, 0.000000}.x;
MOVH H0.w, H1;
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 2 
