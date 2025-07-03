!!FP1.0
TEX H0, f[TEX2], TEX1, 2D;
TEX H1, f[TEX0], TEX0, 2D;
MULH H1.xyz, H1, f[COL0];
MULH H1.w, H1, f[COL0];
MULH H1.xyz, H0, H1;
MULH H1.xyz, H1, {1.987654, 0, 0, 0}.x;
MOVH H0, H1;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 1 

# Textures = 2 
