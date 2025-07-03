!!FP1.0 
TEX H0, f[TEX0], TEX0, 2D;
MULH H0.xyz, H0, f[COL0];
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MULH H0.w, H0, f[COL0];
MOVH o[COLH], H0; 
END

# Passes = 2 

# Registers = 1 

# Textures = 1 
