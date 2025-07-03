!!FP1.0 
DECLARE C0 = {0.9, 0.8, 0.7, 0.5};
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
MULH H0, H0, f[COL0];
MULH H0.xyz, H1, H0;
MULH H0.xyz, C0, H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
