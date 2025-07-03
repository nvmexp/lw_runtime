!!FP1.0 
# Pixelshader 269
# TSS count 2
TXP H1, f[TEX0], TEX0, 2D;
MULH H0, H1, { 1.0, 0.0, 0.4, 0.7 }; # color & alpha paired
TXP H1, f[TEX1], TEX1, 2D;
MULH H0.xyz, H1, H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MULH H0.w, H1, H0;
MOVH o[COLH], H0; 
END

# Passes = 2 

# Registers = 1 

# Textures = 2 
