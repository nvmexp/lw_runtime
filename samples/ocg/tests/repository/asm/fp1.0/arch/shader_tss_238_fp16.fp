!!FP1.0 
# Pixelshader 238
# TSS count 1
TXP H1, f[TEX0], TEX0, 2D;
MULH H0.xyz, H1, f[COL0];
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MULH H0.w, H1, f[COL0];
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 1 

# Textures = 1 
