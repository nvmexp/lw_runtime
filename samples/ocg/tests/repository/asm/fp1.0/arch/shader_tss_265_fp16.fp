!!FP1.0 
# Pixelshader 265
# TSS count 1
TXP H1, f[TEX0], TEX0, 2D;
MOVH H0.xyz, { 1.0, 0.0, 0.4, 0.7 };
MOVH H0.w, H1;
MOVH o[COLH], H0; 
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
