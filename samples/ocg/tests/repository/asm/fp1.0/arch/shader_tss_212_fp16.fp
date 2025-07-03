!!FP1.0 
# Pixelshader 212
# Fog: Enabled as Linear vertex fog
# TSS count 1
TXP H1, f[TEX0], TEX0, 2D;
MULH H0.xyz, H1, f[COL0];
MOVH H0.w, H1;
MOVH o[COLH], H0; 
END

# Passes = 2 

# Registers = 2 

# Textures = 1 
