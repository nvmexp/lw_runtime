!!FP1.0 
# Pixelshader 078
# Fog: Enabled as Vertex shader fog
# TSS count 1
TXP H1, f[TEX0], TEX0, 2D;
MULH H0.xyz, f[COL0].w, H1;
MOVH H0.w, f[COL0];
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 2 

# Textures = 1 
