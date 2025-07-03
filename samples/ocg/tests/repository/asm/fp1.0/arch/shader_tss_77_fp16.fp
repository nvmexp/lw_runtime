!!FP1.0 
# Pixelshader 077
# Fog: Enabled as Vertex shader fog
# TSS count 1
TXP H1, f[TEX0], TEX0, 2D;
ADDH H3.xyz, H1, -{ 1.0, 0.0, 0.4, 0.7 };
MADH H0.xyz, H3, f[COL0].w, { 1.0, 0.0, 0.4, 0.7 };
MOVH H0.w, H1;
MOVH o[COLH], H0; 
END

# Passes = 2 

# Registers = 2 

# Textures = 1 
