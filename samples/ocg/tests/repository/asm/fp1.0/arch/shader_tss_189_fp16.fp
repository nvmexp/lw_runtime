!!FP1.0 
# Pixelshader 189
# Fog: Enabled as Vertex shader fog
# TSS count 2
TXP H1, f[TEX0], TEX0, 2D;
MULH H2.xyz, H1, f[COL0];
MULH H2.xyz, H2, {2, 0, 0, 0}.x; 
MULH H2.w, H1, f[COL0];
TXP H1, f[TEX1], TEX1, LWBE;
MADH H0.xyz, H2.w, H1, H2;
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 2 
