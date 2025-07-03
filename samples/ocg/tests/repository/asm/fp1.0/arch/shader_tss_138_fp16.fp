!!FP1.0 
# Pixelshader 138
# Fog: Enabled as Vertex shader fog
# TSS count 2
TXP H1, f[TEX0], TEX0, 2D;
MULH H2.xyz, H1, f[COL0];
MULH H2.xyz, H2, {2, 0, 0, 0}.x; 
MOVH H2.w, f[COL0];
TXP H1, f[TEX1], TEX1, 2D;
MULH H0.xyz, H2, H1;
MOVH H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 2 
