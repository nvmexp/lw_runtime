!!FP1.0 
# Pixelshader 100
# Fog: Enabled as Vertex shader fog
# TSS count 2
TXP H2, f[TEX0], TEX0, 2D; # eliminated a MOV
MULH H0.xyz, H2, { 1.0, 0.0, 0.4, 0.7 };
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MULH H0.w, H2, { 1.0, 0.0, 0.4, 0.7 };
MOVH o[COLH], H0; 
END

# Passes = 2 

# Registers = 2 

# Textures = 1 
