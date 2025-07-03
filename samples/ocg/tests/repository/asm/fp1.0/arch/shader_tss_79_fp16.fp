!!FP1.0 
# Pixelshader 079
# Fog: Enabled as Vertex shader fog
# TSS count 2
TXP H2, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H0.w, f[TEX1], TEX1, 2D; # eliminated a MOV
MOVR H0.xyz, H2;
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 2 

# Textures = 2 
