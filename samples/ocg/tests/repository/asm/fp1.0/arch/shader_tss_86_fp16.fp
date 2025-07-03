!!FP1.0 
# Pixelshader 086
# Fog: Enabled as Vertex shader fog
# TSS count 2
TXP H2, f[TEX0], TEX0, 2D; # eliminated a MOV
TXP H1, f[TEX1], TEX1, 2D;
ADDH H0.xyz, H1, H2;
# alpha disabled
MOVR H0.w, H2;
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 2 

# Textures = 2 
