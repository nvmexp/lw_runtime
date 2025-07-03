!!FP1.0 
DEFINE C1={1.0, 1.0, 1.0, 1.0};
DEFINE C6={6.0, 6.0, 6.0, 6.0};
TEX H0, f[TEX2], TEX1, 2D;
MULH H0.xyz, H0, C1;
TEX H1, f[TEX0], TEX0, 2D;
MULH H1.xyz, H1, f[COL0];
MULH H0.xyz, H0, C6.x;
MULH H0.xyz, H0, H1;
MULH H0.w, H0.w, f[COL0].w;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 1 

# Textures = 2 
