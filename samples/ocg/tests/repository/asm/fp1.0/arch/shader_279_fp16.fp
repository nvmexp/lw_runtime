!!FP1.0 
DEFINE C0={0.3, 0.3, 0.3, 0.3};
DEFINE C1={0.5, 0.5, 0.5, 0.5};
DEFINE C2={0.6, 0.6, 0.6, 0.6};
DEFINE C3={0.8, 0.8, 0.8, 0.8};
TEX H1, f[TEX0], TEX0, 2D;
MULH H0, H1, C0;
TEX H1, f[TEX1], TEX1, 2D;
MADR H0, H1, C1, H0;
TEX H1, f[TEX2], TEX2, 2D;
MADR H0, H1, C2, H0;
TEX H1, f[TEX3], TEX3, 2D;
MADR H0, H1, C3, H0;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 1 

# Textures = 4 
