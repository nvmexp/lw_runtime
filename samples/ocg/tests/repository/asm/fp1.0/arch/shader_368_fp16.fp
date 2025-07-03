!!FP1.0
DECLARE C0={-0.000975, 0.000975, 0.000000, 0.000000};
DECLARE C1={0.000975, -0.000975, 0.000000, 0.000000};
ADDH H0.xy, f[TEX0], C0.x;
ADDH H1.xy, f[TEX0], C1;
ADDH H2.xy, f[TEX0], C0;
ADDH H3.xy, f[TEX0], C0.y;
TEX H0, H0, TEX0, 2D;
TEX H1, H1, TEX0, 2D;
TEX H2, H2, TEX0, 2D;
TEX H3, H3, TEX0, 2D;
MOVH H0.y, H1.x;
MOVH H0.z, H2.x;
MOVH H0.w, H3.x;
MOVH o[COLH], H0; 
END

# Passes = 9 

# Registers = 2 

# Textures = 1 
