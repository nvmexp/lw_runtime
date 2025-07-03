!!FP2.0
DECLARE C0={-0.000975, 0.000975, 0.000000, 0.000000};
DECLARE C1={0.000975, -0.000975, 0.000000, 0.000000};
MOVH H3.xy, f[TEX0];
ADDH H0.xy, H3, C0.x;
TEX H0, H0, TEX0, 2D;
ADDH H1.xy, H3, C1;
TEX H1, H1, TEX0, 2D;
ADDH H2.xy, H3, C0.y;
TEX H2, H2, TEX0, 2D;
MOVH H0.w, H2.x;
ADDH H2.xy, H3, C0;
TEX H2, H2, TEX0, 2D;
MOVH H0.y, H1.x;
MOVH H0.z, H2.x;
END

# Passes = 5 

# Registers = 2 

# Textures = 1 
