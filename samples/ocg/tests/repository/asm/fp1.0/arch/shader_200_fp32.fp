!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
ADDR R2, C0, -f[TEX2];
DP3R R2.w, R2, R2;
ADDR R2.w, R2.w, C2.y;
RCPR R2.w, R2.w;
MULR R2, C1, R2.w;
ADDR R0, R0, R2;
MADR R3, R1.w, -R0, R0;
MADR R0, R1.w, R1, R3;
MOVR o[COLR], R0; 
END

# Passes = 10 

# Registers = 4 

# Textures = 3 
