!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={0.25, 0.2, 1, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX R0, f[TEX0], TEX3, 2D;
MADR R0, R0, C1.x, C1.y;
TEX R1, f[TEX3], TEX3, 2D;
MADR R1, R1, C1.x, C1.y;
MOVR R3, f[TEX1];
ADDR R0, R0, R1;
MULR R0, R0, C2.x;
DP3R R1.x, R0, f[TEX2];
ADDR R1.y, R1.x, R1.x;
MADR R0, R1.y, R0, -f[TEX2];
TEX R0, R0, TEX7, LWBE;
ADDR R1.x, C1.w, -R1;
LG2R R1.x, R1.x;
MULR R1.x, C1.z, R1.x;
EX2R R1.x, R1.x;
TEX R2, f[TEX0], TEX0, 2D;
MADR R2, R2, C1.x, C1.y;
DIVR R3, R3, R3.w;
MADR R2, R2, C0, R3;
TEX R3, R2, TEX1, 2D;
TEX R2, R2, TEX2, 2D;
ADDR R0, R0, R3;
TEX R3, f[TEX4], TEX5, 2D;
MULR R1.x, R1.x, R3.w;
MADR R2, R1.x, -R2, R2;
MADR R0, R1.x, R0, R2;
MULR H0, R0, R3;
END

# Passes = 18 

# Registers = 4 

# Textures = 5 
