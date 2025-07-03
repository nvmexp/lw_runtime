!!FP2.0
DECLARE C0={ 0.2, 0.1, 0.3, 0.2};
DECLARE C1={ 0.5, 0.4, 0.3, 0.1};
DECLARE C2={-0.004000, 1.000000, 0.150000, 0.000000};
DECLARE C3={1.000000, -1.000000, 0.254545, 256.000000};
DECLARE C4={0.500000, -0.500000, 0.000000, -2.000000};
DECLARE C5={2.000000, -1.000000, 0.000000, 0.000000};
MOVR R1.xy, f[TEX1];
RCPR R0.w, f[TEX1].w;
MULR R0.xy, R0.w, R1;
ADDR R0.xy, C3, R0;
MOVR R2.xyz, f[TEX4];
MULR R0.xy, R0, C4;
ADDR R0.xy, R0, C0;
TEX R1, R0, TEX4, 2D;
MULR R1.xyz, C5.x, R1;
DP3R R1.w, R2, R2;
LG2R_d2 R0.z, |R1.w|;
ADDR R3.xyz, C5.y, R1;
EX2R R0.w, -R0.z;
MOVR R1.xyz, f[TEX4];
MULR R2.xyz, R0.w, R1;
MOVR R1.xyz, f[TEX3];
DP3R R0.w, R1, R1;
LG2R_d2 R0.z, |R0.w|;
MOVR R1.xyz, f[TEX3];
EX2R R0.w, -R0.z;
MULR R1.xyz, R0.w, R1;
DP3R R3.w, R3, R1;
MULR R1.w, R3, C4.w;
MULR R3.xyz, R3, R1.w;
ADDR R1.xyz, R1, R3;
DP3R R0.w, R1, -R2;
MOVR R2.xyz, f[TEX2];
SGER R0.z, R0.w, {0, 0, 0, 0}.x;
TEX R1, R0, TEX1, 2D;
ADDR R0.w, R0, -C2.w;
MULR R0.w, R0.z, R0;
ADDR R0.w, C2.w, R0;
MULR R1.w, R1, C3.w;
LG2R R0.w, |R0.w|;
MULR R1.w, R0, R1;
EX2R R1.w, R1.w;
MULR R1.xyz, R1, R1.w;
TEX R0, R0, TEX0, 2D;
ADDR R0.w, R3, -C2.w;
SGER R1.w, R3, {0, 0, 0, 0}.x;
MULR R1.w, R1, R0;
ADDR R0.w, C2.w, R1;
MULR R0.xyz, R0, R0.w;
DP3R R1.w, R2, R2;
ADDR R3, C2.w, -C2.w;
ADDR R0.xyz, R1, R0;
MULR R1.xyz, R0, C1;
LG2R_d2 R0.x, |R1.w|;
EX2R R0.w, -R0.x;
MULR R0.w, R0, R1;
ADDR R2.w, R0, C2.x;
TEX R0, f[TEX2], TEX7, LWBE;
SGER R0, R0, {0, 0, 0, 0};
MULR R0, R0, R3;
ADDR R0, C2.y, R0;
DP4R R0.w, R0, C2.z;
ADDR R0.w, -R0, C2.y;
MULR R1.xyz, R1, R0.w;
SGER R0.w, R1, -{0, 0, 0, 0}.x;
ADDR R1.w, -R1, C2.y;
ADDR R0.z, R1.w, -C2.w;
MULR R0.w, R0, R0.z;
ADDR R1.w, C2.w, R0;
MULR R1.w, R1, R1;
MOVR R0.w, C2.y;
MULR R0.xyz, R1, R1.w;
LG2R R0.x, |R0.x|;
LG2R R0.y, |R0.y|;
LG2R R0.z, |R0.z|;
MULR R0.xyz, R0, C3.z;
EX2R R0.x, R0.x;
EX2R R0.y, R0.y;
EX2R R0.z, R0.z;
END

# Passes = 48 

# Registers = 4 

# Textures = 4 
