ps_3_0
def c9, 0.15000001, 0.03125000, 0.41608700, -0.30338100 ; 0x3e19999a 0x3d000000 0x3ed5095b 0xbe9b54c1
def c10, 0.13519500, 0.22041900, -0.18368299, 0.07725400 ; 0x3e0a708f 0x3e61b585 0xbe3c1765 0x3d9e3758
def c11, -0.25281799, -0.23776400, 1.00000000, 0.00000000 ; 0xbe81715c 0xbe737868 0x3f800000 0x80000000
def c12, 0.06250000, -0.05412700, 0.66291302, -0.03125000 ; 0x3d800000 0xbd5db446 0x3f29b4ab 0xbd000000
def c13, -0.48613599, 0.39774799, -0.39774799, -0.50000000 ; 0xbef8e6d1 0x3ecba5a0 0xbecba5a0 0xbf000000
def c14, 0.57452399, -0.06250000, -0.57452399, 0.10825300 ; 0x3f131401 0xbd800000 0xbf131401 0x3dddb3c0
def c15, -0.62500000, -0.75000000, 0.87500000, 0.18750000 ; 0xbf200000 0xbf400000 0x3f600000 0x3e400000
def c16, 0.50000000, 1.00000000, 0.10000000, 16.00000000 ; 0x3f000000 0x3f800000 0x3dcccccd 0x41800000
def c17, 2.00000000, -1.00000000, 0.20000000, 1.00000000 ; 0x40000000 0xbf800000 0x3e4ccccd 0x3f800000
def c18, 0.31830987, -0.89999998, 10.00000000, 0.00000000 ; 0x3ea2f983 0xbf666666 0x41200000 0x000000
def c19, -2.00000000, 3.00000000, 0.00000000, 0.00000000 ; 0xc0000000 0x40400000 0x000000 0x000000
def c20, 0.15000001, 0.30000001, 0.40000001, 0.00000000 ; 0x3e19999a 0x3e99999a 0x3ecccccd 0x000000
dcl_texcoord0 v0.rg
dcl_texcoord1 v1.rgb
dcl_texcoord2 v2.rgb
dcl_texcoord3 v3.rgb
dcl_texcoord4 v4.rgb
dcl_texcoord6 v5.rgb
dcl_texcoord7 v6.rgb
dcl_position1 vPos.rg
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_2d s5
dcl_2d s6
dcl_lwbe s7
dcl_lwbe s8
dsx r0, v6.xyxy
dsy r1, v6.xyxy
add r0, r0_abs, r1_abs
mov r1.w, c2.x
mad_pp r1, r0, r1.w, c1.xyxy
mul r0.xy, vPos, c9.y
texld_pp r0, r0, s1
mul_pp r0, r1, r0.zwxy
mul_pp r1, r0, c9.zzw
mad_pp r1, r0.zwxy, c10.xxy, r1
add r2, r1, v6.xyxy
texldl r1, r2.xyxy, s0
texldl r3, r2.zwzw, s0
mul_pp r2, r0, c10.zzw
mad_pp r2, r0.zwxy, c11.xxy, r2
mov r1.y, r3.x
add r2, r2, v6.xyxy
texldl r3, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mov r1.z, r3.x
mov r1.w, r2.x
add r2, r1, -v6.z
mul_pp r1, r0, c12.yyz
cmp_pp r2, r2, c11.z, c11.w
mad_pp r1, r0.zwxy, c12.wwz, r1
dp4_pp r5.w, r2, c12.x
add r1, r1, v6.xyxy
texldl r2, r1.xyxy, s0
texldl r3, r1.zwzw, s0
mul_pp r1, r0, c13.xxy
mad_pp r1, r0.zwxy, c13.xxz, r1
mov r2.y, r3.x
add r3, r1, v6.xyxy
texldl r1, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mov r2.z, r1.x
mul_pp r1, r0, c14.xxy
mov r2.w, r3.x
mad_pp r1, r0.zwxy, c14.zzw, r1
add r2, r2, -v6.z
add r3, r1, v6.xyxy
texldl r1, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mov r1.y, r3.x
mad r3, r0, c15.xxy, v6.xyxy
texldl r4, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mov r1.z, r4.x
mov r1.w, r3.x
cmp_pp r2, r2, c11.z, c11.w
add r1, r1, -v6.z
dp4 r2.w, r2, c12.x
cmp_pp r1, r1, c11.z, c11.w
add_pp r2.w, r5.w, r2.w
dp4 r1.w, r1, c12.x
add_pp r3.w, r2.w, r1.w
mad r2, r0, c15.zzw, v6.xyxy
mad r1, r0.zwzw, c16.xxy, v6.xyxy
texldl r0, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mov r0.y, r2.x
texldl r2, r1.xyxy, s0
texldl r1, r1.zwzw, s0
mov r0.z, r2.x
mul r2.xy, v0, c3
texld r2, r2, s4
mad_pp r5.xyz, c17.x, r2.wyzw, c17.y
nrm r8.xyz, v1
mov r0.w, r1.x
dp3_pp r1.w, -r8, r5
add r0, r0, -v6.z
add_pp r1.w, r1.w, r1.w
mad_pp r1.xyz, r5, -r1.w, -r8
nrm_pp r9.xyz, v5
cmp_pp r0, r0, c11.z, c11.w
dp3 r1.w, r9, r1
dp4 r0.z, r0, c12.x
add r0.w, r1.w, c18.y
add_pp r0.y, r3.w, r0.z
mul_sat r0.z, r0.w, c18.z
mad r0.w, r0.z, c19.x, c19.y
mul r0.z, r0.z, r0.z
mul_pp r2, r0.y, c0.xyz
mul r0.w, r0.w, r0.z
mul r1, r2, r0.w
texld r0, v0.xyx, s3
mul_pp r0, r0.xyz, c6.xyz
dp3_pp r4.w, r5, r8
mul r3, r1, r0
add_sat_pp r6.y, -r4.w, c17.w
mul_pp r0, r0, r0
mul_pp r1.w, r6.y, r6.y
mad_sat_pp r5.w, r1.w, r1.w, c9.x
dp3_pp r1.w, r5, r9
mul r3, r3, r5.w
mov_sat_pp r6.xzw, r1.w
texldl r1, r6, s6
mul r4.xy, v0, c4
texld r4, r4, s5
mad_pp r4.xyz, c17.x, r4.wyzw, c17.y
rcp_pp r1.w, r4.z
mul r4.xy, r4, r1.w
mul r4.xy, r4, c17.z
rcp_pp r1.w, r5.z
mad r4.xy, r5, r1.w, r4
mov_pp r4.z, c17.w
dp3 r1.w, r4, r4
rsq r7.z, r1.w
mul_pp r7.xy, r4, r7.z
dp3_sat_pp r1.w, r7, r9
add_pp r6.w, -r5.w, c17.w
mul_pp r1, r1.xyz, r1.w
mul r5, r3, c20.xyz
mul_pp r1, r2, r1
texld r3, v0.xyx, s2
mul_pp r4, r3.xyz, c5.xyz
dp3_sat_pp r3.z, r8, -r9
mad_pp r3.z, r3.z, r6.y, c13.w
mad_pp r3.y, r3.w, -c16.z, c16.y
mul_pp r1, r1, r4
mad_pp r3.z, r3.z, r3.y, -c13.w
mul_pp r1, r1, c18.x
mul_pp r2, r2, r3.z
mad_pp r1, r1, r6.w, r5
mul_pp r2, r2, c7.xyz
mul_sat_pp r5.w, r9.z, c16.w
mul_pp r2, r2, c8.x
mul_pp r3, r3.w, r2
dp3_pp r2.x, r7, v2
dp3_pp r2.y, r7, v3
dp3_pp r2.z, r7, v4
texld_pp r2, r2, s7
dp3 r2.w, -r8, r7
add r5.z, r2.w, r2.w
mul_pp r2, r4, r2.xyz
mad r5.xyz, r7, -r5.z, -r8
dp3_sat r6.w, r7, r8
dp3_pp r4.x, r5, v2
dp3_pp r4.y, r5, v3
dp3_pp r4.z, r5, v4
texld_pp r4, r4, s8
add r4.w, -r6.w, c17.w
mul r4.w, r4.w, r4.w
mad_pp r0, r0, r4.xyz, -r2
mad_sat_pp r4.w, r4.w, r4.w, c9.x
mad_pp r1, r1, r5.w, r3
mad_pp r0, r4.w, r0, r2
add oC0, r1, r0
