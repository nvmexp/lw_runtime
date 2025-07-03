ps_3_0

def c6, -0.01872930, 0.07426100, 1.57072878, 10000.00000000 ; 0xbc996e30 0x3d981627 0x3fc90da4 0x461c4000
def c7, 1.00000000, 1.00100005, -0.79719388, 0.01456723 ; 0x3f800000 0x3f8020c5 0xbf4c14e6 0x3c6eab60
def c8, 0.03125000, 0.66291302, 0.87500000, -0.57452399 ; 0x3d000000 0x3f29b4ab 0x3f600000 0xbf131401
def c9, 2.00000000, -1.00000000, 1.00000000, -0.21211439 ; 0x40000000 0xbf800000 0x3f800000 0xbe593484
def c10, 1.00000000, 0.00000000, 0.06250000, -0.48613599 ; 0x3f800000 0x000000 0x3d800000 0xbef8e6d1
def c11, -0.75000000, -0.48613599, -0.62500000, 0.39774799 ; 0xbf400000 0xbef8e6d1 0xbf200000 0x3ecba5a0
def c12, 0.50000000, 0.41608700, 0.13519500, 0.07725400 ; 0x3f000000 0x3ed5095b 0x3e0a708f 0x3d9e3758
def c13, 0.22041900, -0.25281799, -0.30338100, -0.18368299 ; 0x3e61b585 0xbe81715c 0xbe9b54c1 0xbe3c1765
def c14, -0.23776400, 0.18750000, 0.10825300, -0.03125000 ; 0xbe737868 0x3e400000 0x3dddb3c0 0xbd000000
def c15, -0.06250000, -0.05412700, 0.00000001, 3.00000000 ; 0xbd800000 0xbd5db446 0x322bd517 0x40400000
def c16, 0.63661975, -1.00999999, -1.12000000, 0.00010001 ; 0x3f22f983 0xbf8147ae 0xbf8f5c29 0x38d1bc5b
def c17, 0.15915494, 0.31830987, 16.00000000, 0.00000000 ; 0x3e22f983 0x3ea2f983 0x41800000 0x000000
dcl_texcoord0 v0.rg
dcl_texcoord1 v1.rg
dcl_texcoord2 v2.rgb
dcl_texcoord3 v3.rgb
dcl_texcoord4 v4.rgb
dcl_texcoord5 v5.rgb
dcl_texcoord6 v6.rgb
dcl_texcoord7 v7.rgb
dcl vPos.rg
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_lwbe s5
dcl_lwbe s6
dsx r0, v7.xyxy
dsy r1, v7.xyxy
add r0, r0_abs, r1_abs
mov r1.w, c2.x
mad_pp r1, r0, r1.w, c1.xyxy
mul r0.xy, vPos, c8.x
texld_pp r0, r0, s1
mul_pp r2, r1, r0.zwxy
mul_pp r0.zw, r2, c8.y
mad_pp r0.zw, r2.xyxy, c8.y, r0
mov_pp r0.xy, r2.zwzw
add r0, r0, v7.xyxy
texldl r1, r0.xyxy, s0
texldl r3, r0.zwzw, s0
mul_pp r0, r2, c8.zzw
mad_pp r0.zw, r2.xyxy, -c8.w, r0
mov r1.y, r3.x
add r0, r0, v7.xyxy
texldl r3, r0.xyxy, s0
texldl r0, r0.zwzw, s0
mov r1.z, r3.x
mov r1.w, r0.x
mul_pp r0, r2.zwxy, c11.xxy
mad_pp r0.zw, r2, c10.w, r0
add r1, r1, -v7.z
add r3, r0, v7.xyxy
texldl r0, r3.xyxy, s0
texldl r4, r3.zwzw, s0
mul_pp r3, r2, c11.zzw
mad_pp r3.zw, r2.xyxy, -c11.w, r3
mov r0.y, r4.x
add r3, r3, v7.xyxy
texldl r4, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mov r0.z, r4.x
mov r0.w, r3.x
cmp_pp r1, r1, c10.x, c10.y
add r0, r0, -v7.z
dp4_pp r1.w, r1, c10.z
cmp_pp r0, r0, c10.x, c10.y
dp4 r1.z, r0, c10.z
mul_pp r0, r2.zwxy, c12.xxy
mad_pp r0.zw, r2, c12.z, r0
add_pp r5.w, r1.w, r1.z
add r0, r0, v7.xyxy
texldl r1, r0.xyxy, s0
texldl r3, r0.zwzw, s0
mul_pp r0, r2, c13.xxy
mad_pp r0, r2.zwxy, c13.zzw, r0
mov r1.y, r3.x
add r0, r0, v7.xyxy
texldl r3, r0.xyxy, s0
texldl r0, r0.zwzw, s0
mov r1.z, r3.x
mov r1.w, r0.x
mul_pp r0, r2, c14.xxy
mad_pp r0.xy, r2.zwzw, c12.w, r0
add r1, r1, -v7.z
add r3, r0, v7.xyxy
texldl r0, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mul_pp r4, r2, c14.zzw
mad_pp r2, r2.zwxy, c15.xxy, r4
mov r0.y, r3.x
add r2, r2, v7.xyxy
texldl r3, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mov r0.z, r3.x
mov r0.w, r2.x
cmp_pp r1, r1, c10.x, c10.y
add r0, r0, -v7.z
dp4 r1.w, r1, c10.z
cmp_pp r0, r0, c10.x, c10.y
add_pp r1.w, r5.w, r1.w
dp4 r0.w, r0, c10.z
add_pp r0.w, r1.w, r0.w
mul_pp r1, r0.w, c0.xyz
texld r2, v0.xyx, s2
mul_pp r5, r2.xyz, c3.xyz
mov r0.w, c4.x
add r2.z, -r0.w, c5.x
mul_pp r0, r1, r5
mad_pp r2.w, r2.w, r2.z, c4.x
add_pp r7.w, -r2.w, c9.z
mul_pp r2.z, r2.w, r2.w
mul_pp r0, r0, r7.w
mad_pp r6.w, r2.w, -r2.z, c9.z
mul_pp r0, r0, r6.w
texld r2, v1.xyx, s4
nrm_pp r4.xyz, v6
nrm_pp r7.xyz, v2
mul_pp r0, r0, r2.x
add_pp r3.xyz, r4, r7
rcp_pp r2.z, r7.w
nrm_pp r6.xyz, r3
texld r3, v0.xyx, s3
mad_pp r8.xyz, c9.x, r3.wyzw, c9.y
mul_pp r3.z, r2.z, c15.w
dp3_sat_pp r3.w, r8, r6
add r2.z, r3.z, c9.z
pow r2.y, r3.w, r3.z
mul r3.w, r2.z, r2.y
dp3_sat_pp r2.y, r8, r4
mad_pp r2.z, r2.y, c6.x, c6.y
add_pp r3.z, -r2.y, c9.z
mad_pp r2.z, r2.z, r2.y, c9.w
rsq_pp r3.z, r3.z
mad_pp r2.z, r2.z, r2.y, c6.z
rcp_pp r3.z, r3.z
mul_sat r3.w, r3.w, c17.x
mul_pp r2.z, r2.z, r3.z
mul r1, r1, r3.w
mad r3.xy, r2.z, c16.x, c16.zyzw
mul_sat_pp r2.z, r4.z, c17.z
mul r3.xy, r3, r3
rcp r3.x, r3.x
rcp r3.y, r3.y
dp3_sat_pp r4.w, r8, r7
mad_pp r3.w, r4.w, c6.x, c6.y
add_pp r3.z, -r4.w, c9.z
mad_pp r3.w, r3.w, r4.w, c9.w
rsq_pp r3.z, r3.z
mad_pp r3.w, r3.w, r4.w, c6.z
rcp_pp r4.w, r3.z
add r3.z, r3.x, c7.z
mul_pp r3.x, r3.w, r4.w
add r3.w, -r3.y, c6.w
mad r3.xy, r3.x, c16.x, c16.yzzw
mul r3.z, r3.z, c7.w
mul r3.xy, r3, r3
mul r3.z, r3.w, r3.z
rcp r3.x, r3.x
rcp r3.y, r3.y
add r3.w, -r3.x, c6.w
add r4.w, r3.y, c7.z
mul r3.z, r3.z, r3.w
add_pp r3.xy, -r6.w, c7
mul r3.w, r3.w, c16.w
mul r3.z, r3.z, r3.y
mad_sat r4.z, r3.z, c15.z, r3.x
mad_sat_pp r8.w, r3.y, r3.w, r3.x
add_pp r3, r5, c9.y
mad r4.w, r4.w, -c7.w, c7.x
mul r1, r1, r4.z
mad r4, r4.w, r3, c9.z
dp3_pp r6.x, r8, v3
dp3_pp r6.y, r8, v4
dp3_pp r6.z, r8, v5
dp3_pp r3.x, r7, v3
dp3_pp r3.y, r7, v4
dp3_pp r3.z, r7, v5
mul r1, r1, r4
dp3_pp r3.w, -r3, r6
mul r1, r2.w, r1
add_pp r3.w, r3.w, r3.w
mad_pp r0, r0, c17.y, r1
mad_pp r3.xyz, r6, -r3.w, -r3
texld_pp r1, r6, s5
texld_pp r3, r3, s6
mul_pp r3, r8.w, r3.xyz
mul_pp r1, r5, r1.xyz
mul r3, r4, r3
mul_pp r1, r7.w, r1
mul r3, r2.w, r3
mul_pp r1, r6.w, r1
mul_pp r0, r2.y, r0
mad_pp r1, r1, r2.x, r3
mad oC0, r0, r2.z, r1
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex ff9c -texrange 17f8 -partialtexld 1400
