ps_3_0

def c9, 8.00000000, -0.01872930, 0.07426100, -0.21211439 ; 0x41000000 0xbc996e30 0x3d981627 0xbe593484
def c10, 2.00000000, -1.00000000, 1.00000000, 3.00000000 ; 0x40000000 0xbf800000 0x3f800000 0x40400000
def c11, 10000.00000000, 0.00010001, 1.00000000, 1.00100005 ; 0x461c4000 0x38d1bc5b 0x3f800000 0x3f8020c5
def c12, -0.79719388, 0.01456723, 1.00000000, 0.03125000 ; 0xbf4c14e6 0x3c6eab60 0x3f800000 0x3d000000
def c13, 0.66291302, 0.87500000, -0.57452399, 0.57452399 ; 0x3f29b4ab 0x3f600000 0xbf131401 0x3f131401
def c14, 1.00000000, 0.00000000, 0.06250000, -0.48613599 ; 0x3f800000 0x000000 0x3d800000 0xbef8e6d1
def c15, -0.75000000, -0.48613599, -0.62500000, 0.39774799 ; 0xbf400000 0xbef8e6d1 0xbf200000 0x3ecba5a0
def c16, 0.50000000, 0.41608700, 0.13519500, 0.07725400 ; 0x3f000000 0x3ed5095b 0x3e0a708f 0x3d9e3758
def c17, 0.22041900, -0.25281799, -0.30338100, -0.18368299 ; 0x3e61b585 0xbe81715c 0xbe9b54c1 0xbe3c1765
def c18, -0.23776400, 0.18750000, 0.10825300, -0.03125000 ; 0xbe737868 0x3e400000 0x3dddb3c0 0xbd000000
def c19, -0.06250000, -0.05412700, 0.00000001, 0.15915494 ; 0xbd800000 0xbd5db446 0x322bd517 0x3e22f983
def c20, 1.57072878, 0.63661975, -1.00999999, -1.12000000 ; 0x3fc90da4 0x3f22f983 0xbf8147ae 0xbf8f5c29
def c21, 0.31830987, 16.00000000, 0.00000000, 0.00000000 ; 0x3ea2f983 0x41800000 0x000000 0x000000
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
dcl_lwbe s7
dcl_lwbe s8
dsx r0, v7.xyxy
dsy r1, v7.xyxy
add r0, r0_abs, r1_abs
mov r1.w, c3.x
mad_pp r1, r0, r1.w, c2.xyxy
mul r0.xy, vPos, c12.w
texld_pp r0, r0, s1
mul_pp r2, r1, r0.zwxy
mul_pp r0.zw, r2, c13.x
mad_pp r0.zw, r2.xyxy, c13.x, r0
mov_pp r0.xy, r2.zwzw
add r0, r0, v7.xyxy
texldl r1, r0.xyxy, s0
texldl r3, r0.zwzw, s0
mul_pp r0, r2, c13.yyz
mad_pp r0.zw, r2.xyxy, c13.w, r0
mov r1.y, r3.x
add r0, r0, v7.xyxy
texldl r3, r0.xyxy, s0
texldl r0, r0.zwzw, s0
mov r1.z, r3.x
mov r1.w, r0.x
mul_pp r0, r2.zwxy, c15.xxy
mad_pp r0.zw, r2, c14.w, r0
add r1, r1, -v7.z
add r3, r0, v7.xyxy
texldl r0, r3.xyxy, s0
texldl r4, r3.zwzw, s0
mul_pp r3, r2, c15.zzw
mad_pp r3.zw, r2.xyxy, -c15.w, r3
mov r0.y, r4.x
add r3, r3, v7.xyxy
texldl r4, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mov r0.z, r4.x
mov r0.w, r3.x
cmp_pp r1, r1, c14.x, c14.y
add r0, r0, -v7.z
dp4_pp r1.w, r1, c14.z
cmp_pp r0, r0, c14.x, c14.y
dp4 r1.z, r0, c14.z
mul_pp r0, r2.zwxy, c16.xxy
mad_pp r0.zw, r2, c16.z, r0
add_pp r5.w, r1.w, r1.z
add r0, r0, v7.xyxy
texldl r1, r0.xyxy, s0
texldl r3, r0.zwzw, s0
mul_pp r0, r2, c17.xxy
mad_pp r0, r2.zwxy, c17.zzw, r0
mov r1.y, r3.x
add r0, r0, v7.xyxy
texldl r3, r0.xyxy, s0
texldl r0, r0.zwzw, s0
mov r1.z, r3.x
mov r1.w, r0.x
mul_pp r0, r2, c18.xxy
mad_pp r0.xy, r2.zwzw, c16.w, r0
add r1, r1, -v7.z
add r3, r0, v7.xyxy
texldl r0, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mul_pp r4, r2, c18.zzw
mad_pp r2, r2.zwxy, c19.xxy, r4
mov r0.y, r3.x
add r2, r2, v7.xyxy
texldl r3, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mov r0.z, r3.x
mov r0.w, r2.x
cmp_pp r1, r1, c14.x, c14.y
add r0, r0, -v7.z
dp4 r1.w, r1, c14.z
cmp_pp r0, r0, c14.x, c14.y
add_pp r1.w, r5.w, r1.w
dp4 r0.w, r0, c14.z
add_pp r1.w, r1.w, r0.w
mov r0.xyz, c0
mul r0, r0.xyz, c1.x
mul_pp r0, r1.w, r0
mov r1.w, c5.x
add r2.w, -r1.w, c6.x
texld r1, v0.xyx, s2
mad_pp r5.w, r1.w, r2.w, c5.x
nrm_pp r4.xyz, v6
nrm_pp r7.xyz, v2
add_pp r1.w, -r5.w, c10.z
add_pp r2.xyz, r4, r7
rcp_pp r1.w, r1.w
nrm_pp r5.xyz, r2
texld r2, v0.xyx, s3
mad_pp r3.xyz, c10.x, r2.wyzw, c10.y
mul_pp r2.w, r1.w, c10.w
dp3_sat_pp r2.z, r3, r5
add r1.w, r2.w, c10.z
pow r3.w, r2.z, r2.w
mul_pp r6, r1.xyz, c4.xyz
mul r1.w, r1.w, r3.w
log_pp r3.w, r2.w
mul_sat r1.w, r1.w, c19.w
mul r1, r0, r1.w
dp3_sat_pp r7.w, r3, r4
mad_pp r2.w, r7.w, c9.y, c9.z
add_pp r2.z, -r7.w, c10.z
mad_pp r2.w, r2.w, r7.w, c9.w
rsq_pp r2.z, r2.z
mad_pp r2.w, r2.w, r7.w, c20.x
rcp_pp r2.z, r2.z
mul_pp r0, r0, r6
mul_pp r2.w, r2.w, r2.z
mul_sat_pp r9.w, r4.z, c21.y
mad r2.xy, r2.w, c20.y, c20.wzzw
mul r2.xy, r2, r2
dp3_sat_pp r4.w, r3, r7
mad_pp r2.w, r4.w, c9.y, c9.z
add_pp r2.z, -r4.w, c10.z
mad_pp r2.w, r2.w, r4.w, c9.w
rsq_pp r2.z, r2.z
mad_pp r2.w, r2.w, r4.w, c20.x
rcp_pp r2.z, r2.z
rcp r4.x, r2.x
rcp r4.y, r2.y
mul_pp r2.w, r2.w, r2.z
add r2.z, r4.x, c12.x
mad r2.xy, r2.w, c20.y, c20.zwzw
add r2.w, -r4.y, c11.x
mul r2.xy, r2, r2
mul r2.z, r2.z, c12.y
rcp r2.x, r2.x
rcp r2.y, r2.y
mul r2.z, r2.w, r2.z
add r2.x, -r2.x, c11.x
add r2.w, r2.y, c12.x
mul_pp r2.y, r5.w, r5.w
mul r2.z, r2.z, r2.x
mad_pp r10.w, r5.w, -r2.y, c10.z
mul r4.w, r2.x, c11.y
add_pp r4.xy, -r10.w, c11.zwzw
mov_pp r2.y, c10.z
mad_pp r10.z, c7.x, -r5.w, r2.y
mul r2.z, r2.z, r4.y
mad_sat r5.w, r2.z, c19.z, r4.x
mad r4.z, r2.w, -c12.y, c12.z
add_pp r2, r6, c10.y
mul r4.z, r4.z, c7.x
mul r1, r1, r5.w
mad r5, r4.z, r2, c10.z
mad_sat_pp r10.y, r4.y, r4.w, r4.x
mul r1, r1, r5
texld r2, v1.xyx, s4
mul_pp r0, r0, r10.z
mul r4, r1, r2.w
mul_pp r0, r10.w, r0
dp3_pp r1.x, r7, v3
dp3_pp r1.y, r7, v4
dp3_pp r1.z, r7, v5
dp3_pp r9.x, r3, v3
dp3_pp r9.y, r3, v4
dp3_pp r9.z, r3, v5
mul_pp r0, r2.x, r0
dp3_pp r1.w, -r1, r9
mul_pp r0, r7.w, r0
add_pp r2.z, r1.w, r1.w
add_pp r1.w, -r3.w, c9.x
mad_pp r1.xyz, r9, -r2.z, -r1
texldl_pp r7, r1, s6
texldl_pp r1, r1, s8
lrp_pp r3, c8.x, r1.xyz, r7.xyz
texld_pp r8, r9, s5
texld_pp r7, r9, s7
lrp_pp r1, c8.x, r7.xyz, r8.xyz
mul_pp r3, r10.y, r3
mul_pp r1, r6, r1
mul r3, r5, r3
mul_pp r1, r10.z, r1
mul r3, r2.w, r3
mul_pp r1, r10.w, r1
mad_pp r0, r0, c21.x, r4
mad_pp r1, r1, r2.x, r3
mad oC0, r0, r9.w, r1
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex fffc -texrange 3fff8
