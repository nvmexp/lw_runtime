ps_3_0

def c11, 0.03125000, 0.66291302, 0.87500000, -0.57452399 ; 0x3d000000 0x3f29b4ab 0x3f600000 0xbf131401
def c12, -0.75000000, -0.48613599, -0.62500000, 0.39774799 ; 0xbf400000 0xbef8e6d1 0xbf200000 0x3ecba5a0
def c13, 1.00000000, 0.00000000, 0.06250000, -0.48613599 ; 0x3f800000 0x80000000 0x3d800000 0xbef8e6d1
def c14, 0.50000000, 0.41608700, 0.13519500, 0.07725400 ; 0x3f000000 0x3ed5095b 0x3e0a708f 0x3d9e3758
def c15, 0.22041900, -0.25281799, -0.30338100, -0.18368299 ; 0x3e61b585 0xbe81715c 0xbe9b54c1 0xbe3c1765
def c16, -0.23776400, 0.18750000, 0.10825300, -0.03125000 ; 0xbe737868 0x3e400000 0x3dddb3c0 0xbd000000
def c17, -0.06250000, -0.05412700, 16.00000000, 0.31830987 ; 0xbd800000 0xbd5db446 0x41800000 0x3ea2f983
def c18, 2.00000000, -1.00000000, 0.15000001, 1.00000000 ; 0x40000000 0xbf800000 0x3e19999a 0x3f800000
def c19, 50.00000000, 0.00000000, 0.00000000, 0.00000000 ; 0x42480000 0x000000 0x000000 0x000000
dcl_texcoord0 v0.rg
dcl_texcoord2 v1.rgb
dcl_texcoord3 v2.rgb
dcl_texcoord4 v3.rgb
dcl_texcoord5 v4.rgb
dcl_texcoord6 v5.rgb
dcl_texcoord7 v6.rgb
dcl vPos.rg
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
mov r1.w, c3.x
mad_pp r1, r0, r1.w, c2.xyxy
mul r0.xy, vPos, c11.x
texld_pp r0, r0, s1
mul_pp r1, r1, r0.zwxy
mul_pp r0.zw, r1, c11.y
mad_pp r0.zw, r1.xyxy, c11.y, r0
mov_pp r0.xy, r1.zwzw
add r2, r0, v6.xyxy
texldl r0, r2.xyxy, s0
texldl r3, r2.zwzw, s0
mul_pp r2, r1, c11.zzw
mad_pp r2.zw, r1.xyxy, -c11.w, r2
mov r0.y, r3.x
add r2, r2, v6.xyxy
texldl r3, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mov r0.z, r3.x
mov r0.w, r2.x
add r2, r0, -v6.z
mul_pp r0, r1.zwxy, c12.xxy
mad_pp r0.zw, r1, c13.w, r0
cmp_pp r2, r2, c13.x, c13.y
add r3, r0, v6.xyxy
texldl r0, r3.xyxy, s0
texldl r4, r3.zwzw, s0
mul_pp r3, r1, c12.zzw
mad_pp r3.zw, r1.xyxy, -c12.w, r3
mov r0.y, r4.x
add r3, r3, v6.xyxy
texldl r4, r3.xyxy, s0
texldl r3, r3.zwzw, s0
mov r0.z, r4.x
mov r0.w, r3.x
dp4_pp r4.w, r2, c13.z
add r0, r0, -v6.z
cmp_pp r2, r0, c13.x, c13.y
mul_pp r0, r1.zwxy, c14.xxy
mad_pp r0.zw, r1, c14.z, r0
dp4 r4.z, r2, c13.z
add r2, r0, v6.xyxy
texldl r0, r2.xyxy, s0
texldl r3, r2.zwzw, s0
mul_pp r2, r1, c15.xxy
mad_pp r2, r1.zwxy, c15.zzw, r2
mov r0.y, r3.x
add r2, r2, v6.xyxy
texldl r3, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mov r0.z, r3.x
mov r0.w, r2.x
add_pp r4.w, r4.w, r4.z
add r0, r0, -v6.z
cmp_pp r2, r0, c13.x, c13.y
mul_pp r0, r1, c16.xxy
mad_pp r0.xy, r1.zwzw, c14.w, r0
dp4 r4.z, r2, c13.z
add r2, r0, v6.xyxy
texldl r0, r2.xyxy, s0
texldl r2, r2.zwzw, s0
mul_pp r3, r1, c16.zzw
mad_pp r1, r1.zwxy, c17.xxy, r3
mov r0.y, r2.x
add r1, r1, v6.xyxy
texldl r2, r1.xyxy, s0
texldl r1, r1.zwzw, s0
mov r0.z, r2.x
mov r0.w, r1.x
add_pp r2.w, r4.w, r4.z
add r0, r0, -v6.z
cmp_pp r0, r0, c13.x, c13.y
mul r1.xy, v0, c5
texld r1, r1, s5
mad_pp r1.xyz, c18.x, r1.wyzw, c18.y
dp4 r1.w, r0, c13.z
rcp_pp r0.w, r1.z
mul r1.xy, r1, r0.w
mul r0.xy, v0, c4
texld r0, r0, s4
mad_pp r4.xyz, c18.x, r0.wyzw, c18.y
mul r0.xy, r1, c18.z
rcp_pp r0.w, r4.z
mad r1.xy, r4, r0.w, r0
mov_pp r1.z, c18.w
add_pp r1.w, r2.w, r1.w
dp3 r1.z, r1, r1
mov r0, c0
mul r5, r0, c1.x
rsq r7.z, r1.z
mul_pp r7.xy, r1, r7.z
nrm_pp r6.xyz, v5
mul_pp r0, r1.w, r5
dp3_sat_pp r1.w, r7, r6
mul_pp r2, r0, r1.w
texld r1, v0.xyx, s2
mul_pp r3, r1, c6
nrm_pp r8.xyz, r4
mul_pp r4, r2, r3
texld r2, v0.xyx, s6
mul_pp r4, r4, r2
nrm_pp r1.xyz, v4
dp3_pp r5.w, r8, r1
dp3_pp r5.z, r1, -r6
add_sat_pp r5.w, -r5.w, c18.w
mad_sat_pp r5.z, r5.z, c14.x, c14.x
mul_sat_pp r7.w, r6.z, c17.z
mul_pp r6.w, r5.w, r5.z
mul_pp r4, r4, r7.w
pow r5.w, r6.w, c9.x
mul_pp r4, r4, c17.w
mul r5.w, r5.w, c8.x
add_pp r8.xyz, r6, r1
mul r1.z, r5.x, r5.w
mad_pp r4, r1.z, r1.w, r4
dp3_pp r1.x, r7, v1
dp3_pp r1.y, r7, v2
dp3_pp r1.z, r7, v3
texld_pp r6, r1, s7
texld_pp r5, r1, s8
nrm_pp r1.xyz, r8
dp3_sat_pp r1.w, r7, r1
pow_pp r7.z, r1.w, c19.x
lrp_pp r1, c10.x, r5, r6
mul_pp r0, r0, r7.z
texld r5, v0.xyx, s3
mul_pp r5, r5, c7
mul_pp r1, r2, r1
mul_pp r0, r0, r5
mad r1, r1, r3, r4
mul_pp r0, r2, r0
mad oC0, r0, r7.w, r1
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex fffc -texrange 3fff8
