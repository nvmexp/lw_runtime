ps_3_0

def c5, 0.00000000, 0.00000000, 0.00000000, 0.00000000 ; 0x000000 0x000000 0x000000 0x000000
def c6, 2.00000000, -1.00000000, 1.00000000, 16.00000000 ; 0x40000000 0xbf800000 0x3f800000 0x41800000
dcl_texcoord0 v0.rg
dcl_texcoord1 v1.rg
dcl_texcoord2_pp v2.rgb
dcl_texcoord3_pp v3.rgb
dcl_texcoord4_pp v4.rgb
dcl_texcoord5_pp v5.rgb
dcl_texcoord6 v6.rgb
dcl_texcoord7 v7
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_lwbe s5
dcl_2d s6
rcp r0.w, v7.w
mul r0.xyz, r0.w, v7
mov r0.w, c6.z
texldl_pp r1, r0, s0
dp3_pp r0.w, v5, v5
rsq_pp r1.w, r0.w
nrm_pp r5.xyz, v6
mov r0, c0
mul r0, r0, c1.x
mad_pp r2.xyz, v5, r1.w, r5
mul_pp r3, r1.x, r0
nrm_pp r1.xyz, r2
texld r0, v0.xyx, s3
mad_pp r4.xyz, c6.x, r0.wyzw, c6.y
mov r2.zw, c5.x
dp3_pp r2.x, r4, r1
texld_pp r0, v0.xyx, s2
mul_pp r2.y, r0.w, c4.x
mul_pp r1.xyz, r0, c3
texldl_pp r0, r2, s6
mov_pp r1.w, r2.y
mul_pp r0, r3, r0.x
dp3_pp r2.w, r4, r5
mul_pp r0, r1, r0
mul_sat_pp r5.w, r2.w, c6.w
mov_sat_pp r4.w, r2.w
dp3_pp r1.x, r4, v2
dp3_pp r1.y, r4, v3
dp3_pp r1.z, r4, v4
texld_pp r2, r1, s5
texld_pp r1, v1.xyx, s4
mul_pp r4, r3, r4.w
texld_pp r3, v0.xyx, s1
mul_pp r3, r3, c2
mul_pp r4, r4, r3
mul_pp r2, r2, r1
mul_pp r4, r1, r4
mul_pp r0, r0, r5.w
mad r2, r2, r3, r4
mad oC0, r0, r1, r2
;Auto options added
;#PASM_OPTS: -srcalpha 0 -fog 0 -signtex ffbe -texrange 1fff -TexShadowMap 1
