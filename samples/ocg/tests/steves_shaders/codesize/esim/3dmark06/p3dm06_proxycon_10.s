ps_3_0

def c5, 1.00000000, 2.00000000, -1.00000000, 16.00000000 ; 0x3f800000 0x40000000 0xbf800000 0x41800000
dcl_texcoord0 v0.rg
dcl_texcoord1_pp v1.rgb
dcl_texcoord6 v2.rgb
dcl_texcoord7 v3
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
rcp r0.w, v3.w
mul r0.xyz, r0.w, v3
mov r0.w, c5.x
texldl_pp r1, r0, s0
dp3_pp r0.w, v1, v1
rsq_pp r1.w, r0.w
nrm_pp r2.xyz, v2
mov r0, c0
mul r0, r0, c1.x
mad_pp r4.xyz, v1, r1.w, r2
mul_pp r0, r1.x, r0
nrm_pp r3.xyz, r4
texld r1, v0, s3
mad_pp r1.xyz, c5.y, r1.wyzw, c5.z
dp3_pp r3.x, r1, r3
dp3_pp r3.w, r1, r2
texld_pp r1, v0, s2
mul_pp r3.y, r1.w, c4.x
mul_pp r2.xyz, r1, c3
texld_pp r1, r3, s4
mov_pp r2.w, r3.y
mul_pp r1, r0, r1.x
mul_pp r1, r2, r1
mul_sat_pp r2.w, r3.w, c5.w
mov_sat_pp r3.w, r3.w
mul_pp r2, r1, r2.w
texld_pp r1, v0, s1
mul_pp r1, r1, c2
mul_pp r0, r0, r3.w
mad oC0, r0, r1, r2
;Auto options added
;#PASM_OPTS: -srcalpha 0 -fog 0 -signtex ffee -texrange 1ff -TexShadowMap 1
