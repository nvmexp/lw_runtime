ps_3_0

def c3, 1.00000000, 0.50000000, 0.00000000, 0.00000000 ; 0x3f800000 0x3f000000 0x000000 0x000000
dcl_texcoord0 v0.rg
dcl_texcoord1_pp v1.rgb
dcl_texcoord2_pp v2.rgb
dcl_texcoord7 v3
dcl_2d s0
dcl_2d s1
dcl_lwbe s2
rcp r0.w, v3.w
mul r0.xyz, r0.w, v3
mov r0.w, c3.x
texldl_pp r0, r0, s0
nrm_pp r2.xyz, v2
mov r1.xyz, c0
mul r1.xyz, r1, c1.x
dp3_pp r0.w, r2, c2
mul_pp r0.xyz, r0.x, r1
mad_pp r0.w, r0.w, c3.y, c3.y
mul_pp r1.xyz, r0, r0.w
texld_pp r0, v0, s1
mul_pp r2.xyz, r1, r0
texld_pp r1, v1, s2
mad oC0.xyz, r1, r0, r2
mov_pp oC0.w, r0.w
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex fffe -texrange 3f -TexShadowMap 1
