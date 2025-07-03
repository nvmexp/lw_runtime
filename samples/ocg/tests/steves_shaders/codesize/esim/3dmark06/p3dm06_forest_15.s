ps_3_0

def c4, 1.00000000, 0.00000000, 0.50000000, 0.00000000 ; 0x3f800000 0x000000 0x3f000000 0x000000
dcl_texcoord0 v0.rg
dcl_color0_pp v1.rgb
dcl_texcoord2_pp v2.rgb
dcl_texcoord6 v3.rgb
dcl_texcoord7 v4.rgb
dcl_lwbe s0
dcl_lwbe s1
dcl_2d s2
dcl_2d s3
mul r0.xyz, v4, c3.x
max r2.w, r0_abs.x, r0_abs.y
max r1.w, r2.w, r0_abs.z
rcp r0.w, r1.w
mad r0.z, c2.x, -r0.w, c2.y
mov r1.xyz, -v4
texld r1, r1, s1
mul r0.xyw, r1.xyzx, c4.xxzy
texldl_pp r0, r0, s2
dp3 r0.w, v4, v4
rcp_pp r0.w, r0.w
mov r1.xyz, c0
mul r1.xyz, r1, c1.x
mul r4.xyz, r0.w, r1
texld r1, v4, s0
nrm_pp r2.xyz, v2
nrm_pp r3.xyz, v3
mul r1.xyz, r4, r1
dp3_pp r0.w, r2, r3
mul_pp r0.xyz, r0.x, r1
mad_pp r0.w, r0.w, c4.z, c4.z
mul_pp r1.xyz, r0, r0.w
texld_pp r0, v0, s3
mul_pp r0.xyz, r0, v1
mov_pp oC0.w, r0.w
mul_pp oC0.xyz, r1, r0
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex fff9 -texrange f7 -TexShadowMap 4
