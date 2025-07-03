ps_3_0

def c8, 1.00000000, 0.00000000, 2.00000000, -1.00000000 ; 0x3f800000 0x000000 0x40000000 0xbf800000
def c9, 16.00000000, 0.00000000, 0.00000000, 0.00000000 ; 0x41800000 0x000000 0x000000 0x000000
dcl_texcoord0 v0.rg
dcl_texcoord1 v1.rg
dcl_texcoord5_pp v2.rgb
dcl_texcoord6 v3.rgb
dcl_texcoord7 v4
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_2d s5
dcl_2d s6
cmp_pp r0.w, -v3.z, c8.x, c8.y
if_ne r0.w, -r0.w
mov oC0, c8.y
else
cmp_pp r0.w, -v4.w, c8.y, c8.x
if_ne r0.w, -r0.w
dp3 r0.x, v3, c0
dp3 r0.y, v3, c1
dp3 r0.z, v3, c2
dp3 r0.w, r0, r0
add_sat_pp r2.w, -r0.w, c8.x
rcp r0.z, v4.w
mov r0.w, c8.x
mul r0.xyz, r0.z, v4
texldl_pp r1, r0, s1
mov r0, c3
mul r0, r0, c4.x
mul r0, r2.w, r0
texldp r2, v4, s0
mul r0, r0, r2
mul r5, r1.x, r0
mov_pp r0, r5
else
mov_pp r5, c8.y
mov_pp r0, r5.x
endif
dp3_pp r1.w, v2, v2
rsq_pp r2.w, r1.w
nrm_pp r2.xyz, v3
texld r1, v0.xyx, s4
mad_pp r1.xyz, c8.z, r1.wyzw, c8.w
mad_pp r3.xyz, v2, r2.w, r2
dp3_pp r6.w, r1, r2
nrm_pp r2.xyz, r3
mov_sat_pp r1.w, r6.w
dp3_pp r3.x, r1, r2
mul_pp r0, r0, r1.w
texld_pp r1, v0.xyx, s2
mul_pp r2, r1, c5
texld_pp r4, v0.xyx, s3
mul_pp r3.w, r4.w, c7.x
mov_pp r3.yz, c8.y
texldl_pp r1, r3.xwzy, s6
mul_pp r1, r5, r1.x
mul_pp r3.xyz, r4, c6
mul_pp r1, r3, r1
mul_sat_pp r3.w, r6.w, c9.x
mul_pp r0, r0, r2
mul_pp r2, r1, r3.w
texld_pp r1, v1.xyx, s5
mul_pp r2, r2, r1
mad oC0, r0, r1, r2
endif
;Auto options added
;#PASM_OPTS: -srcalpha 0 -fog 0 -signtex ffbd -texrange 1fff -TexShadowMap 2
