ps_3_0

dcl_texcoord0 v0.rg
dcl_texcoord2 v1.rgb
dcl_2d s0
mov oC0.xyz, v1
texld r0, v0, s0
mov oC0.w, r0.w
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex 0 -texrange 3
