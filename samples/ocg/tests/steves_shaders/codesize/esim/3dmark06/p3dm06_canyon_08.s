ps_3_0

dcl_texcoord0 v0.rg
rcp r0.w, v0.y
mul r0.w, r0.w, v0.x
mad oC0, r0.w, c0.x, c0.y
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex 0 -texrange 0
