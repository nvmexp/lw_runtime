ps_2_0

dcl t0.rg
dcl t1
dcl_2d s0
texld r0, t0, s0
mul r0, r0, t1
mov oC0, r0
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex 0 -texrange 3
