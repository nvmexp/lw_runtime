vs_1_1
; minification factor
def c20, 1.5,1.5,1.5,1.5
def c21, 0, .5, 1, 2

dcl_position v0
dcl_texcoord0 v1 ; uv is 0..1 across full tile

; c6 has size of render target
; c7 has target tile size
; c10 has size of texture 0
; c11 has size of texture 1

m4x4 oPos, v0, c0

; r0.xy gets target tile size * minification factor
mov r0, c7
mul r0.xy, r0, c20

; r1.xy gets target tile size * minification factor / texture size
rcp r1.x, c10.x
rcp r1.y, c10.y
mul r1.xy, r1, r0

; scale uv (0..1) by target tile size * minification factor / texture size
mul oT0.xy, v1, r1

; set lwbe map xy to -1..1 over tile, set z to 1/1.5 = .66666
mad r0, v1, c21.w, -c21.z
rcp r0.z, c20.z
mov oT1.xyz, r0
