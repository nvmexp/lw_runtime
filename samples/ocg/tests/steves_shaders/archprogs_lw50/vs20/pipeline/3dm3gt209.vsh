vs_1_1
def c20, 0, .5, 1, 2
; vertex has only x,y,z,u,v
; x,y are 0 and 1 at four corners of quad
; u,v are 0 and 1 at four corners or quad or tile
; z is always 1
; near, far planes set to .5 and 1.5
; c0-3 has world-view-proj matrix
; c4/5.z have nearz/farz
; c6.xy has render target size
; c7.xy has target texture size
; c9 has draw counter (0..n) - reset at clear
dcl_position v0
dcl_texcoord0 v1
m4x4 oPos, v0, c0
mov oD0.rgba,v0.xyxy
; c10 has size of texture 0
mov oT0.xy,v1.xy
; c11 has size of texture 1
mov oT1.xyz,v1.xyx
; c12 has size of texture 2
mov oT2.xy,v1.xy
; c13 has size of texture 3
;mov oT3.xyzw,v1.xyxy
; c14 has size of texture 4
mov oT4.xyz,v1.xyx
; c15 has size of texture 5
mov oT5.xy,v1.xy
