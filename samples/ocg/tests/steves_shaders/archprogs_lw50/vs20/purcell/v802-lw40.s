vs_2_0
def c[9], 0.000000, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
m4x4 oPos, v0, c[0]
m4x3 r0.xyz, v0, c[4]
add oT0.xyz, r0, r0
add r2.xyz, -v0, c[8]
add r9.xyz, -v0, c[7]
nrm r4.xyz, r2
nrm r11.xyz, r9
add oT3.xyz, r4, r11
mov oT2.xyz, r11
mov oT0.w, c[9].x
mov oT1.xy, v2
mov oT1.zw, c[9].x
mov oT2.w, c[9].x
mov oT3.w, c[9].x
mov oT4, v1
