vs_2_0
def c[1], 1.000000, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_color0 v1
dcl_texcoord0 v2
mov r0.w, c[1].x
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp4 r1.y, r0, c[10]
add oT3.xyz, -r0, c[2]
add r0.xy, -r0.z, c[2].wzzw
max r0.w, r0.x, c[0].x
rcp r1.w, r0.y
mul r0.w, r0.w, r1.w
mul r0.w, r1.y, r0.w
mov oPos.z, r1.y
mad oFog, -r0.w, c[16].w, c[16].y
dp4 oT0.x, v2, c[90]
dp4 oT0.y, v2, c[91]
dp4 oT1.x, v2, c[92]
dp4 oT1.y, v2, c[93]
dp4 oT2.x, v2, c[94]
dp4 oT2.y, v2, c[95]
mov oT4.xyz, c[1].y
mov oT5.xyz, c[1].y
mov oT6.xyz, c[1].y
mov oD0, v1
mov oD1.xyz, c[1].y
mov oT7.xyz, c[1].y
