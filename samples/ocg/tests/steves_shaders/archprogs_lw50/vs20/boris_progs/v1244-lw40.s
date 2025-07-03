vs_2_0
def c[1], 765.005859, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_blendweight0 v1
dcl_blendindices0 v2
dcl_texcoord0 v3
mul r0.xyz, v2.zyxw, c[1].x
mova a0.xyz, r0
add r1.w, -v1.x, c[0].y
mul r0, v1.y, c[a0.y+42]
add r3.w, r1.w, -v1.y
mad r0, c[a0.x+42], v1.x, r0
mad r0, c[a0.z+42], r3.w, r0
dp4 r0.x, v0, r0
mul r1, v1.y, c[a0.y+43]
mad r2, c[a0.x+43], v1.x, r1
mul r1, v1.y, c[a0.y+44]
mad r2, c[a0.z+43], r3.w, r2
mad r1, c[a0.x+44], v1.x, r1
dp4 r0.y, v0, r2
mad r1, c[a0.z+44], r3.w, r1
dp4 r0.z, v0, r1
mov r0.w, c[0].y
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
add r1.xy, -r0.z, c[2].wzzw
max r1.w, r1.x, c[0].x
rcp r2.w, r1.y
dp4 r1.y, r0, c[10]
mul r0.w, r1.w, r2.w
add oT3.xyz, -r0, c[2]
mul r0.w, r1.y, r0.w
mov oPos.z, r1.y
mad oFog, -r0.w, c[16].w, c[16].y
mul r1.xy, v3, c[90]
mul r0.xy, v3, c[91]
add r0.w, r1.y, r1.x
add r0.z, r0.y, r0.x
mov oT0.x, r0.w
mov oT0.y, r0.z
mov oT1.x, r0.w
mov oT2.x, r0.w
mov oT1.y, r0.z
mov oT2.y, r0.z
mov oT4.xyz, c[0].x
mov oT5.xyz, c[0].x
mov oT6.xyz, c[0].x
mov oD0, c[0].x
mov oD1, c[0].x
mov oT7.xyz, c[0].x
