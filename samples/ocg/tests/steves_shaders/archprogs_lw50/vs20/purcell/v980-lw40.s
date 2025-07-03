vs_2_0
def c[1], 0.000000, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_texcoord0 v1
dcl_texcoord1 v2
dp3 r0.x, v0, c[4]
add oPos.x, r0.x, c[4].w
dp3 r0.x, v0, c[5]
add oPos.y, r0.x, c[5].w
dp3 r0.x, v0, c[7]
add oPos.w, r0.x, c[7].w
dp3 r0.x, v0, c[42]
add r0.x, r0.x, c[42].w
dp3 r1.x, v0, c[43]
add r0.y, r1.x, c[43].w
dp3 r1.x, v0, c[44]
add r0.z, r1.x, c[44].w
add oT4.xyz, -r0, c[2]
add r0.xy, -r0.z, c[2].wzzw
mul r1.xy, v1, c[90]
add oT0.x, r1.y, r1.x
mul r1.xy, v1, c[91]
add oT0.y, r1.y, r1.x
mul r1.xy, v1, c[92]
add oT1.x, r1.y, r1.x
mul r1.xy, v1, c[93]
add oT1.y, r1.y, r1.x
max r0.w, r0.x, c[0].x
rcp r1.w, r0.y
mul r0.w, r0.w, r1.w
dp3 r0.x, v0, c[6]
add r0.y, r0.x, c[6].w
mul r0.w, r0.w, r0.y
mov oPos.z, r0.y
mad oFog, -r0.w, c[16].w, c[16].y
mov oT2.xy, v2
mov oT2.zw, c[1].x
mov oT3, c[1].x
mov oT5.xyz, c[1].x
mov oT6.xyz, c[1].x
mov oT7.xyz, c[1].x
mov oD0, c[38]
