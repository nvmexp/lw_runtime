vs_2_0
def c[1], 765.005859, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_blendweight0 v1
dcl_blendindices0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_tangent0 v6
mul r0.xyz, v2.zyxw, c[1].x
mova a0.xyz, r0
add r1.w, -v1.x, c[0].y
mul r0, v1.y, c[a0.y+42]
add r3.w, r1.w, -v1.y
mad r0, c[a0.x+42], v1.x, r0
mad r0, c[a0.z+42], r3.w, r0
dp3 r4.x, v6, r0
mul r2, v1.y, c[a0.y+43]
mul r1, v1.y, c[a0.y+44]
mad r2, c[a0.x+43], v1.x, r2
mad r1, c[a0.x+44], v1.x, r1
mad r2, c[a0.z+43], r3.w, r2
mad r1, c[a0.z+44], r3.w, r1
dp3 r4.y, v6, r2
dp3 r4.z, v6, r1
dp3 r3.x, v3, r0
dp4 r0.x, v0, r0
dp3 r3.y, v3, r2
dp3 r3.z, v3, r1
dp4 r0.y, v0, r2
mul r2.xyz, r4.yzxw, r3.zxyw
dp4 r0.z, v0, r1
mad r1.xyz, r3.yzxw, r4.zxyw, -r2
mov oT4.xyz, r4
mov r0.w, c[0].y
mul oT5.xyz, r1, v6.w
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 r1.y, r0, c[10]
dp4 oPos.w, r0, c[11]
mad oFog, -r1.y, c[16].w, c[16].x
mov oPos.z, r1.y
add r2.xyz, -r0, c[29]
add oT3.xyz, -r0, c[2]
dp3 r5.x, r2, r2
add r0.xyz, -r0, c[34]
rsq r4.y, r5.x
mov r5.y, c[0].y
mov r4.xz, c[0].y
mul r1.xyz, r5.yxxw, r4
mov r5.z, c[0].y
mad r1.w, r5.x, -c[31].w, r5.z
mul r2.xyz, r2, r4.y
dp3 r1.x, r1, c[31]
rcp r0.w, r1.x
max r1.w, r1.w, c[0].x
min r1.w, r1.w, c[0].y
dp3 r1.x, c[28], -r2
mul r1.w, r0.w, r1.w
add r0.w, r1.x, -c[30].z
mul r1.xyz, r1.w, c[27]
mul r0.w, r0.w, c[30].w
dp3 r5.x, r3, r2
max r1.w, r0.w, c[0].x
pow r0.w, r1.w, c[30].x
slt r2.xyz, r3, c[0].x
mova a0.xyz, r2
mul r2.xyz, r3, r3
min r0.w, r0.w, c[0].y
mul r4.xyz, r2.y, c[a0.y+23]
mul r1.xyz, r1, r0.w
mad r4.xyz, r2.x, c[a0.x+21], r4
mad r4.xyz, r2.z, c[a0.z+25], r4
dp3 r2.z, r0, r0
max r1.w, r5.x, c[0].x
rsq r0.w, r2.z
mad r1.xyz, r1, r1.w, r4
mul r0.xyz, r0, r0.w
dst r2.xy, r2.z, r0.w
dp3 r2.x, c[36], r2
mad r1.w, r2.z, -c[36].w, r5.z
rcp r0.w, r2.x
max r1.w, r1.w, c[0].x
min r1.w, r1.w, c[0].y
dp3 r0.x, r3, r0
mul r0.w, r0.w, r1.w
max r1.w, r0.x, c[0].x
mov oT6.xyz, r3
mul r0.xyz, r1.w, c[32]
mad oD0.xyz, r0, r0.w, r1
mul r0.xy, v4, c[90]
add r0.w, r0.y, r0.x
mul r0.xy, v4, c[91]
mov oT0.x, r0.w
add r0.z, r0.y, r0.x
mov oT0.y, r0.z
mov oT1.x, r0.w
mov oT2.x, r0.w
mov oT1.y, r0.z
mov oT2.y, r0.z
mov oD0.w, c[0].x
mov oD1, c[0].x
mov oT7.xyz, c[0].x
