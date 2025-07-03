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
add r1.xy, -r0.z, c[2].wzzw
max r1.w, r1.x, c[0].x
rcp r2.w, r1.y
dp4 r1.y, r0, c[10]
mul r1.w, r1.w, r2.w
dp4 oPos.w, r0, c[11]
mul r0.w, r1.y, r1.w
mov oPos.z, r1.y
mad oFog, -r0.w, c[16].w, c[16].y
add oT3.xyz, -r0, c[2]
add r1.xyz, -r0, c[29]
add r2.xyz, -r0, c[34]
dp3 r0.x, r1, r1
mov r6.z, c[0].y
mad r1.w, r0.x, -c[31].w, r6.z
rsq r4.y, r0.x
mov r4.xz, c[0].y
mov r0.yw, c[0].y
max r1.w, r1.w, c[0].x
mul r0.xyz, r0.yxxw, r4
min r2.w, r1.w, c[0].y
dp3 r4.x, r0, c[31]
mul r0.xyz, r1, r4.y
rcp r1.w, r4.x
mul r1.w, r2.w, r1.w
dp3 r4.x, c[28], -r0
mul r1.xyz, r1.w, c[27]
add r1.w, r4.x, -c[30].z
dp3 r6.x, r3, r0
mul r1.w, r1.w, c[30].w
max r2.w, r1.w, c[0].x
slt r0.xyz, r3, c[0].x
mova a0.xyz, r0
mul r0.xyz, r3, r3
pow r1.w, r2.w, c[30].x
mul r4.xyz, r0.y, c[a0.y+23]
min r1.w, r1.w, c[0].y
mad r4.xyz, r0.x, c[a0.x+21], r4
mul r1.xyz, r1, r1.w
mad r5.xyz, r0.z, c[a0.z+25], r4
max r1.w, r6.x, c[0].x
dp3 r0.z, r2, r2
rsq r4.y, r0.z
mov r4.xz, c[0].y
mad r1.xyz, r1, r1.w, r5
mul r5.xyz, r0.wzzw, r4
mad r0.w, r0.z, -c[36].w, r6.z
dp3 r0.x, r5, c[36]
mul r2.xyz, r2, r4.y
rcp r1.w, r0.x
max r0.w, r0.w, c[0].x
dp3 r0.x, c[33], -r2
min r2.w, r0.w, c[0].y
add r0.w, r0.x, -c[35].z
mul r1.w, r1.w, r2.w
mul r0.w, r0.w, c[35].w
mul r0.xyz, r1.w, c[32]
max r1.w, r0.w, c[0].x
dp3 r2.x, r3, r2
pow r0.w, r1.w, c[35].x
mov oT6.xyz, r3
min r0.w, r0.w, c[0].y
mul r0.xyz, r0, r0.w
max r0.w, r2.x, c[0].x
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
