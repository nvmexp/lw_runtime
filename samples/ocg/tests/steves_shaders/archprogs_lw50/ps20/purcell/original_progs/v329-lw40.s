vs_1_1
dcl_position0 v0
dcl_texcoord0 v1
dcl_texcoord1 v2
add r0, v0.xzw, c[10].xyz
mul r11.x, r0.x, c[8].x
expp r11.y, r11.x
mul r10.x, r11.y, c[8].y
mov a0.x, r10.x
mov r1.w, c[a0.x+16]
add r10.x, r1.w, r0.y
mul r11.x, r10.x, c[8].x
expp r11.y, r11.x
mul r10.x, r11.y, c[8].y
mov a0.x, r10.x
mov r1, c[a0.x+16]
mov r7, r0
add r7.x, r0, v0.w
mul r11.x, r7.x, c[8].x
expp r11.y, r11.x
mul r10.x, r11.y, c[8].y
mov a0.x, r10.x
mov r2.w, c[a0.x+16]
add r10.x, r2.w, r7.y
mul r11.x, r10.x, c[8].x
expp r11.y, r11.x
mul r10.x, r11.y, c[8].y
mov a0.x, r10.x
mov r2, c[a0.x+16]
expp r6, r0.x
mad r10, r6, -c[9].z, c[9].w
mul r10, r10, r6
mul r6, r10, r6
add r10, r2, -r1
mad r3, r10, r6.y, r1
mov r7, r0
add r7.xy, r0, v0.w
mul r11.x, r7.x, c[8].x
expp r11.y, r11.x
mul r10.x, r11.y, c[8].y
mov a0.x, r10.x
mov r1.w, c[a0.x+16]
add r10.x, r1.w, r7.y
mul r11.x, r10.x, c[8].x
expp r11.y, r11.x
mul r10.x, r11.y, c[8].y
mov a0.x, r10.x
mov r1, c[a0.x+16]
mov r7, r0
add r7.y, r0, v0.w
mul r11.x, r7.x, c[8].x
expp r11.y, r11.x
mul r10.x, r11.y, c[8].y
mov a0.x, r10.x
mov r2.w, c[a0.x+16]
add r10.x, r2.w, r7.y
mul r11.x, r10.x, c[8].x
expp r11.y, r11.x
mul r10.x, r11.y, c[8].y
mov a0.x, r10.x
mov r2, c[a0.x+16]
expp r6, r0.x
mad r10, r6, -c[9].z, c[9].w
mul r10, r10, r6
mul r6, r10, r6
add r10, r1, -r2
mad r4, r10, r6.y, r2
expp r6, r0.y
mad r10, r6, -c[9].z, c[9].w
mul r10, r10, r6
mul r6, r10, r6
add r10, r4, -r3
mad r5, r10, r6.y, r3
mov r0, v0
mad r0.y, r5.x, c[10].w, r0.y
m4x4 oPos, r0, c[0]
add oT0, v1, c[11]
m4x4 oT1, r0, c[4]
add r0.xyz, c[12], -r0
dp3 r0.w, r0, r0
rsq r0.w, r0.w
mul oT2.xyz, r0, r0.w
add oT3, v1, c[13]
mov oT4, v2
