#version 440 core

layout(location = 0) in vec2 coord;
layout(location = 1) in vec2 tc;

out IO { vec2 ftc; };

void main()
{
    gl_Position.xy = coord;
    ftc = tc;
}
