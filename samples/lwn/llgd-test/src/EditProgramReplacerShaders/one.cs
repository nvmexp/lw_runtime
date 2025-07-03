#version 440 core
layout (local_size_x = 1) in;
layout (binding = 0) buffer dst_t { uint data; } dst;
void main() { dst.data = 1; }
