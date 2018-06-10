# rasterizer

This is my second attempt at a software rasterizer (recreating the OpenGL pipeline in code). I follow this guide https://github.com/ssloy/tinyrenderer/wiki in some parts.

I'm using OpenGL to see the image generation in real-time, move the camera and tweak some values with dear imgui (the rasterizer output is uploaded as a texture to
a gpu). Stb_image is used for loading texture resources.
OpenGL / Window initialization stuff is managed by GLFW library and glad generated files.

The only 'external' dependency is GLFW but for windows platform everything needed to compile is checked-in.
No C++ standard library is used.
