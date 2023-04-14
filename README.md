Game of Life in Vulkan (Dynamic Rendering)
===

Just some more learning. Uses VK_KHR_dynamic_rendering and compute shaders

GOL World / patterns are just arrays of bools that are combined together and then "uploaded" into a simulator which generates the textures / images on the GPU.

Simulator essentially just tells the GPU to transform the texture layout for compute, and then to transform it again for rendering/presentation. All GOL rules are in shader.comp and if the world has dimensions multiples of 16 it will correctly wrap as if the world is a torus.

Since the render is just a single texture on a square the world size can be anything and the gpu will scale the texture properly (super cool, still magic to me)