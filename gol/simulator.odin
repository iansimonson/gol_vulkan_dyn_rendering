package gol

import vk "vendor:vulkan"

Simulator :: struct {
    width, height: int,
    world_buffers: [2]vk.Image,
}

simulator_verts := [?][2]f32{
    {-1, -1}, {1, -1}, {1, 1}, {-1, 1}
}

simulator_texture_coords := [?][2]f32{
    {0, 0}, {1, 0}, {1, 1}, {0, 1},
}

simulator_indices := [?]u32{
    0, 1, 3,
    1, 2, 3,
}

simulator_create :: proc(world: World) -> Simulator {
    expanded_grid := make([dynamic][4]f32, len(world.grid))
    for cell, i in world.grid {
        value: f32 = 1.0 if cell else 0.0
        expanded_grid[i] = {value, value, value, 1.0}
    }
    image := Image{
        width = world.width,
        height = world.height,
        data = expanded_grid[:],
    }
    front := renderer_create_texture(global_renderer, image)
    back := renderer_create_texture(global_renderer, image)

    return Simulator{width = world.width, height = world.height, world_buffers = {front, back}}
}

simulator_destroy :: proc(simulator: Simulator) {
    renderer_destroy_texture(global_renderer, simulator.world_buffers[0])
    renderer_destroy_texture(global_renderer, simulator.world_buffers[1])
}