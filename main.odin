package main

import "gol"
import "core:fmt"
import "core:thread"
import "core:runtime"

main :: proc() {
    // gol.init()

    fmt.println("yay")

    /*
        Ignore for a second that it's Vulkan under the covers. We need:
        1. Generate a new world grid e.g. 800x600 grid
        2. Place seed somewhere in the grid (and allow adding seeds to sections of the grid)
        3. Generate a texture of the world grid with input
        4. Simulate the world
        5. Simulating the world consists of:
        5a. Compute new texture from the previous texture based on GOL rules
        5b. Write the new texture into a second texture
        5c. Draw full screen size square using texture to draw
        5d. present to screen
        5e. swap the texture/screen buffers for next step

        Some additional points:
        1. Need to run renderer / render loops on a separate thread as GLFW will block the main thread
        when resizing. Unfortunate but it's just how it is
        2. We still want to be able to submit commands from the main thread so we need command buffers for each
        thread that is going to do the write
        3. Will put those buffers into an array and those buffers can be looked up using a WriteHandle
    */
    writer := gol.register_writer()

    world := gol.world_create(16, 16)
    defer gol.world_destroy(world)

    pattern := gol.world_create(4, 4, context.temp_allocator)
    positions := [][2]int{
        {0, 1},
        {1, 1},
        {2, 1},
        {1, 2},
        {2, 2},
        {3, 3},
    }
    gol.world_set_many(&pattern, positions)

    gol.world_add(&world, pattern, {3, 3})

    gol.world_print(world)

    next_world := gol.world_create(16, 16)
    defer gol.world_destroy(next_world)

    current := &world
    next := &next_world
    for i in 0..<6 {
        gol.world_step(current, next)
        gol.world_print(next^)
        current, next = next, current
    }
}