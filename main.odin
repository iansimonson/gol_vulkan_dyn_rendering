package main

import "gol"
import "core:fmt"
import "core:thread"
import "core:runtime"
import "core:time"

import "vendor:glfw"

global_stop: bool

main :: proc() {

    gol.init()
    defer gol.destroy()

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
        2. We still want to be able to submit commands from the main thread so we need command buffers/pools for each
        thread that is going to do the write
        3. Will put those buffers into an array and those buffers can be looked up using a WriteHandle
    */
    world := gol.world_create(1600, 1600)
    defer gol.world_destroy(world)

    pattern := gol.world_create(13, 13, context.temp_allocator)
    // positions := [][2]int{ // Pulsar
    //     {2, 0}, {3, 0}, {4, 0}, {8, 0}, {9, 0}, {10, 0},
    //     {0, 2}, {5, 2}, {7, 2}, {12, 2},
    //     {0, 3}, {5, 3}, {7, 3}, {12, 3},
    //     {0, 4}, {5, 4}, {7, 4}, {12, 4},
    //     {2, 5}, {3, 5}, {4, 5}, {8, 5}, {9, 5}, {10, 5},
    //     {2, 7}, {3, 7}, {4, 7}, {8, 7}, {9, 7}, {10, 7},
    //     {0, 8}, {5, 8}, {7, 8}, {12, 8},
    //     {0, 9}, {5, 9}, {7, 9}, {12, 9},
    //     {0, 10}, {5, 10}, {7, 10}, {12, 10},
    //     {2, 12}, {3, 12}, {4, 12}, {8, 12}, {9, 12}, {10, 12},
    // }
    positions := [][2]int{ // Glider
        {1, 0},
        {2, 1},
        {0, 2}, {1, 2}, {2, 2},
    }
    gol.world_set_many(&pattern, positions)

    // Add a bunch of pattern into world
    for i in 0..<10000 {
        gol.world_add(&world, pattern, {(i * 5 + i) % world.width, (i * 10 + i / 2) % world.height })
        gol.world_add(&world, pattern, {(i * 15) % world.width, (i * 10) % world.height })
    }

    simulator := gol.simulator_create(world)
    defer gol.simulator_destroy(simulator)

    simulator_thread := thread.create(proc(t: ^thread.Thread) {
        simulator := (^gol.Simulator)(t.user_args[0])
        gol.simulator_init_from_thread(simulator)

        for !global_stop {
            gol.simulator_step(simulator)
            free_all(context.temp_allocator)
            time.sleep(10 * time.Millisecond)
        }
    })
    simulator_thread.user_args[0] = &simulator
    thread.start(simulator_thread)

    for !global_stop {
        free_all(context.temp_allocator)
        glfw.WaitEvents()
        if glfw.WindowShouldClose(gol.global_renderer.window) {
            global_stop = true
        }
    }
    thread.destroy(simulator_thread)

    /*
    After it's working:
    input handled as commands put into a queue
    Get the texture and convert to a World
    update the state and reencode as a texture / reupload to the gpu
    continue simulating

    Simulate at x frames per second (configurable? maybe imgui later)

    Buttons for play/pause/reset simulation
    */

}