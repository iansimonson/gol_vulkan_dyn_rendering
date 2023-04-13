package gol

import "core:mem"
import "core:fmt"

init :: proc() {
    global_render_init()
}

destroy :: proc() {
    global_render_destroy()
}

World :: struct {
    width, height: int,
    grid: [dynamic]bool,
}

world_create :: proc(width, height: int, allocator := context.allocator) -> (world: World) {
    world.width, world.height = width, height
    world.grid = make([dynamic]bool, width * height, allocator)
    return
}

world_destroy :: proc(world: World) {
    delete(world.grid)
}

// Add an input pattern into the world (represented generally as a smaller world)
// this is essentially doing a copy of a smaller rectangle into the larger rectangle
// this will wrap also as the World is a torus
world_add :: proc(world: ^World, pattern: World, upper_left_corner: [2]int) {
    for cell, i in pattern.grid {
        cell_pos := [2]int{i % pattern.width, i / pattern.width}
        world_pos := (upper_left_corner + cell_pos + [2]int{world.width, world.height}) % [2]int{world.width, world.height}
        world_index := world_pos.y * world.width + world_pos.x
        world.grid[world_index] = cell
    }
}

world_set :: proc(world: ^World, pos: [2]int) {
    grid_index := pos.y * world.width + pos.x
    world.grid[grid_index] = true
}

world_set_many :: proc(world: ^World, positions: [][2]int) {
    for pos in positions {
        world_set(world, pos)
    }
}

world_print :: proc(world: World) {
    using world
    fmt.println("World:")
    for i in 0..<height {
        for j in 0..<width {
            pos := i * width + j
            if grid[pos] {
                fmt.print('X')
            } else {
                fmt.print('.')
            }
        }
        fmt.println()
    }
    fmt.println()
}

world_assert_size :: proc(lhs, rhs: ^World) {
    assert(lhs.width == rhs.width && lhs.height == rhs.height)
    assert(len(lhs.grid) == len(rhs.grid))
}

world_count_neighbors_alive :: proc(world: ^World, pos: [2]int) -> (alive: int) {
    neighbors := [][2]int{{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}}
    for neighbor in neighbors {
        neighbor_pos := (pos + neighbor + [2]int{world.width, world.height}) % [2]int{world.width, world.height}
        alive += int(world.grid[neighbor_pos.y * world.width + neighbor_pos.x])
    }
    return
}

// Runs 1 step of the GOL simulaton on the given World
// and writes it out to another world object
// useful for generating starting seeds?
world_step :: proc(current_world, next_world: ^World) {
    world_assert_size(current_world, next_world)
    for i in 0..<current_world.height {
        for j in 0..<current_world.width {
            grid_index := i * current_world.width + j
            neighbor_count := world_count_neighbors_alive(current_world, {j, i})
            if current_world.grid[grid_index] {
                next_world.grid[grid_index] = (neighbor_count == 2 || neighbor_count == 3)
            } else {
                next_world.grid[grid_index] = (neighbor_count == 3)
            }
        }
    }
}