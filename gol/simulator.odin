package gol

import "core:slice"
import "core:mem"
import "core:fmt"
import "core:sync"
import "core:container/queue"

import vk "vendor:vulkan"

queue_lock: sync.Mutex // Todo - lockfree
global_queue: queue.Queue(Simulator_Event)

Simulator :: struct {
    width, height: int,
    sampler: vk.Sampler,
    display: Buffer,
    writer: WriterHandle,
    step: int,
    world_buffers: [2]Texture,
    mouse_position: [2]int,
    current_zoom: f64,
    current_vertices: [4][2]f32,
    current_translates: [2]f32,
}

simulator_verts := [?][2]f32{
    {-1, -1}, {1, -1}, {1, 1}, {-1, 1},
}

simulator_texture_coords := [?][2]f32{
    {0, 0}, {1, 0}, {1, 1}, {0, 1},
}

simulator_indices := [?]u32{
    0, 1, 3,
    1, 2, 3,
}

simulator_create :: proc(world: World) -> (simulator: Simulator) {
    expanded_grid := make([dynamic][4]u8, len(world.grid), context.temp_allocator)
    for cell, i in world.grid {
        value := u8(255) if cell else 0
        expanded_grid[i] = {value, value, value, 1}
    }
    image := Image{
        width = world.width,
        height = world.height,
        data = expanded_grid[:],
    }
    front := renderer_create_texture(global_renderer, image)
    back := renderer_create_texture(global_renderer, image)
    sampler := renderer_create_texture_sampler(global_renderer)
    simulator = Simulator{width = world.width, height = world.height, sampler = sampler, world_buffers = {front, back}, current_zoom = 100, current_vertices = simulator_verts}
    simulator_make_descriptors(&simulator)
    load_vertices(&simulator)

    return 
}

simulator_destroy :: proc(simulator: Simulator) {
    defer renderer_destroy_texture(global_renderer, simulator.world_buffers[0])
    defer renderer_destroy_texture(global_renderer, simulator.world_buffers[1])
    defer destroy_buffer(global_renderer.device, simulator.display.buffer, simulator.display.memory)
    defer renderer_destroy_texture_sampler(global_renderer, simulator.sampler)
    defer vk.DeviceWaitIdle(global_renderer.device)
}

// MUST BE CALLED FROM THREAD WHERE THE SIMULATOR WILL RUN
simulator_init_from_thread :: proc(simulator: ^Simulator) {
    simulator.writer = register_writer()
}

global_button_pressed: bool
mouse_pos_previous: [2]int
mouse_pos_now: [2]int

simulator_step :: proc(simulator: ^Simulator) {
    {
        sync.mutex_guard(&queue_lock)
        event: Simulator_Event
        mouse_pos_diff: [2]f32
        for queue.len(global_queue) > 0 {
            new_event := queue.pop_front(&global_queue)
            event.zoom_offset += new_event.zoom_offset * 10
            if new_event.button_down {
                global_button_pressed = true
                event.button_down = true
                mouse_pos_previous = simulator.mouse_position
                mouse_pos_now = mouse_pos_previous
            } else if new_event.button_up {
                global_button_pressed = false
                event.button_up = true
                simulator.mouse_position = mouse_pos_now
                int_diff := mouse_pos_now - mouse_pos_previous
                mouse_pos_diff = [2]f32{f32(int_diff.x), f32(int_diff.y)} / [2]f32{f32(simulator.width), f32(simulator.height)}
                simulator.current_translates += mouse_pos_diff
                mouse_pos_previous = {}
                mouse_pos_now = {}
            } else if global_button_pressed && new_event.mouse_position != {} {
                mouse_pos_previous = mouse_pos_now
                mouse_pos_now = new_event.mouse_position
            }

            if global_button_pressed {
                int_diff := mouse_pos_now - mouse_pos_previous
                mouse_pos_diff = [2]f32{f32(int_diff.x), f32(int_diff.y)} / [2]f32{f32(simulator.width), f32(simulator.height)}
                simulator.current_translates += mouse_pos_diff
            }

        }
        if event != {} {
            simulator.current_zoom += event.zoom_offset
            if simulator.current_zoom < 100.0 {
                simulator.current_zoom = 100.0
            }
            translates: [4][2]f32 = simulator.current_translates
            simulator.current_vertices = simulator_verts * (f32(simulator.current_zoom / 100.0)) + translates


            fmt.println("new vertices:", simulator.current_vertices, "new zoom:", simulator.current_zoom, "from mouse pos:", simulator.mouse_position)
    
            staging_buffer, staging_memory := create_buffer(global_renderer.physical_device, global_renderer.device, vk.DeviceSize(size_of(simulator.current_vertices)), {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT})
            defer destroy_buffer(global_renderer.device, staging_buffer, staging_memory)
    
            staging_data: rawptr
            vk.MapMemory(global_renderer.device, staging_memory, 0, vk.DeviceSize(size_of(simulator.current_vertices)), nil, &staging_data)
            as_bytes := slice.to_bytes(simulator.current_vertices[:])
            mem.copy(staging_data, raw_data(as_bytes), len(as_bytes))
            vk.UnmapMemory(global_renderer.device, staging_memory)
    
            copy_buffer(global_renderer.device, staging_buffer, simulator.display.buffer, []vk.BufferCopy{
                {
                    size = vk.DeviceSize(len(as_bytes)),
                },
            })
        }
    }

    buffer, compute_buffer, image_index := renderer_next_command_buffer(global_renderer, simulator.writer, simulator.step)

    vk_assert(vk.BeginCommandBuffer(compute_buffer, &vk.CommandBufferBeginInfo{
        sType = .COMMAND_BUFFER_BEGIN_INFO,
    }))

    current := int(image_index)
    previous := (current + 1 + len(simulator.world_buffers)) % len(simulator.world_buffers)

    compute_image_barriers := [?]vk.ImageMemoryBarrier2{
        {
            sType = .IMAGE_MEMORY_BARRIER_2,
            oldLayout = .UNDEFINED,
            newLayout = .GENERAL,
            srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
            image = simulator.world_buffers[current].image,
            subresourceRange = {
                aspectMask = {.COLOR},
                baseMipLevel = 0,
                levelCount = 1,
                baseArrayLayer = 0,
                layerCount = 1,
            },
            dstAccessMask = {.SHADER_WRITE},
            dstStageMask = {.COMPUTE_SHADER},
        },
        {
            sType = .IMAGE_MEMORY_BARRIER_2,
            oldLayout = .UNDEFINED,
            newLayout = .GENERAL,
            srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
            image = simulator.world_buffers[previous].image,
            subresourceRange = {
                aspectMask = {.COLOR},
                baseMipLevel = 0,
                levelCount = 1,
                baseArrayLayer = 0,
                layerCount = 1,
            },
            srcAccessMask = nil,
            dstAccessMask = {.SHADER_READ},
            dstStageMask = {.COMPUTE_SHADER},
        },
    }

    compute_dependency_info := vk.DependencyInfo{
        sType = .DEPENDENCY_INFO,
        imageMemoryBarrierCount = len(compute_image_barriers),
        pImageMemoryBarriers = raw_data(compute_image_barriers[:]),
    }

    vk.CmdBindPipeline(compute_buffer, .COMPUTE, global_renderer.compute_pipeline.pipeline)
    vk.CmdPipelineBarrier2(compute_buffer, &compute_dependency_info)
	vk.CmdBindDescriptorSets(compute_buffer, .COMPUTE, global_renderer.compute_pipeline.layout, 0, 1, &global_renderer.descriptors.compute_sets[current], 0, nil)
	vk.CmdDispatch(compute_buffer, u32(simulator.width) / 16, u32(simulator.height) / 16, 1)

    vk_assert(vk.EndCommandBuffer(compute_buffer))

    compute_dst_stage_mask := vk.PipelineStageFlags{.COMPUTE_SHADER}

    vk_assert(vk.QueueSubmit(global_renderer.main_queue, 1, &vk.SubmitInfo{
            sType = .SUBMIT_INFO,
            commandBufferCount = 1,
            pCommandBuffers = &compute_buffer,
            waitSemaphoreCount = 1,
            pWaitSemaphores = &global_renderer.syncs.image_avails[current],
            pWaitDstStageMask = &compute_dst_stage_mask,
            signalSemaphoreCount = 1,
            pSignalSemaphores = &global_renderer.syncs.compute_sems[current],
        }, {}))


    vk_assert(vk.BeginCommandBuffer(buffer, &vk.CommandBufferBeginInfo{
        sType = .COMMAND_BUFFER_BEGIN_INFO,
    }))

    frag_shader_image_barriers := [?]vk.ImageMemoryBarrier2{
        { // swapchain image
            sType = .IMAGE_MEMORY_BARRIER_2,
            dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
            dstStageMask = {.COLOR_ATTACHMENT_OUTPUT},
            oldLayout = .UNDEFINED,
            newLayout = .COLOR_ATTACHMENT_OPTIMAL,
            image = global_renderer.swapchain.images[current],
            subresourceRange = {
                aspectMask = {.COLOR},
                baseMipLevel = 0,
                levelCount =1,
                baseArrayLayer = 0,
                layerCount = 1,
            },
        },
        { // read this world
            sType = .IMAGE_MEMORY_BARRIER_2,
            oldLayout = .UNDEFINED,
            newLayout = .SHADER_READ_ONLY_OPTIMAL,
            srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
            image = simulator.world_buffers[current].image,
            subresourceRange = {
                aspectMask = {.COLOR},
                baseMipLevel = 0,
                levelCount = 1,
                baseArrayLayer = 0,
                layerCount = 1,
            },
            srcAccessMask = {.SHADER_WRITE},
            dstAccessMask = {.SHADER_READ},
            srcStageMask = {.COMPUTE_SHADER},
            dstStageMask = {.FRAGMENT_SHADER},
        },
        { // transition other world as well for syncing
            sType = .IMAGE_MEMORY_BARRIER_2,
            oldLayout = .UNDEFINED,
            newLayout = .SHADER_READ_ONLY_OPTIMAL,
            srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
            image = simulator.world_buffers[previous].image,
            subresourceRange = {
                aspectMask = {.COLOR},
                baseMipLevel = 0,
                levelCount = 1,
                baseArrayLayer = 0,
                layerCount = 1,
            },
            srcAccessMask = {.SHADER_READ},
            dstAccessMask = {.SHADER_READ},
            srcStageMask = {.COMPUTE_SHADER},
            dstStageMask = {.FRAGMENT_SHADER},
        },
    }
    dependency_info := vk.DependencyInfoKHR{
        sType = .DEPENDENCY_INFO_KHR,
        imageMemoryBarrierCount = len(frag_shader_image_barriers),
        pImageMemoryBarriers = raw_data(frag_shader_image_barriers[:]),

    }
    vk.CmdPipelineBarrier2(buffer, &dependency_info)

    clear_value := vk.ClearValue{color = vk.ClearColorValue{float32 = {0, 0, 0, 1}}}
    vk.CmdBeginRenderingKHR(buffer, &vk.RenderingInfoKHR{
        sType = .RENDERING_INFO_KHR,
        renderArea = {
            extent = global_renderer.swapchain.extent,
        },
        layerCount = 1,
        colorAttachmentCount = 1,
        pColorAttachments = &vk.RenderingAttachmentInfoKHR{
            sType = .RENDERING_ATTACHMENT_INFO_KHR,
            imageView = global_renderer.swapchain.image_views[current],
            imageLayout = .ATTACHMENT_OPTIMAL_KHR,
            loadOp = .CLEAR,
            storeOp = .STORE,
            clearValue = clear_value,
        },
    })
    vk.CmdSetViewport(buffer, 0, 1, &vk.Viewport{
        width = f32(global_renderer.swapchain.extent.width),
        height = f32(global_renderer.swapchain.extent.height),
        maxDepth = 1.0,
    })
    vk.CmdSetScissor(buffer, 0, 1, &vk.Rect2D{
        extent = global_renderer.swapchain.extent,
    })

    vk.CmdBindPipeline(buffer, .GRAPHICS, global_renderer.pipeline.pipeline)
    
    vertex_buffers := []vk.Buffer{simulator.display.buffer, simulator.display.buffer}
    offsets := []vk.DeviceSize{0, vk.DeviceSize(len(simulator_verts[:]) * size_of(simulator_verts[0]))}
    index_buffer := simulator.display.buffer
    index_offset := vk.DeviceSize(len(simulator_verts[:]) * size_of(simulator_verts[0]) + len(simulator_texture_coords[:]) * size_of(simulator_texture_coords[0]))
    vk.CmdBindVertexBuffers(buffer, 0, u32(len(vertex_buffers)), raw_data(vertex_buffers), raw_data(offsets))
    vk.CmdBindIndexBuffer(buffer, index_buffer, index_offset, .UINT32)
	vk.CmdBindDescriptorSets(buffer, .GRAPHICS, global_renderer.pipeline.layout, 0, 1, &global_renderer.descriptors.sets[current], 0, nil)
    vk.CmdDrawIndexed(buffer, u32(len(simulator_indices[:])), 1, 0, 0, 0)

    vk.CmdEndRenderingKHR(buffer)

    vk.CmdPipelineBarrier(buffer, {.COLOR_ATTACHMENT_OUTPUT}, {.BOTTOM_OF_PIPE}, nil, 0, nil, 0, nil, 1, &vk.ImageMemoryBarrier{
        sType = .IMAGE_MEMORY_BARRIER,
		srcAccessMask = {.COLOR_ATTACHMENT_WRITE},
		oldLayout = .COLOR_ATTACHMENT_OPTIMAL,
		newLayout = .PRESENT_SRC_KHR,
		image = global_renderer.swapchain.images[current],
		subresourceRange = {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
    })

    vk_assert(vk.EndCommandBuffer(buffer))

    dst_stage_mask := vk.PipelineStageFlags{.FRAGMENT_SHADER}

    vk_assert(vk.QueueSubmit(global_renderer.main_queue, 1, &vk.SubmitInfo{
            sType = .SUBMIT_INFO,
            commandBufferCount = 1,
            pCommandBuffers = &buffer,
            waitSemaphoreCount = 1,
            pWaitSemaphores = &global_renderer.syncs.compute_sems[current],
            pWaitDstStageMask = &dst_stage_mask,
            signalSemaphoreCount = 1,
            pSignalSemaphores = &global_renderer.syncs.render_finishes[current],
    }, global_renderer.syncs.inflight_fences[current]))

    vk_assert(vk.QueuePresentKHR(global_renderer.main_queue, &vk.PresentInfoKHR{
        sType = .PRESENT_INFO_KHR,
        waitSemaphoreCount = 1,
        pWaitSemaphores = &global_renderer.syncs.render_finishes[current],
        swapchainCount = 1,
        pSwapchains = &global_renderer.swapchain.swapchain,
        pImageIndices = &image_index,
        pResults = nil,
    }))

    simulator.step += 1
}

simulator_add_event :: proc(event: Simulator_Event) {

}

vk_assert :: proc(result: vk.Result, loc := #caller_location) {
    if result != .SUCCESS {
        fmt.panicf("Failure performing operation at %v", loc)
    }
}

simulator_make_descriptors :: proc(simulator: ^Simulator) {

    front, back := expand_values(global_renderer.descriptors.sets)
    compute_front, compute_back := expand_values(global_renderer.descriptors.compute_sets)
    { // FRONT
        descriptor_sets := []vk.WriteDescriptorSet{
            {
                sType = .WRITE_DESCRIPTOR_SET,
                dstSet = front,
                dstBinding = 0,
                dstArrayElement = 0,
                descriptorType = .COMBINED_IMAGE_SAMPLER,
                descriptorCount = 1,
                pImageInfo = &vk.DescriptorImageInfo{
                    imageLayout = .SHADER_READ_ONLY_OPTIMAL,
                    imageView = simulator.world_buffers[0].image_view,
                    sampler = simulator.sampler,
                },
            },
        }
        compute_descriptor_sets := []vk.WriteDescriptorSet{
            { // previous frame / read-only
                sType = .WRITE_DESCRIPTOR_SET,
                dstSet = compute_front,
                dstBinding = 0,
                dstArrayElement = 0,
                descriptorType = .STORAGE_IMAGE,
                descriptorCount = 1,
                pImageInfo = &vk.DescriptorImageInfo{
                    imageLayout = .GENERAL,
                    imageView = simulator.world_buffers[1].image_view,
                },
            },
            { // current frame / write-only
                sType = .WRITE_DESCRIPTOR_SET,
                dstSet = compute_front,
                dstBinding = 1,
                dstArrayElement = 0,
                descriptorType = .STORAGE_IMAGE,
                descriptorCount = 1,
                pImageInfo = &vk.DescriptorImageInfo{
                    imageLayout = .GENERAL,
                    imageView = simulator.world_buffers[0].image_view,
                },
            },
        }
        vk.UpdateDescriptorSets(global_renderer.device, u32(len(descriptor_sets)), raw_data(descriptor_sets), 0, nil)
        vk.UpdateDescriptorSets(global_renderer.device, u32(len(compute_descriptor_sets)), raw_data(compute_descriptor_sets), 0, nil)
    }
    { // back
        descriptor_sets := []vk.WriteDescriptorSet{
            {
                sType = .WRITE_DESCRIPTOR_SET,
                dstSet = back,
                dstBinding = 0,
                dstArrayElement = 0,
                descriptorType = .COMBINED_IMAGE_SAMPLER,
                descriptorCount = 1,
                pImageInfo = &vk.DescriptorImageInfo{
                    imageLayout = .SHADER_READ_ONLY_OPTIMAL,
                    imageView = simulator.world_buffers[1].image_view,
                    sampler = simulator.sampler,
                },
            },
        }
        compute_descriptor_sets := []vk.WriteDescriptorSet{
            { // previous frame / read-only
                sType = .WRITE_DESCRIPTOR_SET,
                dstSet = compute_back,
                dstBinding = 0,
                dstArrayElement = 0,
                descriptorType = .STORAGE_IMAGE,
                descriptorCount = 1,
                pImageInfo = &vk.DescriptorImageInfo{
                    imageLayout = .GENERAL,
                    imageView = simulator.world_buffers[0].image_view,
                },
            },
            { // current frame / write-only
                sType = .WRITE_DESCRIPTOR_SET,
                dstSet = compute_back,
                dstBinding = 1,
                dstArrayElement = 0,
                descriptorType = .STORAGE_IMAGE,
                descriptorCount = 1,
                pImageInfo = &vk.DescriptorImageInfo{
                    imageLayout = .GENERAL,
                    imageView = simulator.world_buffers[1].image_view,
                },
            },
        }
        vk.UpdateDescriptorSets(global_renderer.device, u32(len(descriptor_sets)), raw_data(descriptor_sets), 0, nil)
        vk.UpdateDescriptorSets(global_renderer.device, u32(len(compute_descriptor_sets)), raw_data(compute_descriptor_sets), 0, nil)
    }
    
}

load_vertices :: proc(simulator: ^Simulator) {
    pos_bytes, tex_bytes, index_bytes := slice.to_bytes(simulator_verts[:]), slice.to_bytes(simulator_texture_coords[:]), slice.to_bytes(simulator_indices[:])
    total_allocation_size := len(pos_bytes) + len(tex_bytes) + len(index_bytes)

    simulator.display.buffer, simulator.display.memory = create_buffer(global_renderer.physical_device, global_renderer.device, vk.DeviceSize(total_allocation_size), {.TRANSFER_DST, .VERTEX_BUFFER, .INDEX_BUFFER}, {.DEVICE_LOCAL})
    staging_buffer, staging_memory := create_buffer(global_renderer.physical_device, global_renderer.device, vk.DeviceSize(total_allocation_size), {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT})
    defer destroy_buffer(global_renderer.device, staging_buffer, staging_memory)

    staging_data: rawptr
    vk.MapMemory(global_renderer.device, staging_memory, 0, vk.DeviceSize(total_allocation_size), nil, &staging_data)

    pos, tex, ind :=
        staging_data,
        rawptr(uintptr(staging_data) + uintptr(len(pos_bytes))),
        rawptr(uintptr(staging_data) + uintptr(len(pos_bytes) + len(tex_bytes)))
    
    mem.copy(pos, raw_data(pos_bytes), len(pos_bytes))
    mem.copy(tex, raw_data(tex_bytes), len(tex_bytes))
    mem.copy(ind, raw_data(index_bytes), len(index_bytes))

    vk.UnmapMemory(global_renderer.device, staging_memory)

    copy_buffer(global_renderer.device, staging_buffer, simulator.display.buffer, []vk.BufferCopy{
        {
            size = vk.DeviceSize(len(pos_bytes)),
        },
        {
            size = vk.DeviceSize(len(tex_bytes)),
            srcOffset = vk.DeviceSize(len(pos_bytes)),
            dstOffset = vk.DeviceSize(len(pos_bytes)),
        },
        {
            size = vk.DeviceSize(len(index_bytes)),
            srcOffset = vk.DeviceSize(len(pos_bytes) + len(tex_bytes)),
            dstOffset = vk.DeviceSize(len(pos_bytes) + len(tex_bytes)),
        },
    })
}
