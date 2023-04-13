package gol

import "core:sync"
import "core:strings"
import "core:slice"
import "core:fmt"
import "core:runtime"
import "core:mem"

import vk "vendor:vulkan"
import "vendor:glfw"

WriterHandle :: distinct u64
@thread_local _thread_global_handle: WriterHandle
_global_atomic_counter: u64

global_renderer: Renderer

validation_layers := []cstring{"VK_LAYER_KHRONOS_validation"}

vertex_shader :: #load("../shaders/vert.spv")
fragment_shader :: #load("../shaders/frag.spv")
compute_shader :: #load("../shaders/comp.spv")

MAX_FRAMES_IN_FLIGHT :: 2
DEFAULT_THREAD_CAPACITY :: 4
global_command_pools: [dynamic]vk.CommandPool
global_command_buffers: [dynamic][MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer
global_compute_command_pools: [dynamic]vk.CommandPool
global_compute_command_buffers: [dynamic][MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer

register_writer :: proc() -> WriterHandle {
    if _thread_global_handle == {} {
        current_value := sync.atomic_load(&_global_atomic_counter)
        value, swapped := sync.atomic_compare_exchange_strong(&_global_atomic_counter, current_value, current_value + 1)
        for !swapped {
            current_value = sync.atomic_load(&_global_atomic_counter)
            value, swapped = sync.atomic_compare_exchange_strong(&_global_atomic_counter, current_value, current_value + 1)
        }
        _thread_global_handle = WriterHandle(value + 1) // cas returns previous value
        assert(value + 1 <= DEFAULT_THREAD_CAPACITY) // TODO add more pools
    }
    
    return _thread_global_handle
}

global_render_init :: proc() {
    global_command_pools = make([dynamic]vk.CommandPool, DEFAULT_THREAD_CAPACITY)
    global_compute_command_pools = make([dynamic]vk.CommandPool, DEFAULT_THREAD_CAPACITY)
    global_command_buffers = make([dynamic][MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer, DEFAULT_THREAD_CAPACITY)
    global_compute_command_buffers = make([dynamic][MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer, DEFAULT_THREAD_CAPACITY)

    glfw.Init()
    vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))
    glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)

    global_renderer.window = glfw.CreateWindow(WIDTH, HEIGHT, "Hello Dynamic Rendering Vulkan", nil, nil)
	assert(global_renderer.window != nil, "Window could not be crated")

    glfw.SetWindowUserPointer(global_renderer.window, &global_renderer)
	glfw.SetFramebufferSizeCallback(global_renderer.window, framebuffer_resize_callback)


    instance_extension := get_required_instance_extensions()
    defer delete(instance_extension)

    enabled := []vk.ValidationFeatureEnableEXT{.DEBUG_PRINTF}
    features := vk.ValidationFeaturesEXT{
        sType = .VALIDATION_FEATURES_EXT,
        enabledValidationFeatureCount = 1,
        pEnabledValidationFeatures = raw_data(enabled),
    }
    if vk.CreateInstance(&vk.InstanceCreateInfo{
        sType = .INSTANCE_CREATE_INFO,
        enabledExtensionCount = u32(len(instance_extension)),
        ppEnabledExtensionNames = raw_data(instance_extension),
        pApplicationInfo = &vk.ApplicationInfo{
            sType = .APPLICATION_INFO,
            pApplicationName = "GOL in Vulkan",
            applicationVersion = vk.MAKE_VERSION(0, 1, 1),
            pEngineName = "No Engine",
            engineVersion = vk.MAKE_VERSION(0, 1, 0),
            apiVersion = vk.API_VERSION_1_3,
        },
        enabledLayerCount = u32(len(validation_layers)),
        ppEnabledLayerNames = raw_data(validation_layers),
        pNext = &features,
    }, nil, &global_renderer.instance) != .SUCCESS {
        panic("Couldn not create instance")
    }

    get_instance_proc_addr :: proc "system" (
		instance: vk.Instance,
		name: cstring,
	) -> vk.ProcVoidFunction {
		f := glfw.GetInstanceProcAddress(instance, name)
		return (vk.ProcVoidFunction)(f)
	}
	vk.GetInstanceProcAddr = get_instance_proc_addr
	vk.load_proc_addresses(global_renderer.instance)

    vk.CreateDebugUtilsMessengerEXT(global_renderer.instance, &vk.DebugUtilsMessengerCreateInfoEXT{
        sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        messageSeverity = {.VERBOSE, .INFO, .WARNING, .ERROR},
        messageType = {.GENERAL, .VALIDATION, .PERFORMANCE},
        pfnUserCallback = debug_callback,
    }, nil, &global_renderer.debug_messenger)

    if glfw.CreateWindowSurface(global_renderer.instance, global_renderer.window, nil, &global_renderer.surface) != .SUCCESS {
        panic("Could not create surface")
    }

    graphics_family: u32 = 1000
    present_family: u32 = 1000

    { // CREATE LOGICAL DEVICE AND GET QUEUE HANDLE
        device_count: u32
        vk.EnumeratePhysicalDevices(global_renderer.instance, &device_count, nil)
        devices := make([]vk.PhysicalDevice, device_count, context.temp_allocator)
        vk.EnumeratePhysicalDevices(global_renderer.instance, &device_count, raw_data(devices))
        physical_device := devices[0]
        global_renderer.physical_device = physical_device

        queue_family_count: u32
        vk.GetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nil)
        queue_families := make([]vk.QueueFamilyProperties, queue_family_count, context.temp_allocator)
        vk.GetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, raw_data(queue_families))

        for queue_family, i in queue_families {
            if .GRAPHICS in queue_family.queueFlags && .COMPUTE in queue_family.queueFlags {
                graphics_family = u32(i)
                break
            }
        }
        for queue_family, i in queue_families {
            present_support: b32
            vk.GetPhysicalDeviceSurfaceSupportKHR(physical_device, u32(i), global_renderer.surface, &present_support)
            if present_support {
                present_family = u32(i)
                break
            }
        }

        queue_set := make(map[u32]u32, 100, context.temp_allocator)
        queue_set[graphics_family] = 1
        queue_set[present_family] = 1
        assert(len(queue_set) == 1)

        queue_create_infos := make([dynamic]vk.DeviceQueueCreateInfo, 0, len(queue_set), context.temp_allocator)
        priority: f32 = 1.0

        for queue_family, _ in queue_set {
            append(&queue_create_infos, vk.DeviceQueueCreateInfo{
                sType = .DEVICE_QUEUE_CREATE_INFO,
                queueFamilyIndex = queue_family,
                queueCount = 1,
                pQueuePriorities = &priority,
            })
        }

        device_features := vk.PhysicalDeviceFeatures{
            samplerAnisotropy = true,
        }

        device_extensions := make([]cstring, len(DEVICE_EXTENSION_LIST), context.temp_allocator)
        for ext, i in DEVICE_EXTENSION_LIST {
            device_extensions[i] = strings.clone_to_cstring(ext, context.temp_allocator)
        }

        if vk.CreateDevice(physical_device, &vk.DeviceCreateInfo{
            sType = .DEVICE_CREATE_INFO,
            pQueueCreateInfos = raw_data(queue_create_infos[:]),
            queueCreateInfoCount = u32(len(queue_create_infos)),
            pEnabledFeatures = &device_features,
            enabledExtensionCount = u32(len(device_extensions)),
            ppEnabledExtensionNames = raw_data(device_extensions),
            pNext = &vk.PhysicalDeviceDynamicRenderingFeaturesKHR{
                sType = .PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
                dynamicRendering = true,
                pNext = &vk.PhysicalDeviceSynchronization2Features{
                    sType = .PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
                    synchronization2 = true,
                }
            }
        }, nil, &global_renderer.device) != .SUCCESS {
            panic("Could not create logical device!")
        }
        vk.GetDeviceQueue(global_renderer.device, graphics_family, 0, &global_renderer.main_queue)

        // CREATE SWAP CHAIN AND SWAP CHAIN IMAGES/VIEWS
        capabilities: vk.SurfaceCapabilitiesKHR
        vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, global_renderer.surface, &capabilities)

        image_count := capabilities.minImageCount + 1
        if capabilities.maxImageCount > 0 {
            image_count = clamp(image_count, capabilities.minImageCount, capabilities.maxImageCount)
            image_count = clamp(image_count, capabilities.minImageCount, MAX_FRAMES_IN_FLIGHT)
        }

        width, height := glfw.GetFramebufferSize(global_renderer.window)
        extent := vk.Extent2D{
            width = clamp(u32(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            height = clamp(u32(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
        }

        if vk.CreateSwapchainKHR(global_renderer.device, &vk.SwapchainCreateInfoKHR{
            sType = .SWAPCHAIN_CREATE_INFO_KHR,
            surface = global_renderer.surface,
            minImageCount = image_count,
            imageFormat = .B8G8R8A8_SRGB,
            imageColorSpace = .SRGB_NONLINEAR,
            imageExtent = extent,
            imageArrayLayers = 1,
            imageUsage = {.COLOR_ATTACHMENT},
            preTransform = capabilities.currentTransform,
            compositeAlpha = {.OPAQUE},
            presentMode = .FIFO,
            clipped = true,
            oldSwapchain = {},
        }, nil, &global_renderer.swapchain.swapchain) != .SUCCESS {
            panic("Couldn't create swap chain!")
        }

        sc_image_count: u32
        vk.GetSwapchainImagesKHR(global_renderer.device, global_renderer.swapchain.swapchain, &sc_image_count, nil)
        global_renderer.swapchain.images = make([]vk.Image, sc_image_count)
        vk.GetSwapchainImagesKHR(global_renderer.device, global_renderer.swapchain.swapchain, &sc_image_count, raw_data(global_renderer.swapchain.images))
        global_renderer.swapchain.extent = extent
        global_renderer.swapchain.format = .B8G8R8A8_SRGB
        global_renderer.swapchain.image_views = make([]vk.ImageView, sc_image_count)
        for image, i in global_renderer.swapchain.images {
            global_renderer.swapchain.image_views[i] = create_image_view(global_renderer.device, image, .B8G8R8A8_SRGB, {.COLOR}, 1)
        }


    }

    { // DESCRIPTOR SET LAYOUT
        using global_renderer
        graphics_layout := [?]vk.DescriptorSetLayoutBinding{
            {
                binding = 0,
                descriptorCount = 1,
                descriptorType = .COMBINED_IMAGE_SAMPLER,
                stageFlags = {.FRAGMENT},
            },
        }

        compute_layout := [?]vk.DescriptorSetLayoutBinding{
            {
                binding = 0,
                descriptorCount = 1,
                descriptorType = .STORAGE_IMAGE,
                stageFlags = {.COMPUTE},
            },
            {
                binding = 1,
                descriptorCount = 1,
                descriptorType = .STORAGE_IMAGE,
                stageFlags = {.COMPUTE},
            },
        }

        if vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount = u32(len(graphics_layout[:])),
            pBindings = raw_data(graphics_layout[:]),
        }, nil, &descriptors.layout) != .SUCCESS {
            panic("Failed to create descriptor set layout!")
        }
        if vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount = u32(len(compute_layout[:])),
            pBindings = raw_data(compute_layout[:]),
        }, nil, &descriptors.compute_layout) != .SUCCESS {
            panic("Failed to create descriptor set layout!")
        }
    }

    { // PIPELINE
        using global_renderer
        vert_shader_module := create_shader_module(device, vertex_shader)
        frag_shader_module := create_shader_module(device, fragment_shader)
        compute_shader_module := create_shader_module(device, compute_shader)
        defer vk.DestroyShaderModule(device, vert_shader_module, nil)
        defer vk.DestroyShaderModule(device, frag_shader_module, nil)
        defer vk.DestroyShaderModule(device, compute_shader_module, nil)

        shader_stages := [?]vk.PipelineShaderStageCreateInfo{
            {
                sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage = {.VERTEX},
                module = vert_shader_module,
                pName = "main",
            },
            {
                sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage = {.FRAGMENT},
                module = frag_shader_module,
                pName = "main",
            },
        }

        dynamic_states := [?]vk.DynamicState{.VIEWPORT, .SCISSOR}
        dynamic_state := vk.PipelineDynamicStateCreateInfo{
            sType = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount = u32(len(dynamic_states)),
            pDynamicStates = raw_data(dynamic_states[:]),
        }
        binding_descriptions := [?]vk.VertexInputBindingDescription{
            { // inPosition
                binding = 0,
                stride = size_of([2]f32),
                inputRate = .VERTEX,
            },
            { // inTextureCoords
                binding = 1,
                stride = size_of([2]f32),
                inputRate = .VERTEX,
            },
        }
        attribute_descriptions := [?]vk.VertexInputAttributeDescription{
            {
                binding = 0,
                location = 0,
                format = .R32G32_SFLOAT,
            },
            {
                binding = 1,
                location = 1,
                format = .R32G32_SFLOAT,
            },
        }

        vertex_input_info := vk.PipelineVertexInputStateCreateInfo{
            sType = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount = u32(len(binding_descriptions)),
            vertexAttributeDescriptionCount = u32(len(attribute_descriptions)),
            pVertexBindingDescriptions = raw_data(binding_descriptions[:]),
            pVertexAttributeDescriptions = raw_data(attribute_descriptions[:]),
        }

        input_assembly_info := vk.PipelineInputAssemblyStateCreateInfo{
            sType = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology = .TRIANGLE_LIST,
            primitiveRestartEnable = false,
        }

        viewport_state := vk.PipelineViewportStateCreateInfo{
            sType = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount = 1,
            scissorCount = 1,
        }

        rasterizer := vk.PipelineRasterizationStateCreateInfo{
            sType = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            polygonMode = .FILL,
            cullMode = {.FRONT},
            frontFace = .COUNTER_CLOCKWISE,
            lineWidth = 1.0,
        }
        multisampling := vk.PipelineMultisampleStateCreateInfo{
            sType = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterizationSamples = {._1},
            minSampleShading = 1.0,
        }
        color_blend_attachment := vk.PipelineColorBlendAttachmentState {
            colorWriteMask = {.R, .G, .B, .A},
            srcColorBlendFactor = .ONE,
            dstColorBlendFactor = .ZERO,
            colorBlendOp = .ADD,
            srcAlphaBlendFactor = .ONE,
            dstAlphaBlendFactor = .ZERO,
            alphaBlendOp = .ADD,
        }
    
        color_blending := vk.PipelineColorBlendStateCreateInfo {
            sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOp         = .COPY,
            attachmentCount = 1,
            pAttachments    = &color_blend_attachment,
        }

        if vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount = 1,
            pSetLayouts = &descriptors.layout,
        }, nil, &pipeline.layout) != .SUCCESS {
            panic("Error creating pipeline layout")
        }

        if vk.CreateGraphicsPipelines(device, {}, 1, &vk.GraphicsPipelineCreateInfo{
            sType = .GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount = u32(len(shader_stages)),
            pStages = raw_data(shader_stages[:]),
            pVertexInputState = &vertex_input_info,
            pInputAssemblyState = &input_assembly_info,
            pViewportState = &viewport_state,
            pRasterizationState = &rasterizer,
            pMultisampleState = &multisampling,
            pColorBlendState = &color_blending,
            pDynamicState = &dynamic_state,
            layout = pipeline.layout,
            basePipelineIndex = -1,
            pNext = &vk.PipelineRenderingCreateInfoKHR{
                sType = .PIPELINE_RENDERING_CREATE_INFO_KHR,
                colorAttachmentCount = 1,
                pColorAttachmentFormats = &swapchain.format,
            },
        }, nil, &pipeline.pipeline) != .SUCCESS {
            panic("failed to create pipeline")
        }

        vk_assert(vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount = 1,
            pSetLayouts = &descriptors.compute_layout
        }, nil, &compute_pipeline.layout))

        vk_assert(vk.CreateComputePipelines(device, {}, 1, &vk.ComputePipelineCreateInfo{
            sType = .COMPUTE_PIPELINE_CREATE_INFO,
            layout = compute_pipeline.layout,
            stage = vk.PipelineShaderStageCreateInfo{
                sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage = {.COMPUTE},
                module = compute_shader_module,
                pName = "main",
            },
        }, nil, &compute_pipeline.pipeline))
    }

    { // COMMAND POOLS
        using global_renderer
        for pool, i in &global_command_pools {
            if vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{
                sType = .COMMAND_POOL_CREATE_INFO,
                flags = {.RESET_COMMAND_BUFFER},
                queueFamilyIndex = graphics_family,
            }, nil, &pool) != .SUCCESS {
                panic("Failed to create command pool!")
            }

            vk_assert(vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
                sType = .COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool = pool,
                level = .PRIMARY,
                commandBufferCount = MAX_FRAMES_IN_FLIGHT,
            }, raw_data(global_command_buffers[i][:])))
        }
        for pool, i in &global_compute_command_pools {
            vk_assert(vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{
                sType = .COMMAND_POOL_CREATE_INFO,
                flags = {.RESET_COMMAND_BUFFER},
                queueFamilyIndex = graphics_family,
            }, nil, &pool))

            vk_assert(vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
                sType = .COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool = pool,
                level = .PRIMARY,
                commandBufferCount = MAX_FRAMES_IN_FLIGHT,
            }, raw_data(global_compute_command_buffers[i][:])))
        }
    }

    { // Descriptor pool/sets
        using global_renderer
        pool_sizes := []vk.DescriptorPoolSize{
            {
                type = .COMBINED_IMAGE_SAMPLER,
                descriptorCount = u32(MAX_FRAMES_IN_FLIGHT),
            },
            {
                type = .STORAGE_IMAGE,
                descriptorCount = 2 * u32(MAX_FRAMES_IN_FLIGHT),
            },
        }

        if vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
            sType = .DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount = u32(len(pool_sizes)),
            pPoolSizes = raw_data(pool_sizes),
            maxSets = 2 * u32(MAX_FRAMES_IN_FLIGHT),
        }, nil, &descriptors.pool) != .SUCCESS {
            panic("Failed to create descriptor pool!")
        }

        layouts := [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSetLayout{
            descriptors.layout,
            descriptors.layout,
        }
        compute_layouts := [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSetLayout{
            descriptors.compute_layout,
            descriptors.compute_layout,
        }

        vk_assert(vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
            sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool = descriptors.pool,
            descriptorSetCount = u32(MAX_FRAMES_IN_FLIGHT),
            pSetLayouts = raw_data(layouts[:]),
        }, raw_data(descriptors.sets[:])))
        vk_assert(vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
            sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool = descriptors.pool,
            descriptorSetCount = u32(MAX_FRAMES_IN_FLIGHT),
            pSetLayouts = raw_data(compute_layouts[:]),
        }, raw_data(descriptors.compute_sets[:])))
    }

    { // SYNC OBJECTS
        using global_renderer
        for i in 0..<MAX_FRAMES_IN_FLIGHT {
            s1 := vk.CreateSemaphore(device, &vk.SemaphoreCreateInfo{
                sType = .SEMAPHORE_CREATE_INFO,
            }, nil, &syncs.image_avails[i])
            s2 := vk.CreateSemaphore(device, &vk.SemaphoreCreateInfo{
                sType = .SEMAPHORE_CREATE_INFO,
            }, nil, &syncs.render_finishes[i])
            s3 := vk.CreateSemaphore(device, &vk.SemaphoreCreateInfo{
                sType = .SEMAPHORE_CREATE_INFO,
            }, nil, &syncs.compute_sems[i])
            fen := vk.CreateFence(device, &vk.FenceCreateInfo{
                sType = .FENCE_CREATE_INFO,
                flags = {.SIGNALED},
            }, nil, &syncs.inflight_fences[i])
        
            if s1 != .SUCCESS || s2 != .SUCCESS || s3 != .SUCCESS || fen != .SUCCESS {
                panic("failed to create sync objects")
            }
        }
    }
}

global_render_destroy :: proc() {
    defer glfw.Terminate()
    defer glfw.DestroyWindow(global_renderer.window)
    defer vk.DestroyInstance(global_renderer.instance, nil)
    defer vk.DestroyDevice(global_renderer.device, nil)
    defer vk.DestroySwapchainKHR(global_renderer.device, global_renderer.swapchain.swapchain, nil)
    defer delete(global_renderer.swapchain.images)
    defer delete(global_renderer.swapchain.image_views)
    defer vk.DestroyDescriptorSetLayout(global_renderer.device, global_renderer.descriptors.layout, nil)
    defer vk.DestroyPipelineLayout(global_renderer.device, global_renderer.pipeline.layout, nil)
    defer vk.DestroyPipeline(global_renderer.device, global_renderer.pipeline.pipeline, nil)
}

renderer_create_texture :: proc(renderer: Renderer, image: Image) -> Texture {
    image_data_as_bytes := slice.to_bytes(image.data)
    image_bytes_len := vk.DeviceSize(len(image_data_as_bytes))
    staging_buffer, staging_memory := create_buffer(renderer.physical_device, renderer.device, image_bytes_len, {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT})
    defer destroy_buffer(renderer.device, staging_buffer, staging_memory)

    data: rawptr
    vk.MapMemory(renderer.device, staging_memory, 0, image_bytes_len, nil, &data)
    mem.copy(data, raw_data(image_data_as_bytes), int(image_bytes_len))
    vk.UnmapMemory(renderer.device, staging_memory)

    texture, texture_memory := create_image(renderer.physical_device, renderer.device, u32(image.width), u32(image.height), 1, .R8G8B8A8_UNORM, .OPTIMAL, {.TRANSFER_DST, .SAMPLED, .STORAGE}, {.DEVICE_LOCAL})
    
    command_buffer := scoped_single_time_commands(renderer.device, global_command_pools[int(_thread_global_handle)], renderer.main_queue)
	// transition to transfer_dst_optimal
    vk.CmdPipelineBarrier(command_buffer, {.TOP_OF_PIPE}, {.TRANSFER}, {}, 0, nil, 0, nil, 1, &vk.ImageMemoryBarrier{
		sType = .IMAGE_MEMORY_BARRIER,
		oldLayout = .UNDEFINED,
		newLayout = .TRANSFER_DST_OPTIMAL,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = texture,
		subresourceRange = {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		srcAccessMask = nil,
		dstAccessMask = nil,
	})

    // copy staging buffer to texture
	vk.CmdCopyBufferToImage(command_buffer, staging_buffer, texture, .TRANSFER_DST_OPTIMAL, 1, &vk.BufferImageCopy{
		bufferOffset = 0,
		bufferRowLength = 0,
		bufferImageHeight = 0,

		imageSubresource = {
			aspectMask = {.COLOR},
			mipLevel = 0,
			baseArrayLayer = 0,
			layerCount = 1,
		},

		imageOffset = {0, 0, 0},
		imageExtent = {u32(image.width), u32(image.height), 1},

	})

    vk.CmdPipelineBarrier(command_buffer, {.TRANSFER}, {.FRAGMENT_SHADER}, {}, 0, nil, 0, nil, 1, &vk.ImageMemoryBarrier{
		sType = .IMAGE_MEMORY_BARRIER,
		oldLayout = .TRANSFER_DST_OPTIMAL,
		newLayout = .SHADER_READ_ONLY_OPTIMAL,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = texture,
		subresourceRange = {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		srcAccessMask = {.TRANSFER_WRITE},
		dstAccessMask = {.SHADER_READ},
	})

    texture_image_view := create_image_view(renderer.device, texture, .R8G8B8A8_UNORM, {.COLOR}, 1)

    return {image = texture, image_view = texture_image_view, image_memory = texture_memory}
}

renderer_destroy_texture :: proc(renderer: Renderer, texture: Texture) {
    defer vk.FreeMemory(renderer.device, texture.image_memory, nil)
    defer vk.DestroyImage(renderer.device, texture.image, nil)
    defer vk.DestroyImageView(renderer.device, texture.image_view, nil)
}

renderer_create_texture_sampler :: proc(renderer: Renderer) -> (sampler: vk.Sampler) {
    props: vk.PhysicalDeviceProperties
    vk.GetPhysicalDeviceProperties(renderer.physical_device, &props)

    if vk.CreateSampler(renderer.device, &vk.SamplerCreateInfo{
        sType = .SAMPLER_CREATE_INFO,
        magFilter = .NEAREST,
        minFilter = .NEAREST,
        mipmapMode = .NEAREST,
        addressModeU = .REPEAT,
        addressModeV = .REPEAT,
        addressModeW = .REPEAT,
        anisotropyEnable = false,
        maxAnisotropy = props.limits.maxSamplerAnisotropy,
        borderColor = .INT_OPAQUE_BLACK,
        unnormalizedCoordinates = false,
        compareEnable = false,
        compareOp = .ALWAYS,
    }, nil, &sampler) != .SUCCESS {
        panic("Could not create sampler")
    }

    return
}

renderer_next_command_buffer :: proc(renderer: Renderer, writer: WriterHandle, step: int) -> (command_buffer, compute_command_buffer: vk.CommandBuffer, image_index: u32) {
    renderer := renderer
    current_frame := step % MAX_FRAMES_IN_FLIGHT
    vk.WaitForFences(renderer.device, 1, &renderer.syncs.inflight_fences[current_frame], true, max(u64))

    result := vk.AcquireNextImageKHR(renderer.device, renderer.swapchain.swapchain, max(u64), renderer.syncs.image_avails[current_frame], {}, &image_index)
    if result == .ERROR_OUT_OF_DATE_KHR {
        panic("Need to recreate swapchain")
    }
    else if result != .SUCCESS && result != .SUBOPTIMAL_KHR {
        panic("Couldn't get swapchain")
    }
    
    assert(image_index == u32(current_frame))
    vk.ResetFences(renderer.device, 1, &renderer.syncs.inflight_fences[current_frame])

    command_buffer = global_command_buffers[int(writer - 1)][current_frame]
    compute_command_buffer = global_compute_command_buffers[int(writer) - 1][current_frame]
    vk.ResetCommandBuffer(command_buffer, {})
    vk.ResetCommandBuffer(compute_command_buffer, {})

    return
}

renderer_submit_commmand_buffer :: proc(renderer: Renderer, command_buffer: vk.CommandBuffer, image_index: u32) {
    current_frame := int(image_index)
    renderer := renderer
    command_buffer := command_buffer
    image_index := image_index

    dst_stage_mask := vk.PipelineStageFlags{.COMPUTE_SHADER, .COLOR_ATTACHMENT_OUTPUT}

    if vk.QueueSubmit(renderer.main_queue, 1, &vk.SubmitInfo{
        sType = .SUBMIT_INFO,
        waitSemaphoreCount = 1,
        pWaitSemaphores = &renderer.syncs.image_avails[current_frame],
        pWaitDstStageMask = &dst_stage_mask,
        commandBufferCount = 1,
        pCommandBuffers = &command_buffer,
        signalSemaphoreCount = 1,
        pSignalSemaphores = &renderer.syncs.render_finishes[current_frame],
    }, renderer.syncs.inflight_fences[int(image_index)]) != .SUCCESS {
        panic("Failed to submit draw command buffer!")
    }

    vk.QueuePresentKHR(renderer.main_queue, &vk.PresentInfoKHR{
        sType = .PRESENT_INFO_KHR,
        waitSemaphoreCount = 1,
        pWaitSemaphores = &renderer.syncs.render_finishes[current_frame],
        swapchainCount = 1,
        pSwapchains = &renderer.swapchain.swapchain,
        pImageIndices = &image_index,
    })
}

WIDTH :: 800
HEIGHT :: 600

ENABLE_VALIDATION_LAYERS :: ODIN_DEBUG || #config(enable_validation_layers, false)

Render_Context :: struct {
    window: glfw.WindowHandle,
    surface: vk.SurfaceKHR,
    instance: vk.Instance,
    debug_messenger: vk.DebugUtilsMessengerEXT,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    main_queue: vk.Queue,
    swapchain: Swapchain,
    descriptors: Descriptors,
    pipeline: Pipeline,
    compute_pipeline: Pipeline,
    syncs: Syncs,
}

Renderer :: Render_Context

Image :: struct {
    width, height: int,
    data: [][4]u8,
}

Buffer :: struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
}

Texture :: struct {
    image: vk.Image,
    image_memory: vk.DeviceMemory,
    image_view: vk.ImageView,
}

Swapchain :: struct {
    swapchain: vk.SwapchainKHR,
    extent: vk.Extent2D,
    format: vk.Format,
    images: []vk.Image,
    image_views: []vk.ImageView,
}

Descriptors :: struct {
    layout: vk.DescriptorSetLayout,
    compute_layout: vk.DescriptorSetLayout,
    pool: vk.DescriptorPool,
    sets: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSet,
    compute_sets: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSet,
}

Pipeline :: struct {
    layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,
}

Syncs :: struct {
    compute_sems: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore,
    image_avails: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore,
    render_finishes: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore,
    inflight_fences: [MAX_FRAMES_IN_FLIGHT]vk.Fence,
}

get_required_instance_extensions :: proc() -> (result: [dynamic]cstring) {

	extensions := glfw.GetRequiredInstanceExtensions()
	append(&result, ..extensions)

	if ENABLE_VALIDATION_LAYERS {
		append(&result, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)
	}
	return
}

get_device :: proc(instance: vk.Instance) -> vk.Device {
    return {}
}

framebuffer_resize_callback :: proc "c" (window: glfw.WindowHandle, width, height: i32) {
}

DEVICE_EXTENSION_LIST := [?]string{
    vk.KHR_SWAPCHAIN_EXTENSION_NAME,
    vk.KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
    vk.KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    // vk.KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
    // vk.KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
}

create_image :: proc(physical_device: vk.PhysicalDevice, device: vk.Device, width, height, mip_levels: u32, format: vk.Format, tiling: vk.ImageTiling, usage: vk.ImageUsageFlags, properties: vk.MemoryPropertyFlags) -> (image: vk.Image, memory: vk.DeviceMemory) {
	if vk.CreateImage(device, &vk.ImageCreateInfo{
		sType = .IMAGE_CREATE_INFO,
		imageType = .D2,
		extent = vk.Extent3D{width = width, height = height, depth = 1},
		mipLevels = mip_levels,
		arrayLayers = 1,
		format = format,
		tiling = tiling,
		initialLayout = .UNDEFINED,
		usage = usage,
		sharingMode = .EXCLUSIVE,
		samples = {._1},
		flags = nil,
	}, nil, &image) != .SUCCESS {
		panic("Failed to create image!")
	}

	mem_requirements: vk.MemoryRequirements
	vk.GetImageMemoryRequirements(device, image, &mem_requirements)

	if vk.AllocateMemory(device, &vk.MemoryAllocateInfo{
		sType = .MEMORY_ALLOCATE_INFO,
		allocationSize = mem_requirements.size,
		memoryTypeIndex = find_memory_type(physical_device, mem_requirements.memoryTypeBits, properties),
	}, nil, &memory) != .SUCCESS {
		panic("failed to allocate image memory!")
	}

	vk.BindImageMemory(device, image, memory, 0)
	return
}

create_image_view :: proc(device: vk.Device, image: vk.Image, format: vk.Format, aspect_flags: vk.ImageAspectFlags, mip_levels: u32) -> (view: vk.ImageView) {
	if vk.CreateImageView(device, &vk.ImageViewCreateInfo{
		sType = .IMAGE_VIEW_CREATE_INFO,
		image = image,
		viewType = .D2,
		format = format,
		subresourceRange = {
			aspectMask = aspect_flags,
			baseMipLevel = 0,
			levelCount = mip_levels,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}, nil, &view) != .SUCCESS {
		panic("Failed to create texture image")
	}
	return
}

create_shader_module :: proc(device: vk.Device, code: []byte) -> (sm: vk.ShaderModule) {
	if result := vk.CreateShaderModule(
		   device,
		   &vk.ShaderModuleCreateInfo{
			   sType = .SHADER_MODULE_CREATE_INFO,
			   codeSize = len(code),
			   pCode = (^u32)(raw_data(code)),
		   },
		   nil,
		   &sm,
	   ); result != .SUCCESS {
		panic("Failed to create shader module")
	}
	return
}

create_buffer :: proc(physical_device: vk.PhysicalDevice, device: vk.Device, size: vk.DeviceSize, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags) -> (buffer: vk.Buffer, memory: vk.DeviceMemory) {
	buffer_info := vk.BufferCreateInfo{
		sType = .BUFFER_CREATE_INFO,
		size = size,
		usage = usage,
		sharingMode = .EXCLUSIVE,
	}

	if vk.CreateBuffer(device, &buffer_info, nil, &buffer) != .SUCCESS {
		fmt.panicf("Failed to create buffer: {%v, %v, %v}\n", size, usage, properties)
	}

	mem_requirements: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(device, buffer, &mem_requirements)

	alloc_info := vk.MemoryAllocateInfo{
		sType = .MEMORY_ALLOCATE_INFO,
		allocationSize = mem_requirements.size,
		memoryTypeIndex = find_memory_type(physical_device, mem_requirements.memoryTypeBits, properties),
	}

	if vk.AllocateMemory(device, &alloc_info, nil, &memory) != .SUCCESS {
		fmt.panicf("failed to allocate memory for the buffer: {%v, %v, %v}\n", size, usage, properties)
	}

	vk.BindBufferMemory(device, buffer, memory, 0)

	return
}

destroy_buffer :: proc(device: vk.Device, buffer: vk.Buffer, memory: vk.DeviceMemory) {
	defer vk.DestroyBuffer(device, buffer, nil)
	defer  vk.FreeMemory(device, memory, nil)
}

copy_buffer :: proc(device: vk.Device, src, dst: vk.Buffer, copy_infos: []vk.BufferCopy) {
	temp_command_buffer := scoped_single_time_commands(device, global_command_pools[int(_thread_global_handle)], global_renderer.main_queue)

	vk.CmdCopyBuffer(temp_command_buffer, src, dst, u32(len(copy_infos)), raw_data(copy_infos))
}

@(deferred_in_out = end_single_time_commands)
scoped_single_time_commands :: proc(device: vk.Device, command_pool: vk.CommandPool, submit_queue: vk.Queue) -> vk.CommandBuffer {
	return begin_single_time_commands(device, command_pool, submit_queue)
}


// Creates a temporary command buffer for one time submit / oneshot commands
// to be written to GPU
begin_single_time_commands :: proc(device: vk.Device, command_pool: vk.CommandPool, submit_queue: vk.Queue) -> (buffer: vk.CommandBuffer) {
	vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
		sType = .COMMAND_BUFFER_ALLOCATE_INFO,
		level = .PRIMARY,
		commandPool = command_pool,
		commandBufferCount = 1,
	}, &buffer)

	vk.BeginCommandBuffer(buffer, &vk.CommandBufferBeginInfo{
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {.ONE_TIME_SUBMIT},
	})
	
	return
}

// Ends the temporary command buffer and submits the commands
end_single_time_commands :: proc(device: vk.Device, command_pool: vk.CommandPool, submit_queue: vk.Queue, buffer: vk.CommandBuffer) {
	buffer := buffer

	vk.EndCommandBuffer(buffer)

	vk.QueueSubmit(submit_queue, 1, &vk.SubmitInfo{
		sType = .SUBMIT_INFO,
		commandBufferCount = 1,
		pCommandBuffers = &buffer,
	}, {})
	vk.QueueWaitIdle(submit_queue)

	vk.FreeCommandBuffers(device, command_pool, 1, &buffer)
}

find_memory_type :: proc(physical_device: vk.PhysicalDevice, type_filter: u32, properties: vk.MemoryPropertyFlags) -> u32 {
	mem_properties: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(physical_device, &mem_properties)

	for i in 0..<mem_properties.memoryTypeCount {
		if type_filter & (1 << i) != 0 && (mem_properties.memoryTypes[i].propertyFlags & properties == properties) {
			return i
		}
	}

	panic("Failed to find suitable memory type!")

}

transition_image_layout :: proc(device: vk.Device, queue: vk.Queue, image: vk.Image, format: vk.Format, old_layout, new_layout: vk.ImageLayout, mip_levels: u32) {
	command_buffer := scoped_single_time_commands(device, global_command_pools[int(_thread_global_handle)], queue)
	barrier := vk.ImageMemoryBarrier{
		sType = .IMAGE_MEMORY_BARRIER,
		oldLayout = old_layout,
		newLayout = new_layout,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = image,
		subresourceRange = {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = mip_levels,
			baseArrayLayer = 0,
			layerCount = 1,
		},
		srcAccessMask = nil,
		dstAccessMask = nil,
	}

	source_stage, destination_stage: vk.PipelineStageFlags

	if old_layout == .UNDEFINED && new_layout == .TRANSFER_DST_OPTIMAL {
		barrier.srcAccessMask = nil
		barrier.dstAccessMask = nil
		
		source_stage = {.TOP_OF_PIPE}
		destination_stage = {.TRANSFER}
	} else if old_layout == .TRANSFER_DST_OPTIMAL && new_layout == .SHADER_READ_ONLY_OPTIMAL {
		barrier.srcAccessMask = {.TRANSFER_WRITE}
		barrier.dstAccessMask = {.SHADER_READ}

		source_stage = {.TRANSFER}
		destination_stage = {.FRAGMENT_SHADER}
	} else {
		panic("unsupported layout transition!")
	}
	vk.CmdPipelineBarrier(command_buffer, source_stage, destination_stage, {}, 0, nil, 0, nil, 1, &barrier)
}

debug_callback :: proc "system" (
	message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
	message_type: vk.DebugUtilsMessageTypeFlagsEXT,
	p_callback_data: ^vk.DebugUtilsMessengerCallbackDataEXT,
	p_user_data: rawptr,
) -> b32 {
	context = runtime.default_context()
	// if message_severity & {.WARNING, .ERROR} != nil {
		fmt.println()
        fmt.printf("MESSAGE: (")
        for ms in vk.DebugUtilsMessageSeverityFlagEXT {
            if ms in message_severity {
                fmt.printf("%v, ", ms)
            }
        }
        for t in vk.DebugUtilsMessageTypeFlagEXT {
            if t in message_type {
                fmt.printf("%v", t)
            }
        }
        fmt.printf(")\n")
        fmt.println("---------------")
		fmt.printf("%#v\n", p_callback_data.pMessage)
        fmt.println()
	// }

    if .ERROR in message_severity {
        panic("stop on first error")
    }
	return false
}
