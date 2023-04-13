package gol

import "core:sync"
import "core:strings"
import "core:slice"

import vk "vendor:vulkan"
import "vendor:glfw"

WriterHandle :: distinct u64
@thread_local _thread_global_handle: WriterHandle
_global_atomic_counter: u64

global_renderer: Renderer

vertex_shader :: #load("../shaders/vert.spv")
fragment_shader :: #load("../shaders/frag.spv")
compute_shader :: #load("../shaders/comp.spv")

MAX_FRAMES_IN_FLIGHT :: 2
DEFAULT_THREAD_CAPACITY :: 4
global_command_pools: [dynamic]vk.CommandPool
global_command_buffers: [dynamic][MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer

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
    global_command_buffers = make([dynamic][MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer, DEFAULT_THREAD_CAPACITY)

    glfw.Init()
    vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))
    glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)

    global_renderer.window = glfw.CreateWindow(WIDTH, HEIGHT, "Hello Dynamic Rendering Vulkan", nil, nil)
	assert(global_renderer.window != nil, "Window could not be crated")

    glfw.SetWindowUserPointer(global_renderer.window, &global_renderer)
	glfw.SetFramebufferSizeCallback(global_renderer.window, framebuffer_resize_callback)


    instance_extension := get_required_instance_extensions()
    defer delete(instance_extension)

    if vk.CreateInstance(&vk.InstanceCreateInfo{
        sType = .INSTANCE_CREATE_INFO,
        enabledExtensionCount = u32(len(instance_extension)),
        ppEnabledExtensionNames = raw_data(instance_extension),
        pApplicationInfo = &vk.ApplicationInfo{},
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
        sampler := [?]vk.DescriptorSetLayoutBinding{
            {
                binding = 0,
                descriptorCount = 1,
                descriptorType = .COMBINED_IMAGE_SAMPLER,
                stageFlags = {.FRAGMENT},
            },
            {
                binding = 1,
                descriptorCount = 1,
                descriptorType = .STORAGE_BUFFER,
                stageFlags = {.COMPUTE},
            },
            {
                binding = 2,
                descriptorCount = 1,
                descriptorType = .STORAGE_BUFFER,
                stageFlags = {.COMPUTE},
            },
        }

        if vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount = u32(len(sampler[:])),
            pBindings = raw_data(sampler[:]),
        }, nil, &descriptors.layout) != .SUCCESS {
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
            {
                sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage = {.COMPUTE},
                module = compute_shader_module,
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

        rasterizer := vk.PipelineRasterizationStateCreateInfo{
            sType = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            polygonMode = .FILL,
            cullMode = {.BACK},
            frontFace = .COUNTER_CLOCKWISE,
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
            pRasterizationState = &rasterizer,
            pMultisampleState = &multisampling,
            pColorBlendState = &color_blending,
            pDynamicState = &dynamic_state,
            layout = pipeline.layout,
            basePipelineIndex = -1,
        }, nil, &pipeline.pipeline) != .SUCCESS {
            panic("failed to create pipeline")
        }
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

            if vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
                sType = .COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool = pool,
                level = .PRIMARY,
                commandBufferCount = MAX_FRAMES_IN_FLIGHT,
            }, raw_data(global_command_buffers[i][:])) != .SUCCESS {
                panic("Failed to create command buffer!")
            }
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
                type = .STORAGE_BUFFER,
                descriptorCount = 2 * u32(MAX_FRAMES_IN_FLIGHT),
            },
        }

        if vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
            sType = .DESCRIPTOR_POOL_CREATE_INFO,
            poolSizeCount = u32(len(pool_sizes)),
            pPoolSizes = raw_data(pool_sizes),
            maxSets = u32(MAX_FRAMES_IN_FLIGHT),
        }, nil, &descriptors.pool) != .SUCCESS {
            panic("Failed to create descriptor pool!")
        }

        layouts := [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSetLayout{
            descriptors.layout,
            descriptors.layout,
        }

        if vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
            sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool = descriptors.pool,
            descriptorSetCount = u32(MAX_FRAMES_IN_FLIGHT),
            pSetLayouts = raw_data(layouts[:]),
        }, raw_data(descriptors.sets[:])) != .SUCCESS {
            panic("Failed to allocate descriptor sets!")
        }
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

renderer_create_texture :: proc(renderer: Renderer, image: Image) -> vk.Image {
    return {}
}

renderer_destroy_texture :: proc(renderer: Renderer, texture: vk.Image) {

}

WIDTH :: 800
HEIGHT :: 600

ENABLE_VALIDATION_LAYERS :: ODIN_DEBUG || #config(enable_validation_layers, false)

Render_Context :: struct {
    window: glfw.WindowHandle,
    surface: vk.SurfaceKHR,
    instance: vk.Instance,
    device: vk.Device,
    main_queue: vk.Queue,
    swapchain: Swapchain,
    descriptors: Descriptors,
    pipeline: Pipeline,
    syncs: Syncs,
}

Renderer :: Render_Context

Image :: struct {
    width, height: int,
    data: [][4]f32,
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
    pool: vk.DescriptorPool,
    sets: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSet,
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