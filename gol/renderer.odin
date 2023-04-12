package gol

import "core:sync"

import vk "vendor:vulkan"
import "vendor:glfw"

WriterHandle :: distinct u64
@thread_local _thread_global_handle: WriterHandle
_global_atomic_counter: u64

register_writer :: proc() -> WriterHandle {
    if _thread_global_handle == {} {
        current_value := sync.atomic_load(&_global_atomic_counter)
        value, swapped := sync.atomic_compare_exchange_strong(&_global_atomic_counter, current_value, current_value + 1)
        for !swapped {
            current_value = sync.atomic_load(&_global_atomic_counter)
            value, swapped = sync.atomic_compare_exchange_strong(&_global_atomic_counter, current_value, current_value + 1)
        }
        _thread_global_handle = WriterHandle(value + 1) // cas returns previous value
    }
    return _thread_global_handle
}

init :: proc() {
    glfw.Init()
    defer glfw.Terminate()
    vk.load_proc_addresses_global(rawptr(glfw.GetInstanceProcAddress))
    glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)

    render_context: Render_Context

    render_context.window = glfw.CreateWindow(WIDTH, HEIGHT, "Hello Dynamic Rendering Vulkan", nil, nil)
	assert(render_context.window != nil, "Window could not be crated")

    glfw.SetWindowUserPointer(render_context.window, &render_context)
	glfw.SetFramebufferSizeCallback(render_context.window, framebuffer_resize_callback)


    instance_extension := get_required_instance_extensions()
    defer delete(instance_extension)

    instance: vk.Instance
    if vk.CreateInstance(&vk.InstanceCreateInfo{
        sType = .INSTANCE_CREATE_INFO,
        enabledExtensionCount = u32(len(instance_extension)),
        ppEnabledExtensionNames = raw_data(instance_extension),
        pApplicationInfo = &vk.ApplicationInfo{},
    }, nil, &instance) != .SUCCESS {
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
	vk.load_proc_addresses(instance)
}

WIDTH :: 800
HEIGHT :: 600

ENABLE_VALIDATION_LAYERS :: ODIN_DEBUG || #config(enable_validation_layers, false)

Render_Context :: struct {
    window: glfw.WindowHandle,
    device: vk.Device,
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

DYNAMIC_RENDERING_FEATURES := vk.PhysicalDeviceDynamicRenderingFeaturesKHR{
    sType = .PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
    dynamicRendering = true,
}