import carla
import numpy as np
import torch
import onnxruntime
import time
import cv2
import queue

# Initialize CARLA client
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
client.load_world('Town01')
world = client.get_world()
world.set_weather(carla.WeatherParameters.ClearSunset)
settings = world.get_settings()
settings.synchronous_mode = True  # Enable synchronous mode
settings.fixed_delta_seconds = 0.05  # Set fixed time step
world.apply_settings(settings)



# ONNX Model Setup
model_onnx = "carla2cs.onnx"
options = onnxruntime.SessionOptions()
options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL

options.add_session_config_entry("tensorrt_engine_cache_path", "./engine_cache")
options.add_session_config_entry("tensorrt_engine_decryption_enable", "1")
options.add_session_config_entry("tensorrt_engine_cache_enable", "1")

print("Started building the TensorRT engine...")
session = onnxruntime.InferenceSession(
    model_onnx,
    options,
    providers=[
        ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
        "CUDAExecutionProvider"
    ]
)
io_binding = session.io_binding()
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Image normalization functions
def denormalize_image(img, mean=(0.5,), std=(0.5,), max_pixel_value=255.0):
    img = img * np.array(std).reshape(1, -1, 1, 1)
    img = img + np.array(mean).reshape(1, -1, 1, 1)
    img = img * max_pixel_value
    return np.clip(img, 0, 255).astype(np.uint8)

def normalize_image(img, mean=(0.5,), std=(0.5,), max_pixel_value=255.0):
    img = img / max_pixel_value
    img = (img - np.array(mean).reshape(1, -1, 1, 1)) / np.array(std).reshape(1, -1, 1, 1)
    return img.astype(np.float32)

# Spawn vehicle at a random location
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
spawn_points = world.get_map().get_spawn_points()
spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Attach camera sensor
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", "960")
camera_bp.set_attribute("image_size_y", "540")
camera_bp.set_attribute("fov", "90")
camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=2.0, z=1.0)), attach_to=vehicle)

# Queue for synchronized image processing
image_queue = queue.Queue()

def process_image(image):
    """Store the image in a queue to be processed after world.tick()"""
    image_queue.put(image)

camera.listen(lambda image: process_image(image))

try:
    while True:
        vehicle.set_autopilot(True)
        world.tick()  # Step the simulation

        if not image_queue.empty():
            image = image_queue.get()

            # Convert CARLA image to NumPy array
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3]  # Remove alpha channel
            array = np.transpose(array, (2, 0, 1))

            image_tensor = normalize_image(array)
            image_tensor = np.expand_dims(array, axis=0).astype(np.float32)
            image_tensor = onnxruntime.OrtValue.ortvalue_from_numpy(image_tensor, 'cuda', 0)

            tdtype = np.float32
            io_binding.bind_input(name=input_name, device_type=image_tensor.device_name(), device_id=0,
                                  element_type=tdtype, shape=image_tensor.shape(), buffer_ptr=image_tensor.data_ptr())
            io_binding.bind_output(output_name, device_type='cuda', device_id=0, element_type=tdtype)

            start = time.time()
            session.run_with_iobinding(io_binding)
            print("Inference Time:", time.time() - start)

            output = io_binding.copy_outputs_to_cpu()[0]
            output = denormalize_image(output)

            output = np.squeeze(output, axis=0)
            output_bgr = np.transpose(output, (1, 2, 0))
            cv2.imshow("ONNX Output", output_bgr)
            cv2.waitKey(1)

except KeyboardInterrupt:
    print("Stopping...")
    camera.stop()
    vehicle.destroy()
    cv2.destroyAllWindows()
