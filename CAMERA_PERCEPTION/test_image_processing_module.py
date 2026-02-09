from pathlib import Path
from src.image_preprocess import ImageReader
from src.utils import CameraPosition, CameraLensType
from src.output_interfaces import Visualizer

camera_scene_01_path = Path(r"C:\Users\Robotic Muhirwa\Downloads\Tracking_Opencv\CAMERA_PERCEPTION\data\scene_01\camera_front")


all_camera_images_paths = sorted(camera_scene_01_path.glob("*.png"))

print(all_camera_images_paths)
print(len(all_camera_images_paths))

# shows the current file path
print(__file__)

# shows the current directory path
print(Path(__file__).parent)

# shows the current directory path appended with data/scene_01/camera_front
print(Path(__file__).parent.joinpath("data").joinpath("scene_01").joinpath("camera_front"))
# it is equivalent to
camera_scene_01_path = Path(__file__).parent / "data" / "scene_01" / "camera_front"

# init
camera_front_image_reader = ImageReader(camera_position=CameraPosition.CAMERA_FRONT, camera_lens_type=CameraLensType.FISH_EYE_LENS)
vis = Visualizer()

for index, image_path in enumerate(all_camera_images_paths):
    img = camera_front_image_reader.read_image_from_disk(image_file_path=image_path)
    vis.show_image(image=img, print_frame_id_on_image=True, print_time_on_image=True, wait_time_msec=50)

