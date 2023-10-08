import torch
import imageio
from t2v_main import T2VPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = T2VPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# Frame-level prompt for each frame
prompts = [
    "A colorful hummingbird perched on a flower's petal.",
    "The hummingbird's iridescent feathers glistening in the sunlight.",
    "The bird's long, slender beak probing deep into the flower's bloom.",
    "Hummingbird delicately sipping nectar from the flower.",
    "Its wings starting to flutter rapidly.",
    "The hummingbird taking off into the air with a blur of wings.",
    "It hovers in mid-air, suspended by sheer wing power.",
    "Darting to the next flower in a blink of an eye.",
    "The tiny bird's body remains almost motionless while hovering.",
    "Its wings creating a visible, rapid vibration.",
    "Zooming in, you can see the intricate patterns on its feathers.",
    "The flower's vibrant colors contrasting with the bird's elegance.",
    "Hummingbird's quick and precise movements between blooms.",
    "It appears to dance in the air, fueled by energy.",
    "A close-up of its long, slender beak entering another flower.",
    "The scene concludes as the hummingbird darts away into the distance."
]


# T2V sample
result = pipe(prompt=prompts, generator=torch.Generator('cuda').manual_seed(10)).images
result = [(r * 255).astype("uint8") for r in result]
imageio.mimsave("video_out.mp4", result, fps=5)
