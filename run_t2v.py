import torch
import imageio
from t2v_main import T2VPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = T2VPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# Frame-level prompt for each frame
prompts = [
    "A corgi is running on a grassy field, its ears flopping as it moves.",
    "The corgi continues running, a second corgi starts to appear in the background.",
    "The second corgi starts to run, playfully chasing the first corgi.",
    "The first corgi maintains its pace, the second corgi getting closer.",
    "Both corgis are running side by side, their short legs moving quickly.",
    "The second corgi starts to take the lead, the first corgi following closely.",
    "Both corgis continue running, their tails wagging happily as they race.",
    "The first corgi begins to catch up, the two corgis running neck and neck.",
]

# DirecT2V sample
result = pipe(prompt=prompts, generator=torch.Generator('cuda').manual_seed(10)).images
result = [(r * 255).astype("uint8") for r in result]
imageio.mimsave("video_out.mp4", result, fps=6)
