
# Meant to be deployed to Modal- which creates a REST endpoint.
# You'll need a huggingface token set up and to get access to the stabilityai/stable-audio-open-1.0 repo
# If you're deploying it to Modal, add an authorization header secret to keep it locked down!! (Already implemented in this script)

from diffusers import StableAudioPipeline
from diffusers import DPMSolverMultistepScheduler
from modal import App, Image, Secret, fastapi_endpoint
import torch
import os
from fastapi import Request
from huggingface_hub import login
from fastapi.responses import StreamingResponse
import io
import soundfile as sf


app = App("text-synth")
image = (
    Image.debian_slim()
    .pip_install(
        "torch",
        "fastapi[standard]",
        "diffusers",
        "transformers",
        "huggingface_hub",
        "torchsde",
        "soundfile")
)

@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[Secret.from_name("synth-token"),
             Secret.from_name("inference-secret")]
)
@fastapi_endpoint(method="POST", docs=True)
async def synthesize_audio_from_text(request: Request):
    # Login to huggingface to get the model
    login(os.environ["TOKEN"])
    # print("Beginning inference")
    # Check auth secret
    auth_header = request.headers.get("authorization")
    if auth_header != f"Bearer {os.environ['INFERENCE_SECRET']}":
        return {"error": "Unauthorized"}, 401
    
    body = await request.json()
    prompt = body.get("prompt")
    # print("prompt")
    # print(prompt)
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16, use_auth_token=os.environ["TOKEN"]
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    # probably worth tweaking
    negative_prompt = "Low quality."

    # set the seed for generator
    generator = torch.Generator("cuda").manual_seed(0)

    # run the generation
    audio = pipe(
        prompt,
        negative_prompt=negative_prompt,
        # Can tweak this
        num_inference_steps=30,
        # Audio length
        audio_end_in_s=10.0,
        num_waveforms_per_prompt=1,
        generator=generator,
    ).audios

    output = audio[0].T.to(dtype=torch.float32)
    # Normalize audio
    output = (output / output.abs().max()) * 0.8
    # print("output")
    # print(output)
    output = output.cpu().numpy()

    # Save audio to in-memory WAV
    buf = io.BytesIO()
    sf.write(buf, output, 44100, format="WAV")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")
