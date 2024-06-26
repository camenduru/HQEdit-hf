from __future__ import annotations

import math
from glob import glob
from functools import partial
import random

import gradio as gr
import torch
from PIL import Image, ImageOps
from datasets import load_dataset
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


def generate(
    input_image: Image.Image,
    instruction: str,
    steps: int,
    randomize_seed: bool,
    seed: int,
    randomize_cfg: bool,
    text_cfg_scale: float,
    image_cfg_scale: float,
    pipe: StableDiffusionInstructPix2PixPipeline
):
    seed = random.randint(0, 100000) if randomize_seed else seed
    text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
    image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale

    width, height = input_image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    if instruction == "":
        return [seed, text_cfg_scale, image_cfg_scale, input_image]

    generator = torch.manual_seed(seed)
    edited_image = pipe(
        instruction, image=input_image,
        guidance_scale=text_cfg_scale, image_guidance_scale=image_cfg_scale,
        num_inference_steps=steps, generator=generator,
    ).images[0]
    return [seed, text_cfg_scale, image_cfg_scale, edited_image]


def show_image(image_name, image_options):
    if image_name is None:
        return

    return image_options[image_name]


def reset():
    return [0, "Randomize Seed", 1371, "Fix CFG", 7.5, 1.5, None, None, None, ""]


def sample(dataset):
    sample_id = random.choice(list(range(len(dataset["train"]))))
    sample = dataset["train"][sample_id]
    return [sample["input_image"], sample["output_image"], sample["edit"], sample["inverse_edit"]]


HELP_TEXT = """
If you're not getting what you want, there may be a few reasons:
1. Is the image not changing enough? Your Image CFG weight may be too high. This value dictates how similar the output should be to the input. It's possible your edit requires larger changes from the original image, and your Image CFG weight isn't allowing that. Alternatively, your Text CFG weight may be too low. This value dictates how much to listen to the text instruction. The default Image CFG of 1.5 and Text CFG of 7.5 are a good starting point, but aren't necessarily optimal for each edit. Try:
    * Decreasing the Image CFG weight, or
    * Increasing the Text CFG weight, or
2. Conversely, is the image changing too much, such that the details in the original image aren't preserved? Try:
    * Increasing the Image CFG weight, or
    * Decreasing the Text CFG weight
3. Try generating results with different random seeds by setting "Randomize Seed" and running generation multiple times. You can also try setting "Randomize CFG" to sample new Text CFG and Image CFG values each time.
4. Rephrasing the instruction sometimes improves results (e.g., "turn him into a dog" vs. "make him a dog" vs. "as a dog").
5. Increasing the number of steps sometimes improves results.
6. Do faces look weird? The Stable Diffusion autoencoder has a hard time with faces that are small in the image. Try:
    * Cropping the image so the face takes up a larger portion of the frame.
"""


def main():
    model_id = "MudeHui/ip2p-warp-gpt4v"
    if torch.cuda.is_available():
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
        pipe = pipe.to('cuda')
    else:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float, safety_checker=None)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    image_options = {path.split("/")[-1].split(".")[0]: path for path in sorted(glob("imgs/*png"))}

    with gr.Blocks() as demo:
        gr.HTML("""<h1 style="font-weight: 900; margin-bottom: 7px;">
        HQ-Edit: A High-Quality and High-Coverage Dataset for General Image Editing
</h1>
<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.
<br/>
<a href="https://huggingface.co/spaces/tennant/HQEdit?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
<p/>""")
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                dropdown = gr.Dropdown(list(image_options.keys()), label="Select from Given Images")

            with gr.Column(scale=3):
                instruction = gr.Textbox(lines=1, label="Edit Instruction", interactive=True)

            with gr.Column(scale=1, min_width=100):
                generate_button = gr.Button("Generate")
                reset_button = gr.Button("Reset")

        with gr.Row():
            input_image = gr.Image(label="Input Image", type="pil", interactive=True, height=512, width=512)
            edited_image = gr.Image(label=f"Edited Image", type="pil", interactive=False, height=512, width=512)

        with gr.Row():
            steps = gr.Number(value=20, precision=0, label="Steps", interactive=True)
            randomize_seed = gr.Radio(
                ["Fix Seed", "Randomize Seed"],
                value="Randomize Seed",
                type="index",
                show_label=False,
                interactive=True,
            )
            seed = gr.Number(value=1371, precision=0, label="Seed", interactive=True)
            randomize_cfg = gr.Radio(
                ["Fix CFG", "Randomize CFG"],
                value="Fix CFG",
                type="index",
                show_label=False,
                interactive=True,
            )
            text_cfg_scale = gr.Number(value=7.0, label=f"Text CFG", interactive=True)
            image_cfg_scale = gr.Number(value=1.5, label=f"Image CFG", interactive=True)

        gr.Markdown(HELP_TEXT)

        with gr.Row():
            gr.Markdown("## Dataset Preview")
            sample_button = gr.Button("See Another Sample")

        with gr.Row():
            input_image_preview = gr.Image(label="Input Image", type="pil", height=512, width=512)
            output_image_preview = gr.Image(label="Output Image", type="pil", height=512, width=512)

        edit_text = gr.Textbox(label="Edit Instruction")
        inv_edit_text = gr.Textbox(label="Inverse Edit Instruction")

        generate_func = partial(generate, pipe=pipe)

        generate_button.click(
            fn=generate_func,
            inputs=[
                input_image,
                instruction,
                steps,
                randomize_seed,
                seed,
                randomize_cfg,
                text_cfg_scale,
                image_cfg_scale,
            ],
            outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image],
        )
        reset_button.click(
            fn=reset,
            inputs=[],
            outputs=[steps, randomize_seed, seed, randomize_cfg, text_cfg_scale, image_cfg_scale, input_image, edited_image, dropdown, instruction],
        )

        show_image_func = partial(show_image, image_options=image_options)
        dropdown.change(show_image_func, inputs=dropdown, outputs=input_image)

        dataset = load_dataset("UCSC-VLAA/HQ-Edit-data-demo")
        sample_func = partial(sample, dataset=dataset)
        sample_button.click(
            fn=sample_func,
            inputs=[],
            outputs=[input_image_preview, output_image_preview, edit_text, inv_edit_text]
        )

    demo.queue()
    demo.launch(share=True, max_threads=1)


if __name__ == "__main__":
    main()
