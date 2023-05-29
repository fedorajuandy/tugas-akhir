""" Module for web application """

import gradio
from inference import generate_image


with gradio.Blocks() as block:
    gradio.Markdown("<h1><center>Image Generator</center></h1>")

    text = gradio.Textbox(
        label = "Text prompt",
        max_lines = 1,
    )

    btn = gradio.Button("Run")

    gallery = gradio.Gallery(
        label = "Result image",
    ).style(
        # number of images per category
        columns = [3],
        height = "auto",
    )

    btn.click(fn=generate_image, inputs=text, outputs=gallery)

    gradio.Markdown("<p style='text-align: center'>2019130032 - Fedora Yoshe Juandy</p>")


# use 'queue' if the inference's time > 60s
# use 'share' for more than one user usage
block.queue().launch(share=True)
