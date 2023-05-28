""" Module for web UI """
import gradio
from inference import generate_image # pylint: disable=import-error


block = gradio.Blocks(css=".container { max-width: 800px; margin: auto; }")


# UI
with block:
    gradio.Markdown("<h1><center>Image Generator</center></h1>")

    with gradio.Row().style(equal_height=True):
        text = gradio.Textbox(
            label = "Text prompt",
            max_lines = 1,
        )

    btn = gradio.Button("Run")

    gallery = gradio.Gallery(
        label = "Result image",
    ).style(
        columns = [3],
        height = "auto"
    )

    btn.click(fn=generate_image, inputs=text, outputs=gallery)

    gradio.Markdown("<p style='text-align: center'>2019130032 - Fedora Yoshe Juandy</p>")


block.queue().launch(share=True)
