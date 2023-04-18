""" Module for web UI """
import os
import gradio as gr
from backend import get_images_from_backend


block = gr.Blocks(css=".container { max-width: 800px; margin: auto; }")
backend_url = os.environ["BACKEND_SERVER"] + "/generate"


def infer(prompt):
    """ yatta """
    response = get_images_from_backend(prompt, backend_url)
    return response["images"]


with block:
    gr.Markdown("<h1><center>Face Generator</center></h1>")

    # Text input only
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):

                text = gr.Textbox(
                    label="Text prompt", max_lines=1
                ).style(
                    border=(True, False, True, True),
                    margin=False,
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Enter").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
        gallery = gr.Gallery(label="Result", show_label=False).style(
            grid=[3], height="auto"
        )
        text.submit(infer, inputs=text, outputs=gallery)
        btn.click(infer, inputs=text, outputs=gallery)

    gr.Markdown("<p style='text-align: center'>© Fedora Yoshe Juandy</p>")


block.launch(enable_queue=False)
