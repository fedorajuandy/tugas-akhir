""" Module for web UI """
import os
import gradio as gr
from backend import get_images_from_backend


block = gr.Blocks(css=".container { max-width: 800px; margin: auto; }")
backend_url = os.environ["BACKEND_SERVER"] + "/generate"


def infer(prompt):
    """ Send request, generate image, get image """
    response = get_images_from_backend(prompt, backend_url)
    return response["images"]


# UI
with block:
    gr.Markdown("<h1><center>Face Generator</center></h1>")

    # Text input only
    # with gr.Row().style(mobile_collapse=False, equal_height=True):
    #     text = gr.Textbox(
    #         label = "Text prompt",
    #         max_lines = 1,
    #         max_length = 255
    #     )

    # With options
    gender = gr.Radio(
        ["Female", "Male"],
        label="Gender"
    )

    with gr.Row().style(mobile_collapse=False, equal_height=True):
        with gr.Column(scale=1):
            five_o_clock_shadow = gr.Checkbox(
                label = "5 o'clock shadow",
            )
            arched_eyebrows = gr.Checkbox(
                label = "Arched eyebrows",
            )
            attractive = gr.Checkbox(
                label = "Attractive",
            )
            bags_under_eyes = gr.Checkbox(
                label = "Bags under eyes",
            )
            bald = gr.Checkbox(
                label = "Bald",
            )
            bangs = gr.Checkbox(
                label = "Bangs",
            )
            big_lips = gr.Checkbox(
                label = "Big lips",
            )
            big_nose = gr.Checkbox(
                label = "Big nose",
            )
            black_hair = gr.Checkbox(
                label = "Black hair",
            )
            blond_hair = gr.Checkbox(
                label = "Blond hair",
            )
            blurry = gr.Checkbox(
                label = "Blurry",
            )
            brown_hair = gr.Checkbox(
                label = "Brown hair",
            )
            bushy_eyebrows = gr.Checkbox(
                label = "Bushy eyebrows",
            )
        with gr.Column(scale=1):
            cubby = gr.Checkbox(
                label = "Cubby",
            )
            double_chin = gr.Checkbox(
                label = "Double chin",
            )
            eyeblasses = gr.Checkbox(
                label = "Eyeblasses",
            )
            goatee = gr.Checkbox(
                label = "Goatee",
            )
            gray_hair = gr.Checkbox(
                label = "Gray hair",
            )
            heavy_makeup = gr.Checkbox(
                label = "Heavy makeup",
            )
            high_cheekbones = gr.Checkbox(
                label = "High cheekbones",
            )
            mouth_slightly_open = gr.Checkbox(
                label = "Mouth slightly open",
            )
            mustache = gr.Checkbox(
                label = "Mustache",
            )
            narrow_eyes = gr.Checkbox(
                label = "Narrow eyes",
            )
            no_beard = gr.Checkbox(
                label = "Beard",
            )
            oval_face = gr.Checkbox(
                label = "Oval face",
            )
            pale_skin = gr.Checkbox(
                label = "Pale skin",
            )
        with gr.Column(scale=1):
            pointy_nose = gr.Checkbox(
                label = "Pointy nose",
            )
            receding_hairline = gr.Checkbox(
                label = "Receding hairline",
            )
            rosy_cheeks = gr.Checkbox(
                label = "Rosy cheeks",
            )
            sideburns = gr.Checkbox(
                label = "Sideburns",
            )
            smiling = gr.Checkbox(
                label = "Smiling",
            )
            straight_hair = gr.Checkbox(
                label = "Straight hair",
            )
            wavy_hair = gr.Checkbox(
                label = "Wavy hair",
            )
            wearing_earrings = gr.Checkbox(
                label = "Wearing earrings",
            )
            wearing_hat = gr.Checkbox(
                label = "Wearing hat",
            )
            wearing_lipstick = gr.Checkbox(
                label = "Wearing lipstick",
            )
            wearing_necklace = gr.Checkbox(
                label = "Wearing necklace",
            )
            wearing_necktie = gr.Checkbox(
                label = "Wearing necktie",
            )
            young = gr.Checkbox(
                label = "Young",
            )

    btn = gr.Button("Run")

    gallery = gr.Gallery(
        label = "Result image",
    ).style(
        columns = [3],
        height = "auto"
    )

    import itertools
    iter = itertools.permutations(["Alice", "Bob", "Carol"])
    list(iter)

    text = f"{gender} has {five_o_clock_shadow}"

    text.submit(infer, inputs=text, outputs=gallery)
    btn.click(infer, inputs=text, outputs=gallery)

    gr.Markdown("<p style='text-align: center'>© Fedora Yoshe Juandy</p>")


block.launch(enable_queue=False)
