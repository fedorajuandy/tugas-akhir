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
    # with gr.Group():
        # with gr.Box():
            # with gr.Row().style(mobile_collapse=False, equal_height=True):

    # text = gr.Textbox(
    #     label = "Text prompt",
    #     max_lines = 1
    # )

    five_o_clock_shadow = gr.Checkbox(
        label = "5 o'clock shadow",
    )
    arched_eyebrows = gr.Checkbox(
        label = "Arched eyebrows",
    )
    bushy_eyebrows = gr.Checkbox(
        label = "Bushy eyebrows",
    )
    attractive = gr.Checkbox(
        label = "Attractive",
    )
    bags_under_eyes = gr.Checkbox(
        label = "Bags under eyes",
    )
    backend_urlbangs = gr.Checkbox(
        label = "Backend urlbangs",
    )
    big_lips = gr.Checkbox(
        label = "Big lips",
    )
    big_nose = gr.Checkbox(
        label = "Big nose",
    )
    hair_color = gr.Radio(
        ["black", "blond", "brown", "gray"],
        label="Hair color"
    )
    blurry = gr.Checkbox(
        label = "Blurry",
    )
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
    heavy_makeup = gr.Checkbox(
        label = "Heavy makeup",
    )
    high_cheekbones = gr.Checkbox(
        label = "High cheekbones",
    )
    gender = gr.Radio(
        ["female", "male"],
        label="Gender"
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
