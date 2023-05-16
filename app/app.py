""" Module for web UI """
import os
import gradio
from inference import *
from helpers import *
from checkboxes import *


block = gradio.Blocks(css=".container { max-width: 800px; margin: auto; }")


def infer(prompt):
    response = generate_image(prompt)
    return response["images"]


# UI
with block:
    gradio.Markdown("<h1><center>Face Generator</center></h1>")

    # Text input only
    with gradio.Row().style(equal_height=True):
        text = gradio.Textbox(
            label = "Text prompt",
            max_lines = 1,
            max_length = 255
        )

    # With options
    # gender = gradio.Radio(
    #     ["Female", "Male"],
    #     label="Gender"
    # )

    # with gradio.Row().style(equal_height=True):
    #     with gradio.Column(scale=1):
    #         five_o_clock_shadow = Other("5 o'clock shadow")
    #         arched_eyebrows = Other("Arched eyebrows")
    #         attractive = AdVerb("Attractive")
    #         bags_under_eyes = Other("Bags under eyes")
    #         bald = AdVerb("Bald")
    #         bangs = Other("Bangs")
    #         big_lips = Other("Big lips")
    #         big_nose = Other("Big nose")
    #         black_hair = Other("Black hair")
    #         blond_hair = Other("Blond hair")
    #         blurry = AdVerb("Blurry")
    #         brown_hair = Other("Brown hair")
    #         bushy_eyebrows = Other("Bushy eyebrows")
    #     with gradio.Column(scale=1):
    #         cubby = AdVerb("Cubby")
    #         double_chin = Other("Double chin")
    #         eyeglasses = Other("Eyeglasses")
    #         goatee = Other("Goatee")
    #         gray_hair = Other("Gray hair")
    #         heavy_makeup = Other("Heavy makeup")
    #         high_cheekbones = Other("High cheekbones")
    #         mouth_slightly_open = Other("Mouth slightly open")
    #         mustache = Other("Mustache")
    #         narrow_eyes = Other("Narrow eyes")
    #         no_beard = Other("No beard")
    #         oval_face = Other("Oval face")
    #         pale_skin = Other("Pale skin")
    #     with gradio.Column(scale=1):
    #         pointy_nose = Other("Pointy nose")
    #         receding_hairline = Other("Receding hairline")
    #         rosy_cheeks = Other("Rosy cheeks")
    #         sideburns = Other("Sideburns")
    #         smiling = AdVerb("Smiling")
    #         straight_hair = Other("Straight hair")
    #         wavy_hair = Other("Wavy hair")
    #         wearing_earrings = AdVerb("Wearing earrings")
    #         wearing_hat = AdVerb("Wearing hat")
    #         wearing_lipstick = AdVerb("Wearing lipstick")
    #         wearing_necklace = AdVerb("Wearing necklace")
    #         wearing_necktie = AdVerb("Wearing necktie")
    #         young = AdVerb("Young")

    btn = gradio.Button("Run")

    gallery = gradio.Gallery(
        label = "Result image",
    ).style(
        columns = [3],
        height = "auto"
    )

    # PERSON = "woman"
    # PRONOUN = "She"
    # if gender.value == "male":
    #     PERSON = "man"
    #     PRONOUN = "He"

    # text = f"""This {PERSON} {five_o_clock_shadow.get_text()}, {arched_eyebrows.get_text()}, {attractive.get_text()}, {bags_under_eyes.get_text()}, {bald.get_text()}, {bangs.get_text()}, {big_lips.get_text()}, {big_nose.get_text()}, {black_hair.get_text()}, {blond_hair.get_text()}, {blurry.get_text()}, {brown_hair.get_text()}, {bushy_eyebrows.get_text()}, {cubby.get_text()}, {double_chin.get_text()}, {eyeglasses.get_text()}, {goatee.get_text()}, {gray_hair.get_text()}, {heavy_makeup.get_text()}, {high_cheekbones.get_text()}, {mouth_slightly_open.get_text()}, {mustache.get_text()}, {narrow_eyes.get_text()}, {no_beard.get_text()}, {oval_face.get_text()}, {pale_skin.get_text()}, {pointy_nose.get_text()}, {receding_hairline.get_text()}, {rosy_cheeks.get_text()}, {sideburns.get_text()}, {smiling.get_text()}, {straight_hair.get_text()}, {wavy_hair.get_text()}, {wearing_earrings.get_text()}, {wearing_hat.get_text()}, {wearing_lipstick.get_text()}, {wearing_necklace.get_text()}, {wearing_necktie.get_text()}, {young.get_text()}."""

    btn.click(fn=generate_image, inputs=text, outputs=gallery)

    gradio.Markdown("<p style='text-align: center'>© Fedora Yoshe Juandy</p>")


block.queue().launch(share=True)

