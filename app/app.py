""" Module for web UI """
import os
import gradio
from dataclasses import dataclass, field
from backend import get_images_from_backend


block = gradio.Blocks(css=".container { max-width: 800px; margin: auto; }")
backend_url = os.environ["BACKEND_SERVER"] + "/generate"


def infer(prompt):
    """ Send request, generate image, get image """
    response = get_images_from_backend(prompt, backend_url)
    return response["images"]


@dataclass
class Checkboxes:
    """
    Return value from each checkboxes.
    """

    label: str = field(
        default = None,
        metadata = {"help": "The shown label in app."},
    )
    checkbox: gradio.Checkbox = field(
        default = None,
        metadata = {"help": "A Gradio checkbox."}
    )

    def __init__(self, label):
        self.label = label
        assert (
            self.label is not None
        ), "Label's name needs to be specified."

        self.checkbox = gradio.Checkbox(
            label = self.label
        )

@dataclass
class AdVerb(Checkboxes):
    """
    Checkboxes with adverb or verb attributes.
    """

    def get_text(self):
        """
        Function to return part of input text.
        """
        if self.checkbox.value:
            return f"is {self.label.lower()}"
        else:
            return ""

@dataclass
class Other(Checkboxes):
    """
    Checkboxes with other attributes besides verbs and adverbs.
    """
    def get_text(self):
        """
        Function to return part of final text.
        """
        if self.checkbox.value:
            return f"has {self.label.lower()}"
        else:
            return ""


# UI
with block:
    gradio.Markdown("<h1><center>Face Generator</center></h1>")

    # Text input only
    # with gradio.Row().style(mobile_collapse=False, equal_height=True):
    #     text = gradio.Textbox(
    #         label = "Text prompt",
    #         max_lines = 1,
    #         max_length = 255
    #     )

    # With options
    gender = gradio.Radio(
        ["Female", "Male"],
        label="Gender"
    )

    with gradio.Row().style(equal_height=True):
        with gradio.Column(scale=1):
            five_o_clock_shadow = Other("5 o'clock shadow")
            arched_eyebrows = Other("Arched eyebrows")
            attractive = AdVerb("Attractive")
            bags_under_eyes = Other("Bags under eyes")
            bald = AdVerb("Bald")
            bangs = Other("Bangs")
            big_lips = Other("Big lips")
            big_nose = Other("Big nose")
            black_hair = Other("Black hair")
            blond_hair = Other("Blond hair")
            blurry = AdVerb("Blurry")
            brown_hair = Other("Brown hair")
            bushy_eyebrows = Other("Bushy eyebrows")
        with gradio.Column(scale=1):
            cubby = AdVerb("Cubby")
            double_chin = Other("Double chin")
            eyeglasses = Other("Eyeglasses")
            goatee = Other("Goatee")
            gray_hair = Other("Gray hair")
            heavy_makeup = Other("Heavy makeup")
            high_cheekbones = Other("High cheekbones")
            mouth_slightly_open = Other("Mouth slightly open")
            mustache = Other("Mustache")
            narrow_eyes = Other("Narrow eyes")
            no_beard = Other("No beard")
            oval_face = Other("Oval face")
            pale_skin = Other("Pale skin")
        with gradio.Column(scale=1):
            pointy_nose = Other("Pointy nose")
            receding_hairline = Other("Receding hairline")
            rosy_cheeks = Other("Rosy cheeks")
            sideburns = Other("Sideburns")
            smiling = AdVerb("Smiling")
            straight_hair = Other("Straight hair")
            wavy_hair = Other("Wavy hair")
            wearing_earrings = AdVerb("Wearing earrings")
            wearing_hat = AdVerb("Wearing hat")
            wearing_lipstick = AdVerb("Wearing lipstick")
            wearing_necklace = AdVerb("Wearing necklace")
            wearing_necktie = AdVerb("Wearing necktie")
            young = AdVerb("Young")

    btn = gradio.Button("Run")

    gallery = gradio.Gallery(
        label = "Result image",
    ).style(
        columns = [3],
        height = "auto"
    )

    PERSON = "woman"
    if gender.value == "male":
        PERSON = "man"

    text = f"""This {PERSON} {five_o_clock_shadow.get_text()}."""

    text.submit(infer, inputs=text, outputs=gallery)
    btn.click(infer, inputs=text, outputs=gallery)

    gradio.Markdown("<p style='text-align: center'>© Fedora Yoshe Juandy</p>")


block.launch(enable_queue=False)
