""" Gradio customised checkboxes """

import gradio
from dataclasses import dataclass, field


@dataclass
class Checkboxes:
    """ Return value from each checkboxes. """

    label: str = field(
        default = None,
        metadata = {"help": "The shown label in app."},
    )
    checkbox: gradio.Checkbox = field(
        default = None,
        metadata = {"help": "A Gradio checkbox."}
    )
    val: bool = field(
        default = False,
        metadata = {"help": "Whether it is checked."}
    )

    def __init__(self, label):
        self.label = label
        self.val = False
        assert (
            self.label is not None
        ), "Label's name needs to be specified."

        self.checkbox = gradio.Checkbox(
            label = self.label,
            onchange=checkbox_value
        )

    def checkbox_value(value):
        val = value
        return value

@dataclass
class AdVerb(Checkboxes):
    """ Checkboxes with adverb or verb attributes. """

    def __init__(self, label):
        super().__init__(label)

    def get_text(self):
        """
        Function to return part of input text.
        """
        if self.checkbox_value():
            return f"is {self.label.lower()}"
        else:
            return ""

@dataclass
class Other(Checkboxes):
    """ Checkboxes with other attributes besides verbs and adverbs. """

    def __init__(self, label):
        super().__init__(label)

    def get_text(self):
        """ Function to return part of final text. """
        if self.checkbox_value():
            return f"has {self.label.lower()}"
        else:
            return ""
