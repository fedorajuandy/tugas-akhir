""" Gradio customised checkboxes """

from dataclasses import dataclass, field
import gradio


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

    def __init__(self, label):
        self.label = label
        self.val = False
        assert (
            self.label is not None
        ), "Label's name needs to be specified."

        self.checkbox = gradio.Checkbox(
            label = self.label,
        )


@dataclass
class AdVerb(Checkboxes):
    """ Checkboxes with adverb or verb attributes. """

    def get_text(self):
        """
        Function to return part of input text.
        """
        if self.checkbox:
            return f"is {self.label.lower()}"
        else:
            return ""


@dataclass
class Other(Checkboxes):
    """ Checkboxes with other attributes besides verbs and adverbs. """

    def get_text(self):
        """ Function to return part of final text. """
        if self.checkbox:
            return f"has {self.label.lower()}"
        else:
            return ""
