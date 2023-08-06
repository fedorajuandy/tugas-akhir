"""Orbax setup to redirect installers to checkpoint/export-specific libraries."""

import pathlib
import sys
import setuptools

# ingest readme for pypi description
this_directory = pathlib.Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

if __name__ == "__main__":
  # allows us to push to pypi
  sdist_mode = len(sys.argv) == 2 and sys.argv[1] == "sdist"

  # triggers error on import
  if not sdist_mode:
    sys.exit(
        "\n*** Orbax is a namespace, and not a standalone package. For model"
        " checkpointing and exporting utilities, please install"
        " `orbax-checkpoint` and `orbax-export` respectively (instead of"
        " `orbax`). ***\n"
    )

  setuptools.setup(
      name="orbax",
      version="0.1.8",
      author="Orbax Authors",
      author_email="orbax-dev@google.com",
      description="Orbax",
      long_description=long_description,
      long_description_content_type="text/markdown",
  )
