""" Make dataset containing a pair of a caption and an encoded image """
# caption = string, encoded image = integers

from pathlib import Path
from functools import partial
import jax
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from tqdm.notebook import tqdm
import webdataset as wds
from vqgan_jax.modeling_flax_vqgan import VQModel
import pandas as pd


DATASET = "/content/drive/MyDrive/Colab Notebooks/dalle-mini/tools/dataset/data_256.tar"
output_location = Path("/content/output")

# Pre-trained VQGAN with 16384 vocabulary token
VQGAN_REPO, VQGAN_COMMIT_ID = (
    "dalle-mini/vqgan_imagenet_f16_16384",
    "85eb5d3b51a1c62a0cc8f4ccdee9882c0d0bd384",
)

BATCH_SIZE = 1
NUM_DEVICES = 1
TOTAL_BATCH_SIZE = BATCH_SIZE * jax.device_count()
SAVE_FREQUENCY = 1

datasets = (
    wds.WebDataset(DATASET) # Make WebDataset (PyTorch dataset) object by
    .decode("rgb") # decoding the image
    .to_tuple("jpg", "txt") # then make a list containing a pair of image and text
    .batched(TOTAL_BATCH_SIZE) # per batch
)

dl = (
    wds.WebLoader(datasets, BATCH_SIZE=None, NUM_DEVICES=1) # Make WebLOader object from
    .unbatched() # unbatching the data
    .batched(TOTAL_BATCH_SIZE) # batch again to avoid partial batch
)

vqgan = VQModel.from_pretrained("dalle-mini/vqgan_imagenet_f16_16384")
vqgan_params = replicate(vqgan.params)


# The function's computation is performed along 'batch' using 'pmap'
@partial(jax.pmap, axis_name="batch") # Distribute each 'batch' across devices
def p_encode(batch, params): # batch = dataset
    """ Function for parallel encoding using pre-trained VQGAN  """

    _, indices = vqgan.encode(batch, params=params) # quant_states is not used
    return indices


def encode_dataset(dataloader, output_dir, save_frequency):
    """ Function to encode all selected data """

    images, captions = next(iter(datasets))

    all_captions = []
    all_encoding = []
    n_file = 0

    for idx, (images, captions) in enumerate(tqdm(dataloader)):
        images = images.numpy()
        images = shard(images)
        encoded = p_encode(images, vqgan_params)
        encoded = encoded.reshape(-1, encoded.shape[-1])
        all_captions.extend(captions)
        all_encoding.extend(encoded.tolist())

        if (idx + 1) % save_frequency == 0:
            print(f"Saving file {n_file}")
            batch_df = pd.DataFrame.from_dict(
                {"caption": all_captions, "encoding": all_encoding}
            )
            batch_df.to_parquet(f"{output_dir}/{n_file:03d}.parquet")
            all_captions = []
            all_encoding = []
            n_file += 1

    if all_captions:
        # print(f"Saving final file {n_file}")
        batch_df = pd.DataFrame.from_dict(
            {"caption": all_captions, "encoding": all_encoding}
        )
        batch_df.to_parquet(f"{output_dir}/{n_file:03d}.parquet")


encode_dataset(dl, output_dir=output_location, save_frequency=SAVE_FREQUENCY)
