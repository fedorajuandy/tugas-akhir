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
output_dir = Path("/content/output")

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
    wds.WebDataset(DATASET) # Make WebDataset (PyTorch dataset) object
    .decode("rgb")
    .to_tuple("jpg", "txt")
    .batched(TOTAL_BATCH_SIZE) # Per batch
)

data_loader = (
    wds.WebLoader(datasets, BATCH_SIZE=None, NUM_DEVICES=1) # Make WebLoader object
    .unbatched()
    .batched(TOTAL_BATCH_SIZE) # Batch again to avoid partial batch
)

vqgan = VQModel.from_pretrained("dalle-mini/vqgan_imagenet_f16_16384")
vqgan_params = replicate(vqgan.params)


# The function's computation is performed along 'batch' using 'pmap'
@partial(jax.pmap, axis_name="batch") # Distribute each 'batch' across devices
def p_encode(batch, params):
    """ Function for parallel encoding using pre-trained VQGAN  """

    _, indices = vqgan.encode(batch, params=params) # quant_states is not used
    return indices


def encode_dataset(dataloader, outputdir, safefrequency):
    """ Function to encode all selected data """

    images, captions = next(iter(datasets))

    all_captions = []
    all_encodings = []
    n_file = 0

    for idx, (images, captions) in enumerate(tqdm(dataloader)):
        images = images.numpy() # Convert tensor object into NumPy array
        n_images = len(images)

        # Take the number of dataset pairs according to batch size
        n_images_batch = n_images // BATCH_SIZE * BATCH_SIZE
        if n_images_batch != n_images:
            print(f"Different sizes {n_images_batch} (per batch) vs {n_images} (original)")
            images = images[:n_images_batch]
            captions = captions[:n_images_batch]

        images = shard(images)

        encoded = p_encode(images, vqgan_params)
        # 1D array becomes 2D with all elements intact
        encoded = encoded.reshape(-1, encoded.shape[-1])
        all_captions.extend(captions)
        all_encodings.extend(encoded.tolist())

        if (idx + 1) % safefrequency == 0:
            print(f"Saving file {n_file}")
            batch_df = pd.DataFrame.from_dict(
                {"caption": all_captions, "encoding": all_encodings}
            )
            batch_df.to_parquet(f"{outputdir}/{n_file:03d}.parquet")
            all_captions = []
            all_encodings = []
            n_file += 1

    if all_captions:
        print(f"Saving final file {n_file}")
        batch_df = pd.DataFrame.from_dict( # Create DataFrame from dictionary object; l
            {"caption": all_captions, "encoding": all_encodings}
        )
        batch_df.to_parquet(f"{outputdir}/{n_file:03d}.parquet")


encode_dataset(dataloader=data_loader, outputdir=output_dir, safefrequency=SAVE_FREQUENCY)
