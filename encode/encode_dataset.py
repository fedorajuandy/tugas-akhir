""" Encode a pair of image and text into encoded one """

from pathlib import Path
from functools import partial
import os.path
import jax
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from tqdm.notebook import tqdm
import webdataset as wds
import braceexpand
from vqgan_jax.modeling_flax_vqgan import VQModel
import pandas as pd

DATASETS = "{000..29999}.parquet"
SHARDS = "/content/drive/MyDrive/Colab Notebooks/dalle-mini/tools/dataset/data_256.tar"
encoded_output = Path("/content/output")

VQGAN_REPO, VQGAN_COMMIT_ID = (
    "dalle-mini/vqgan_imagenet_f16_16384",
    "85eb5d3b51a1c62a0cc8f4ccdee9882c0d0bd384",
)

BATCH_SIZE = 1
NUM_WORKERS = 1
TOTAL_BS = BATCH_SIZE * jax.device_count()
SAVE_FREQUENCY = 1

SHARDS = list(
    braceexpand.braceexpand(SHARDS)
)

ds = (
    wds.WebDataset(SHARDS, handler=wds.warn_and_continue)
    .decode("rgb", handler=wds.warn_and_continue)
    .to_tuple("jpg", "txt")
    .batched(TOTAL_BS)
)



dl = (
    wds.WebLoader(ds, BATCH_SIZE=None, NUM_WORKERS=2).unbatched().batched(TOTAL_BS)
)

vqgan = VQModel.from_pretrained("dalle-mini/vqgan_imagenet_f16_16384")
vqgan_params = replicate(vqgan.params)

@partial(jax.pmap, axis_name="batch")
def p_encode(batch, params):
    """ Function to  """
    _, indices = vqgan.encode(batch, params=params)
    return indices

def encode_dataset(dataloader, output_dir, save_frequency):
    """ Function to """
    images, captions = next(iter(ds))

    output_dir.mkdir(parents=True, exist_ok=True)
    all_captions = []
    all_encoding = []
    n_file = 0
    nfile = 0
    file_exists = os.path.exists(f"{output_dir}/{nfile}.parquet")

    for idx, () in enumerate(tqdm(dataloader)):
        images = images.numpy()
        n = len(images)

        if n != len(images):
            print(f"Different sizes {n} vs {len(images)}")
            images = images[:n]
            captions = captions[:n]
            n_file += 1
            file_exists = os.path.exists(f"{output_dir}/{nfile}.parquet")

        if not captions or file_exists:
            print("No images/captions in batch...")
            continue

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
        print(f"Saving final file {n_file}")
        batch_df = pd.DataFrame.from_dict(
            {"caption": all_captions, "encoding": all_encoding}
        )
        batch_df.to_parquet(f"{output_dir}/{n_file:03d}.parquet")

encode_dataset(dl, output_dir=encoded_output, save_frequency=SAVE_FREQUENCY)
