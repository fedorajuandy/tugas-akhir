""" Helper functions """

import os
import wandb


def login_wandb():
    if(os.path.exists("/content")):
        from google.colab import drive
        drive.mount("/content/drive")

        os.environ["WANDB_NOTEBOOK_NAME"] = "/content/tugas-akhir/app/inference_pipeline.ipynb"
        with open("/content/drive/MyDrive/Colab Notebooks/wandb_auth_key.txt") as wak:
            wandb_auth_key = wak.readline()

    elif(os.path.exists("/kaggle")):
        from kaggle_secrets import UserSecretsClient
        os.environ["WANDB_NOTEBOOK_NAME"] = "/kaggle/working/tugas-akhir/app/inference_pipelineain.ipynb"
        wandb_auth_key = user_secrets.get_secret("wandb")

    else:
        wandb_auth_key = "wak"

    wandb.login(key=wandb_auth_key)


def store_images(text_prompt, images, logits, n_predictions):
    """ For personal use and testing """
    project = 'dalle-mini-images'
    run = wandb.init(project=project)

    # Initialise an empty table
    columns = ["captions"] + [f"image_{i+1}" for i in range(n_predictions)]
    gen_table = wandb.Table(columns=columns)

    # Add data to the table

    
    for i, prompt in enumerate(text_prompt):
        # if CLIP score exists
        if logits is not None:
            idxs = logits[i].argsort()[::-1]
            tmp_imgs = images[i::len(text_prompt)]
            tmp_imgs = [tmp_imgs[idx] for idx in idxs]
        else:
            tmp_imgs = images[i::len(text_prompt)]

        gen_table.add_data(prompt, *[wandb.Image(img) for img in tmp_imgs])

    # Log table to dashboard
    wandb.log({"Generated Images": gen_table})

    run.finish()
