""" Helper functions """

import wandb


def store_images(text_prompt, images, logits, n_predictions):
    """ For personal use and testing """
    project = 'dalle-mini-images'
    run = wandb.init(project=project)

    # Initialise an empty table
    columns = ["captions"] + ["image"]
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
