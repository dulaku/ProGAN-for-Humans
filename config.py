import argparse

parser = argparse.ArgumentParser()

##########################################################
# Training parameters; number of epochs, batch sizes, etc#
##########################################################
parser.add_argument(
    "--n_epochs",
    type=int,
    default=27, # 27 epochs is the smallest exceeding the 800k images/phase in the paper
    help="number of epochs of training at each resolution.",
)
parser.add_argument(
    "--batch_sizes",
    type=int,
    nargs="+",
    default=[64, 64, 32, 32, 16, 16, 8, 8, 4],
    help="A list of the batch sizes used in training; shrinks toward the end to save "
         "GPU memory. Should match the length of the blocks list below; not checked.",
)
parser.add_argument("--lr", type=float, default=0.001, help="adam learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.0,
    help="Adam decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.99,
    help="Adam decay of second order momentum of gradient",
)
parser.add_argument(
    "--lambda_gp",
    type=float,
    default=10.0,
    help="Multiplier for gradient penalty loss.",
)

#######################################################################
# Architecture configuration; channels in each block, latent size, etc#
#######################################################################
parser.add_argument(
    "--blocks",
    type=int,
    nargs="+",
    default=[512, 512, 512, 512, 256, 128, 64, 32, 16],
    help="A list of the number of channels in each block of the generator. The length "
         "of the list is also the depth of the generator (in blocks).",
)
parser.add_argument(
    "--latent_dim", type=int, default=512, help="Dimensionality of the latent space"
)
parser.add_argument(
    "--init_res",
    type=int,
    default=4,
    help="Resolution in pixels of the smallest images generated in training",
)

#################################
# Visualizer-specific parameters#
#################################
parser.add_argument(
    "--visualizer_decay",
    type=float,
    default=0.999,
    help="Momentum rate of updates to the visualizer network; high"
    "values mean small updates, thus high momentum.",
)
parser.add_argument(
    "--sample_layout",
    type=int,
    nargs="+",
    default=[3, 3],
    help="Number of rows, columns for sampling generator output.",
)
parser.add_argument(
    "--sample_interval",
    type=int,
    default=2000,
    help="Number of images shown to the discriminator between sample images."
)

##########################################################################
# Environment stuff; number of threads to use, filesystem locations, etc.#
##########################################################################
parser.add_argument(
    "--n_cpu",
    type=int,
    default=16,
    help="Number of CPU threads to use during batch generation",
)
parser.add_argument(
    "--load_location",
    type=str,
    default="progan.pt",
    help="File location from which to load pretrained models and metadata for "
    "restarting training."
)
parser.add_argument(
    "--save_location",
    type=str,
    default="progan.pt",
    help="File location for saving models after each resolution's training phase."
)
parser.add_argument(
    "--data_location",
    type=str,
    default="./celeba-1024",
    help="Location of directory containing PNG Celeb-A 1024x1024 dataset"
)
parser.add_argument(
    "--preprocessed_location",
    type=str,
    default="./celeba-resizes",
    help="Location to store preprocessed examples at various resolutions."
)
parser.add_argument(
    "--sample_location",
    type=str,
    default="./progan_images",
    help="Location to save sample images over the course of training"
)

cfg = parser.parse_args()
