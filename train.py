import torch
import torchvision
import numpy
import copy, os, datetime

import config
import progan_models
import progan_dataloader

# Make sure a directory exists for visualized samples
os.makedirs(config.cfg.sample_location, exist_ok=True)

####################
# Initialize models#
####################

generator = progan_models.Generator()
discriminator = progan_models.Discriminator()

# Make a copy of the generator, which we'll use to produce sample outputs to visualize
# training progress. After each batch, the visualizer's weights will be updated slightly
# in the direction of the current generator's weights. We'll disable tracking of
# gradients in the visualizer, since we never train the visualizer itself. We'll keep
# the visualizer on our second GPU, since Pytorch uses extra space on our first. Since
# we never compute gradients, it occupies such a small space that it easily fits. If you
# have only one GPU, consider moving this to CPU for a performance hit but more space on
# the GPU.
visualizer = copy.deepcopy(generator).to(device=1)
for p in visualizer.parameters():
    p.requires_grad_(False)

#######################################################################
# Try to load previous training results, or start fresh if that fails.#
#######################################################################

try:
    pretrained = torch.load(config.cfg.load_location)
except:
    pretrained = None

if pretrained is None:
    # Start training at the beginning since we don't have any prior history
    start_at = 0

    # Generate a single sample input for visualizing training results - this way we can
    # observe how the network behaves on a constant input. Values are between 0 and 1,
    # and the tensor is BxCx1x1, where B is the number of sample images we generate and
    # C is the size of the latent space.
    visualizer_sample = torch.FloatTensor(
        numpy.random.normal(
            0, 1, (config.cfg.sample_layout[0] * config.cfg.sample_layout[1],
                   config.cfg.latent_dim, 1, 1,),
        )
    ).to(device=1)
else:
    # Reload previously-trained weights. strict=False prevents erroring out due to the
    # deleted to/fromRGB layers.
    generator.load_state_dict(pretrained["generator"], strict=False)
    discriminator.load_state_dict(pretrained["discriminator"], strict=False)
    visualizer.load_state_dict(pretrained["visualizer"], strict=False)

    # Load the visualizer sample we used earlier for continuity in visualized samples
    visualizer_sample = pretrained["visualizer_sample"]

    # Note the resolution step we start training at
    start_at = pretrained["start_at"]

#####################################################
# Finish model parallelization and set up optimizers#
#####################################################

# Constants for normalization, used for visualization
img_mean = torch.tensor([0.5, 0.5, 0.5], requires_grad=False).to(visualizer_sample.device)
img_std_dev = torch.tensor([0.5, 0.5, 0.5], requires_grad=False).to(visualizer_sample.device)

# Apply the AccessibleDataParallel wrapper to parallelize the model across GPUs; you can
# remove these lines if you have only a single GPU.
generator = progan_models.AccessibleDataParallel(generator, (0, 1)).cuda()
discriminator = progan_models.AccessibleDataParallel(discriminator, (0, 1)).cuda()

optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=config.cfg.lr, betas=(config.cfg.b1, config.cfg.b2)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=config.cfg.lr, betas=(config.cfg.b1, config.cfg.b2)
)

############
# Training!#
############

images_seen = 0
runtimes = [] if pretrained is None else pretrained["runtimes"]

for res_step in range(len(config.cfg.blocks)):
    # Skip any resolutions a pretrained model already trained on
    if res_step < start_at:
        continue

    start = datetime.datetime.now()
    batch_size = config.cfg.batch_sizes[res_step]

    # At each resolution, we first fade that resolution's block in on top of the last
    # block, then stabilize with the newly faded-in block. top_blocks holds the two
    # blocks we need to keep track of that; the top two blocks during a fade-in and just
    # the top block when we are stabilizing. "top" here meaning the largest-resolution
    # so far involved in training.
    for top_blocks in ([res_step - 1, res_step], [res_step]):
        # The actual resolution of images at this step of training.
        resolution = config.cfg.init_res * (2 ** res_step)

        # Skip the blending phase of training the first block, since it has nothing to
        # be blended into.
        if res_step == 0 and len(top_blocks) == 2:
            continue

        # Configure a dataloader to provide images of the appropriate resolution.
        dataloader = torch.utils.data.DataLoader(
            progan_dataloader.CelebDataset(
                source_directory=config.cfg.data_location,
                resize_directory=config.cfg.preprocessed_location,
                resolution=resolution
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=16,   # Thread count; should be about the number of CPU cores
            pin_memory=True,  # Faster data transfer to GPU
            # Set drop_last to True since a batch size mismatch between real/fake images
            # breaks the gradient penalty computation without extra code to check.
            drop_last=True,
        )

        ############
        # Training #
        ############
        for epoch in range(config.cfg.n_epochs):
            for i, imgs in enumerate(dataloader):
                # Set the blending ratio between higher-resolution / lower-resolution
                # images. This slowly fades in the higher-resolution blocks over time.
                blend_ratio = (i + epoch * len(dataloader)) / (
                    config.cfg.n_epochs * len(dataloader)
                )

                real_imgs = imgs.cuda()

                # Simulate the gradual de-blockification of input images as the new
                # resolution gets phased in, so the generator's not tasked with
                # generating higher-resolution images than it's actually capable of
                # during early blending. Failure to do this could lead to extreme
                # outputs in the new block early in blending to compensate for the new
                # block's small contribution to the output image
                if len(top_blocks) == 2:
                    small_imgs = torch.nn.functional.avg_pool2d(real_imgs, 2)
                    small_imgs = torch.nn.functional.interpolate(
                        small_imgs, scale_factor=2.0, mode="nearest"
                    )
                    real_imgs = (
                        blend_ratio * real_imgs + (1.0 - blend_ratio) * small_imgs
                    )

                #################
                # Discriminator #
                #################

                # Zero out gradients from the previous batch; the update from those
                # gradients has already been applied and the gradients are no longer
                # valid.
                optimizer_D.zero_grad()

                # Sample noise for the generator's input. This sample is a point in what
                # is called the latent space - a 512-dimensional space that the generator
                # needs to learn how to map onto output images. Each dimension should
                # control some feature of the output, e.g. the direction a face looks.
                latent_sample = torch.cuda.FloatTensor(
                    numpy.random.normal(0, 1, (batch_size, config.cfg.latent_dim, 1, 1))
                )
                fake_imgs = generator(latent_sample, top_blocks, blend_ratio)

                d_loss = discriminator(fake_imgs, real_imgs, top_blocks, blend_ratio)

                # The discriminator computed the mean across samples on each GPU but
                # we still have to merge the GPUs results, hence the mean() below.
                d_loss = torch.mean(d_loss)
                d_loss.backward()
                optimizer_D.step()

                # Get just the numerical loss once we no longer need the graph; frees
                # up the memory before we compute the generator update.
                d_loss = d_loss.item()
                optimizer_D.zero_grad()
                discriminator.zero_grad()

                #############
                # Generator #
                #############

                # This looks the same as the process as above; there are secretly extra
                # steps in computing the loss for the discriminator, but they are built
                # into the discriminator model to parallelize them across GPUs.
                optimizer_G.zero_grad()
                fake_imgs = generator(latent_sample, top_blocks, blend_ratio)
                g_loss = discriminator(fake_imgs, None, top_blocks, blend_ratio)
                g_loss = torch.mean(g_loss)
                g_loss.backward()
                optimizer_G.step()
                g_loss = g_loss.item()
                optimizer_G.zero_grad()
                generator.zero_grad()

                with torch.no_grad():

                    # Update the visualizer weights slightly toward the generator
                    visualizer.momentum_update(generator, config.cfg.visualizer_decay)

                    # Print out a variety of diagnostic information and generate some
                    # sample images every config.cfg.sample_interval image samples.
                    # Since the sample interval doesn't have to be divisible by the
                    # batch size, we're technically checking if at _least_
                    # config.cfg.sample_interval images have been shown to the model.
                    old_modulo = images_seen % config.cfg.sample_interval
                    images_seen += batch_size
                    new_modulo = images_seen % config.cfg.sample_interval
                    if new_modulo < old_modulo or config.cfg.sample_interval < batch_size:
                        print(
                            "[Time: {!s}][Resolution {:04d}, "
                            "Top Blocks {!s}][Epoch {:03d}/{:03d}]"
                            "[Batch {:05d}/{:05d}][D loss: {:.4f}][G loss: {:.4f}]"
                            "[Blend: {:.4f}]".format(
                                str(datetime.datetime.now() - start),
                                resolution,
                                top_blocks,
                                epoch+1,
                                config.cfg.n_epochs,
                                i + 1,
                                len(dataloader),
                                d_loss,
                                g_loss,
                                blend_ratio if len(top_blocks) == 2 else 1.0,
                            )
                        )

                        sample_imgs = visualizer(visualizer_sample,
                                                 top_blocks,
                                                 blend_ratio)

                        # Manually denormalize to get a visualization that doesn't
                        # depend on the range of pixel values in the generated images,
                        # which it would if we used the automatic normalization in
                        # the torchvision save_image function.
                        sample_imgs.mul_(img_std_dev[:, None, None])
                        sample_imgs.add_(img_mean[:, None, None])

                        # Upscale the samples to the highest resolution the model will
                        # ever produce to make comparison easier.
                        final_res = config.cfg.init_res * (
                            2 ** (len(config.cfg.blocks) - 1)
                        )
                        scaled_samples = torch.nn.functional.interpolate(
                            sample_imgs.data,
                            size=(final_res, final_res),
                            mode="nearest"
                        )

                        filename = "celeba-{:04d}-{:s}-{:03d}-{:05d}.png".format(
                            resolution,
                            "0" if len(top_blocks) == 2 else "1",
                            epoch+1,
                            i+1
                        )
                        torchvision.utils.save_image(
                            scaled_samples,
                            os.path.join(config.cfg.sample_location, filename),
                            nrow=config.cfg.sample_layout[0],
                            normalize=False,
                            padding=2 * len(config.cfg.blocks),
                        )


    # After training at each resolution, store the amount of time spent at that
    # resolution and save the models.
    runtimes.append(datetime.datetime.now() - start)

    to_save = {"generator": generator.module.state_dict(),
               "discriminator": discriminator.module.state_dict(),
               "visualizer": visualizer.state_dict(),
               "visualizer_sample": visualizer_sample,
               "start_at": res_step + 1,
               "runtimes": runtimes,
               }
    torch.save(to_save, config.cfg.save_location)

# As a last touch, print the runtime spent at each resolution.
for resolution in range(len(config.cfg.blocks)):
    print(
        "Runtime for resolution %d: " % (config.cfg.init_res * (2 ** resolution)),
        runtimes[resolution],
    )
