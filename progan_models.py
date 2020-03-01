import torch

import progan_layers
import config

class AccessibleDataParallel(torch.nn.DataParallel):
    """
    A slight modification of PyTorch's default DataParallel wrapper object; this allows
    us to access attributes of a DataParallel-wrapped module.
    """
    def __getattr__(self, name):
        try:
            # Return an attribute defined in DataParallel
            return super().__getattr__(name)
        except:
            # Otherwise return the wrapped module's attribute of the same name
            return getattr(self.module, name)

class Generator(torch.nn.Module):
    """
    An image generator as described in the ProGAN paper. This model is composed of a
    set of blocks, each of which are trained in sequence. The first block converts a 1D
    input vector into a 4x4 featuremap; all other blocks upscale by a factor of 2 and
    apply additional convolution layers. Each block uses leaky ReLU activation (0.2 * x
    for x < 0, x otherwise) and pixelwise normalization (see the Pixnorm layer).

    Each block also has a toRGB layer which converts the output of that block to
    the RGB color space.
    """

    def __init__(self):
        super().__init__()

        self.toRGBs = []
        self.blocks = []

        def new_block(block_index):
            """Returns a block; we use a trick from the ProGAN paper to upscale and
            convolve at the same time in the first layer."""
            return torch.nn.Sequential(
                progan_layers.EqualizedConvTranspose2D(
                    in_channels=config.cfg.blocks[block_index - 1],
                    out_channels=config.cfg.blocks[block_index],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    upscale=True
                ),
                progan_layers.Pixnorm(),
                torch.nn.LeakyReLU(0.2, inplace=True),

                progan_layers.EqualizedConv2d(
                    in_channels=config.cfg.blocks[block_index],
                    out_channels=config.cfg.blocks[block_index],
                    kernel_size=3,
                    padding=1,
                ),
                progan_layers.Pixnorm(),
                torch.nn.LeakyReLU(0.2, inplace=True),
            )

        # Block 0
        self.blocks.append(
            torch.nn.Sequential(
                # This pixnorm layer converts gaussian noise inputs to "points on a
                # 512-dimensional hypersphere" as noted in the paper.
                progan_layers.Pixnorm(),

                # A 4x4 transposed convolution applied to a 1x1 input will yield a 4x4
                # output
                progan_layers.EqualizedConvTranspose2D(
                    in_channels=config.cfg.latent_dim,
                    out_channels=config.cfg.blocks[0],
                    kernel_size=config.cfg.init_res,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),
                progan_layers.Pixnorm(),

                progan_layers.EqualizedConv2d(
                    in_channels=config.cfg.blocks[0],
                    out_channels=config.cfg.blocks[0],
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),
                progan_layers.Pixnorm(),
            )
        )

        for block in range(len(config.cfg.blocks)):
            self.toRGBs.append(
                progan_layers.EqualizedConv2d(
                    in_channels=config.cfg.blocks[block],
                    out_channels=3,
                    kernel_size=1,
                    padding=0,
                )
            )

            # Don't add a new block on the last iteration, because the final block was
            # already here.
            if block < len(config.cfg.blocks) - 1:
                self.blocks.append(new_block(block + 1))

        # We need to register the blocks as modules for PyTorch to register their
        # weights as model parameters to optimize.
        self.toRGBs = torch.nn.ModuleList(self.toRGBs)
        self.blocks = torch.nn.ModuleList(self.blocks)

    def forward(self, latent_sample, top_blocks, blend_ratio):
        features = None
        lil_toRGB = self.toRGBs[top_blocks[0]]
        big_block = self.blocks[top_blocks[1]] if len(top_blocks) == 2 else None
        big_toRGB = self.toRGBs[top_blocks[1]] if len(top_blocks) == 2 else None

        for i, block in enumerate(self.blocks):
            features = block(features) if features is not None else block(latent_sample)
            if i == top_blocks[0]:
                if len(top_blocks) == 1:
                    return lil_toRGB(features)
                else:
                    trained_img = lil_toRGB(features)
                    trained_img = torch.nn.functional.interpolate(trained_img,
                                                                  scale_factor=2.0,
                                                                  mode="nearest")
                    new_img = big_toRGB(big_block(features))
                    return blend_ratio * new_img + (1.0 - blend_ratio) * trained_img

    def momentum_update(self, source_model, decay):
        """
        Updates the weights in self based on the weights in source_model.
        New weights will be decay * self's current weights + (1.0 - decay)
        * source_model's weights.

        This is used to make small updates to a visualizer network, moving each weight
        slightly toward the generator's weights. Doing it this way helps reduce
        artifacts and rapid changes in generated images, since we're averaging many
        states of the generator. The visualizer is what generates the sample images
        during training, and should probably be what's used for generation after
        training is complete.

        One thing worth noting is that images will appear less stable as training
        proceeds because, as the batch size shrinks, more updates to the visualizer are
        made between samples so this feature's effect on stability is diminished
        somewhat.
        """

        # Gets a dictionary mapping each generator parameter's name to the actual
        # parameter object. If you aren't using AccessibleDataParallel because you have
        # just one GPU, remove the "module" attribute and just use
        # dict(source_model.named_parameters())
        param_dict_src = dict(source_model.module.named_parameters())

        # For each parameter in the visualization model, get the same parameter in the
        # source model and perform the update.
        with torch.no_grad():
            for p_name, p_target in self.named_parameters():
                p_source = param_dict_src[p_name].to(p_target.device)
                p_target.copy_(decay * p_target + (1.0 - decay) * p_source)


class Discriminator(torch.nn.Module):
    """
    A discriminator between generated and real input images, as described in the ProGAN
    paper. This model is composed of a set of blocks, each of which are trained in
    sequence. Block 0 is the last block data sees, outputting the discriminator's score.
    But Block 0 is also trained first.

    The final block computes the mean standard deviation of pixel values across the
    batch, and adds that as an extra feature to the input, then applies 1 convolution.
    Next, the block applies an unpadded 4x4 kernel to the resulting 4x4 featuremap,
    resulting in a 1x1 output (with 512 channels in the default configuration). A final
    convolution to a 1-channel output yields the discriminator's score of the input's
    "realness". This last convolution is equivalent to a "fully-connected" layer and is
    what the paper actually did in its code.

    As with the generator, each block contains a convolution layer plus a layer that
    applies both a convolution and downsample at the same time. The same leaky ReLU
    activation is used, but unlike the generator pixelwise normalization is not.

    Each block also has a fromRGB layer which converts an RGB image sized for that block
    into a featuremap with the number of channels expected by the block's first layer.
    """
    def __init__(self):
        super().__init__()

        self.blocks = []
        self.fromRGBs = []

        def new_block(block_index):
            return torch.nn.Sequential(
                progan_layers.EqualizedConv2d(
                    in_channels=config.cfg.blocks[block_index],
                    out_channels=config.cfg.blocks[block_index - 1],
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),

                progan_layers.EqualizedConv2d(
                    in_channels=config.cfg.blocks[block_index - 1],
                    out_channels=config.cfg.blocks[block_index - 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    downscale=True
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),
            )

        self.blocks.append(
            torch.nn.Sequential(
                progan_layers.StandardDeviation(),
                progan_layers.EqualizedConv2d(
                    in_channels=config.cfg.blocks[0] + 1,  # +1 for std dev channel
                    out_channels=config.cfg.blocks[0],
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),

                # Input BxCx4x4; Output BxCx1x1
                progan_layers.EqualizedConv2d(
                    in_channels=config.cfg.blocks[0],
                    out_channels=config.cfg.blocks[0],
                    kernel_size=4,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),

                # Input BxCx1x1; collapsed to Bx1x1x1 - 1 score for each sample
                progan_layers.EqualizedConv2d(
                    in_channels=config.cfg.blocks[0], out_channels=1, kernel_size=1
                ),
            )
        )

        # We build the discriminator from back to front; this makes it a bit harder
        # to grok if you're examining debug info, since blocks[0] is the LAST block data
        # passes through, but it is considerably easier to read and follow the code.
        for block in range(len(config.cfg.blocks)):
            self.fromRGBs.append(
                progan_layers.EqualizedConv2d(
                    in_channels=3,
                    out_channels=config.cfg.blocks[block],
                    kernel_size=1,
                    padding=0,
                )
            )
            if block < len(config.cfg.blocks) - 1:
                self.blocks.append(new_block(block + 1))

        # As with the generator, convert our lists into ModuleLists to register them.
        self.blocks = torch.nn.ModuleList(self.blocks)
        self.fromRGBs = torch.nn.ModuleList(self.fromRGBs)

        # We'll also need a downscale layer for blending in training.
        self.halfsize = torch.nn.AvgPool2d(2)

        # Store some constants used in normalization of real samples
        self.mean = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5]),
                                       requires_grad=False)
        self.std_dev = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5]),
                                          requires_grad=False)

    def score_validity(self, img, top_blocks, blend_ratio):
        """
        This is the meat of the forward() method. The actual forward() method includes
        a few miscellaneous steps to support data preprocessing and the loss
        computation; it calls this when it needs to get a validity score for an input.
        """

        lil_fromRGB = self.fromRGBs[top_blocks[0]]
        big_block = self.blocks[top_blocks[1]] if len(top_blocks) == 2 else None
        big_fromRGB = self.fromRGBs[top_blocks[1]] if len(top_blocks) == 2 else None

        # The reverse of the generator - the layer we start training with depends on
        # which head is being trained, but we always proceed through to the end.
        if big_block is not None:
            features = big_fromRGB(img)
            features = big_block(features)
            trained_features = lil_fromRGB(self.halfsize(img))
            features = blend_ratio * features + (1.0 - blend_ratio) * trained_features
        else:
            features = lil_fromRGB(img)
        # The list slice here steps backward from the smaller-resolution of the top
        # blocks being trained
        for block in self.blocks[top_blocks[0]::-1]:
            features = block(features)

        # The view here just takes the output from Bx1x1x1 to B
        return features.view(-1)

    def forward(self, fake_img, real_imgs, top_blocks, blend_ratio):
        if real_imgs == None:
            # When we compute the generator's loss, we don't do anything fancy -
            # just return the negative of the discriminator's score
            return -torch.mean(self.score_validity(fake_img, top_blocks, blend_ratio))
        else:
            # Normalize manually so we can parallelize it on GPU instead of CPU
            # with torch.no_grad():
            #     real_imgs.sub_(self.mean[:, None, None])
            #     real_imgs.div_(self.std_dev[:, None, None])

            # Get the discriminator's opinion on a batch of fake images and one of real
            fake_validity = self.score_validity(fake_img, top_blocks, blend_ratio)
            real_validity = self.score_validity(real_imgs, top_blocks, blend_ratio)

            # WGAN style loss; we want the discriminator to output numbers as
            # negative as possible for fake_validity and as positive as possible
            # for real_validity
            wgan_loss = torch.mean(fake_validity) - torch.mean(real_validity)

            # Add a penalty for the discriminator having gradients far from 1 on images
            # composited from real and fake images (this keeps training from wandering
            # into unstable regions).
            gradient_penalty = config.cfg.lambda_gp * self.gradient_penalty(real_imgs,
                                                                            fake_img,
                                                                            top_blocks,
                                                                            blend_ratio)

            # Add a penalty for the discriminator's score on real images drifting
            # too far from 0; this helps keep the discriminator from being too
            # confident, which can result in near-0 gradients for the generator to
            # learn from. It also keeps numbers from getting big and overflowing.
            drift_penalty = 0.001 * torch.mean(real_validity ** 2.0)

            return wgan_loss + gradient_penalty + drift_penalty

    def gradient_penalty(self, real_samples, fake_samples, top_blocks, blend_ratio):
        """
        Computes a penalty to the discriminator for having gradients far from 1. This is
        desirable to keep training stable since parameter updates will have sane sizes.
        For a more mathematical explanation that is, frankly, over my head, read the
        WGAN-GP paper at https://arxiv.org/abs/1704.00028

        This method interpolates each real image with a generated one, with a random
        weight. The penalty is then computed from the gradient of the input with respect
        to the discriminator's score for that interpolated image. I can't explain why.
        """

        batch_size = real_samples.size(0)

        # Random weight for interpolation between real/fake samples; one weight for each
        # sample in the batch.
        image_weights = torch.rand((batch_size, 1, 1, 1)).to(fake_samples.device)

        # Compute the interpolations between the real and fake samples.
        interpolated = (
            image_weights * real_samples + ((1 - image_weights) * fake_samples)
        ).requires_grad_(True)

        interpolated_validity = self.score_validity(interpolated,
                                                    top_blocks,
                                                    blend_ratio)

        # Get gradient of input with respect to the interpolated images
        gradients = torch.autograd.grad(
            outputs=interpolated_validity,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_validity),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0].view(batch_size, -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
