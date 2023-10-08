import torch
from modules.utils import get_w, get_noise, gradient_penalty, generate_examples
import torchvision.utils as vutils




def train_model(
    critic, gen, path_length_penalty, loader, fixed_noise,
    opt_critic, opt_gen, opt_mapping_network, mapping_network,
    DEVICE, LAMBDA_GP, W_DIM, LOG_RESOLUTION, EPOCHS
):
    
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")

    for epoch in range(EPOCHS):

        if epoch % 100 == 0:
            generate_examples(gen, epoch, mapping_network, W_DIM, DEVICE, LOG_RESOLUTION)

        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(DEVICE)
            cur_batch_size = real.shape[0]

            w     = get_w(cur_batch_size, mapping_network, W_DIM, DEVICE, LOG_RESOLUTION)
            noise = get_noise(cur_batch_size, LOG_RESOLUTION, DEVICE)
            with torch.cuda.amp.autocast():
                fake = gen(w, noise)
                critic_fake = critic(fake.detach())
                
                critic_real = critic(real)
                gp = gradient_penalty(critic, real, fake, device=DEVICE)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
                )

            critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

            gen_fake = critic(fake)
            loss_gen = -torch.mean(gen_fake)

            if batch_idx % 16 == 0:
                plp = path_length_penalty(w, fake)
                if not torch.isnan(plp):
                    loss_gen = loss_gen + plp

            mapping_network.zero_grad()
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            opt_mapping_network.step()


            # Output training stats
            if batch_idx % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                    % (epoch, EPOCHS, batch_idx, len(loader),
                        loss_critic.item(), loss_gen.item()))

            # Save Losses for plotting later
            G_losses.append(loss_gen.item())
            D_losses.append(loss_critic.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == EPOCHS-1) and (batch_idx == len(loader)-1)):
                gen.eval()
                with torch.no_grad():
                    w     = get_w(fixed_noise[0][1].shape[0], mapping_network, W_DIM, DEVICE, LOG_RESOLUTION)
                    fake = gen(w, fixed_noise)
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    print("Finished training!")

    return gen, critic, G_losses, D_losses, img_list





