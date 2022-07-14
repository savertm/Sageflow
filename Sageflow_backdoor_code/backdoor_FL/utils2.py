import copy
import torch





def get_poison_batch(batch, args, device, evaluation=False):

    images, targets = batch

    new_images = copy.deepcopy(images)
    new_targets = copy.deepcopy(targets)

    for index in range(0,len(images)):
        if evaluation:
            new_targets[index] = 2
            if args.dataset == 'mnist' or 'fmnist':
                new_images[index] = add_pixel_pattern_mnist(images[index])
            elif args.dataset == 'cifar':
                new_images[index] = add_pixel_pattern_cifar(images[index])

        else:
            if index < int(len(images) * args.num_img_backdoor/args.local_bs):
                new_targets[index] = 2
                if args.dataset == 'mnist' or 'fmnist':
                    new_images[index] = add_pixel_pattern_mnist(images[index])
                elif args.dataset == 'cifar':
                    new_images[index] = add_pixel_pattern_cifar(images[index])

            else:
                new_images[index] = images[index]
                new_targets[index] = targets[index]

    new_images = new_images.to(device)
    new_targets = new_targets.to(device).long()
    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)
    return new_images, new_targets


# Pixel pattern backdoor attack
def add_pixel_pattern_mnist(image_ori):
    image = copy.deepcopy(image_ori)
    image[0][0][0] = 1
    image[0][1][0] = 1
    image[0][2][0] = 1
    image[0][0][1] = 1
    image[0][1][2] = 1
    image[0][2][3] = 1
    image[0][0][5] = 1
    image[0][1][4] = 1
    image[0][4][0] = 1
    image[0][4][1] = 1
    image[0][4][2] = 1
    image[0][4][3] = 1

    return image

def add_pixel_pattern_cifar(image_ori):
    image = copy.deepcopy(image_ori)
    image[0][0][0] = 1
    image[1][0][0] = 1
    image[2][0][0] = 1

    image[0][1][0] = 1
    image[1][1][0] = 1
    image[2][1][0] = 1

    image[0][2][0] = 1
    image[1][2][0] = 1
    image[2][2][0] = 1


    image[0][0][1] = 1
    image[1][0][1] = 1
    image[2][0][1] = 1


    image[0][1][2] = 1
    image[1][1][2] = 1
    image[2][1][2] = 1


    image[0][2][3] = 1
    image[1][2][3] = 1
    image[2][2][3] = 1



    #image[0][0][6] = 1
    #image[0][2][6] = 1
    image[0][0][5] = 1
    image[1][0][5] = 1
    image[2][0][5] = 1


    #image[0][1][6] = 1
    image[0][1][4] = 1
    image[1][1][4] = 1
    image[2][1][4] = 1


    image[0][4][0] = 1
    image[1][4][0] = 1
    image[2][4][0] = 1


    image[0][4][1] = 1
    image[1][4][1] = 1
    image[2][4][1] = 1


    image[0][4][2] = 1
    image[1][4][2] = 1
    image[2][4][2] = 1


    image[0][4][3] = 1
    image[1][4][3] = 1
    image[2][4][3] = 1

    return image






def model_dist_norm(model, target_params):
    squared_sum = 0
    for name, layer in model.named_parameters():
        squared_sum.append(torch.max(torch.abs(layer.data - target_params[name].data)))
    return squared_sum

