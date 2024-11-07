import torch
torch.cuda.empty_cache()
print(torch.cuda.is_available())



import argparse
from torch.utils.data import DataLoader

from models import *
from datasets import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1750, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=360, help="size of image height")
parser.add_argument("--img_width", type=int, default=360, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=250, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=1, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()

os.makedirs("imagesout/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
# Training data loader
dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))

def predict_image():
    """Saves a generated sample from the test set"""
    for i in iter(dataloader):
        print(i)
        imgs = i
        G_AB.eval()
        G_BA.eval()
        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = G_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = G_BA(real_B)
        # Arange images along x-axis

        # Arange images along y-axis

        save_image(fake_B, "imagesout/%s/%s.png" % (opt.dataset_name, i), normalize=False)
        # image_grid = torch.cat((fake_A), 1)
        save_image(fake_A, "imagesout/%s/%s-1.png" % (opt.dataset_name, i), normalize=False)

if __name__ =='__main__':

    predict_image()
