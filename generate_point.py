import argparse

from model.Latent_diffusion import LatentDiffusion
from model.HybridEncoder import HybridEncoder
from model.Unet import UNetModel
from model.Clip_embedder import CLIPTextEmbedder
from sampler.ddpm import DDPMSampler
from utils import *
from ud import generate_ud

parser = argparse.ArgumentParser(description="PyTorch stable diffusion model Training")
parser.add_argument("--data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    help="model architecture: " + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 128), this is the total "
         "batch size of all GPUs on the current node when "
         "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0,
    type=float,
    metavar="W",
    help="weight decay (default: 0.)",
    dest="weight_decay",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--pretrained", default=None, type=str, help="path to autoencoder pretrained checkpoint"
)
parser.add_argument(
    "--num_classes", default=10, type=int, help= "Number of classes to be classification."
)
parser.add_argument(
    "--n_steps", default=1000, type=int, help= "T in diffusion model."
)
parser.add_argument(
    "--save_dir", default="log", type=str, help="Diffusion model checkpoint dir path"
)
parser.add_argument(
    "--num", default=100, type=int, help= "Number of generated sample in each class."
)

if __name__ == '__main__':
    args = parser.parse_args()
    device =  'cuda:0'


    hybrid_encoder = HybridEncoder(args.num_classes, args.pretrained)
    Clip_emb = CLIPTextEmbedder("openai/clip-vit-large-patch14/clip-vit-large-patch14", device=device)
    Unet = UNetModel(in_channels=2048, out_channels=2048, channels=160, attention_levels=[0, 1], n_res_blocks=2,
                     channel_multipliers=[1, 2], n_heads=8, tf_layers=1, d_cond=768)
    hybrid_encoder.load_state_dict(torch.load('log/checkpoint.pth.tar', weights_only=True))

    Unet.load_state_dict(torch.load('log/unet.pth.tar', weights_only=True))
    hybrid_encoder.to(device)
    Unet.to(device)
    latent_dm = LatentDiffusion(Unet, hybrid_encoder, Clip_emb, 2, 1000, 0.0001,0.2)
    latent_dm.to(device)
    classname = label2text('cifar')
    dm = DDPMSampler(latent_dm)

    generated_data = None
    for i in range(10):
        # xt, label = torch.randn((args.num, 4, 32, 32), device= device), [i] # randomly generate
        # ud generate
        xt, label = torch.as_tensor(generate_ud(args.num, 4, 1024)).reshape([args.num, 2048, 8, 8]).to(device).float(), [i]
        cond = Clip_emb(get_text_labels(label, classname))
        for t in reversed(range(1000)):
            print('step: '+ str(t))
            xt_1= dm.p_sample(xt, cond, torch.tensor([t]).to(device), t)
            xt= xt_1[0]

        if generated_data is not None:
            generated_data = torch.concatenate([generated_data, xt])
        else:
            generated_data = xt
        torch.save(generated_data, 'log/sample.pth.tar')
    
    print('Have a nice day!')