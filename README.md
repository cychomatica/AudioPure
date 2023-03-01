<!-- # Diffusion-Model-for-Audio-Defense

Please download the pretrained models [here](https://drive.google.com/drive/folders/1BqvUYf6aJSDLPL5GeRkb1HKzCF3NhCO3?usp=sharing).

After downloading, set the right path in function [`create_diffwave_model`](https://github.com/cychomatica/Diffusion-Model-for-Audio-Defense/blob/98f8ce7973926442f23eff0a9e603a1d40bbe5cf/diffusion_models/diffwave_ddpm.py#L242)
and [`create_model`](https://github.com/cychomatica/Diffusion-Model-for-Audio-Defense/blob/98f8ce7973926442f23eff0a9e603a1d40bbe5cf/audio_models/ConvNets_SpeechCommands/create_model.py#L7)
.

The conda environment setup is included in file [`environment.yml`](https://github.com/cychomatica/Diffusion-Model-for-Audio-Defense/blob/master/environment.yml). 
Run `conda env create -f environment.yml` to copy the environment. 

For certified robustness experiments, please run `scripts/certified_robust_eval.sh`. 
You can set certification method (DEFENSE), sampling number (N) and sigma. 
At present, we need to conduct **N=100,000** and **sigma=0.25, 0.5 and 1.0** for **DEFENSE=diffusion**.

Argument `--save_path` is the location where the experimental records are saved.  -->
# AudioPure-Pytorch
The official pytorch implementation of AudioPure proposed in paper **Defending against Adversarial Audio via Diffusion Model**

Accepted by **ICLR 2023** as a poster paper.
Paper link: ...

details work in progress...
