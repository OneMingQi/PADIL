# Imitation Learning with Process Adversarial Diffusion


## Quick Start

### Installation
This project requires prior installation of PyTorch and MuJoCo. After installing these dependencies, follow these steps to set up the project environment:

1. Create the environment:
   ```bash
   conda env create -f environment.yml -n PADIL
   ```

2. Activate the environment:
   ```bash
   conda activate PADIL
   ```

3. Run the code:
   ```bash
   python ./main.py -e "./yamls/GDAIL/GDAIL_halfcheetah.yaml" -g 0
   ```
   Modify the parameter files in the `yamls` folder to run corresponding experiments.

### Key Code Files Overview
To quickly understand and utilize this codebase, it is recommended to focus on the following key files and directories:
- `main.py`: Base code for constructing experiments.
- `diffusion/`: Implementation of the diffusion networks used in this paper.
- `scripts/PADIL.py`: Implementation of the models, trainers, and discriminators used in this algorithm.
- `generator/`: Definitions of generators based on diffusion models.
- `algorithms/rlkit/torch/algorithms/adv_irl/adv_irl.py`: PADIL algorithm process built on the original rlkit framework, which can be modified to implement other comparative algorithms like WGAIL, FGAIL, etc.

## Acknowledgements
Thanks to the following researchers and developers for their excellent work, which this project's implementation references in strategy learning and generative models:
1. Zhendong Wang, Jonathan J Hunt, and Mingyuan Zhou. Diffusion policies as an expressive policy class for offline reinforcement learning. ICLR, 2023. [GitHub](https://github.com/zhendong-wang/diffusion-policies-for-offline-rl?tab=readme-ov-file)
2. Long Yang, Zhixiong Huang, Fenghao Lei, Yucun Zhong, Yiming Yang, Cong Fang, Shiting Wen, Binbin Zhou, and Zhouchen Lin. Policy representation via diffusion probability model for reinforcement learning. arXiv preprint arXiv:2305.13122, 2023. [GitHub](https://github.com/LongYang1998/Diffusion-Policy-Representation)
3. RLkit: Reinforcement learning framework and algorithms implemented in PyTorch. [GitHub](https://github.com/rail-berkeley/rlkit)
4. Bingzheng Wang and Guoqiang Wu. Diffail: Diffusion adversarial imitation learning. AAAI Conference on Artificial Intelligence, 2023. [GitHub](https://github.com/ML-Group-SDU/DiffAIL)