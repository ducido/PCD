# use python=3.10
# create a conda environment
# conda create -n pcd python=3.10
# conda activate pcd

uv venv --python 3.10

# install tensorflow
uv pip install tensorflow[and-cuda]==2.15.1
uv pip install git+https://github.com/nathanrooy/simulated-annealing 

# install jax
uv pip install "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install pytorch
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# install detectron2 (optional, for SED)
# pip install https://github.com/MiroPsota/torch_packages_builder/releases/download/detectron2-0.6%2B2a420ed/detectron2-0.6%2B2a420edpt2.0.0cu118-cp310-cp310-linux_x86_64.whl

# install maniskill env
cd third_party/ManiSkill2_real2sim
uv pip install -e .
cd ..

# install grounded sam 2
cd grounded_sam_2
uv pip install -e .
uv pip install --no-build-isolation -e grounding_dino
cd ..

# install SED (optional)
# cd SED/open_clip
# make install
# cd ../..

# install yolo world (optional)
# cd yolo_world
# pip install -e .
# cd ..

# install octo
cd octo
uv pip install -e .
cd ../..

# install requirements
uv pip install numpy==1.24.4
uv pip install -r third_party/SED/requirements.txt
uv pip install -r third_party/inpaint_anything/lama/requirements.txt
uv pip install -r requirements_full_install.txt

# reforce the installation of the following packages
uv pip install timm==0.9.10 accelerate==1.1.1
uv pip install "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
