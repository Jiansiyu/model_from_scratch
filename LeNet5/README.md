# LetNet

```shell
# Build the image
docker build -t lenet-pytorch:latest .

# Train on GPU, mounting your local MNIST data and output dirs
docker run --gpus all \
  -v /path/to/mnist_data:/data \
  -v /path/to/output:/output \
  lenet-pytorch:latest \
  --epochs 20 --batch-size 128 --lr 0.005

```