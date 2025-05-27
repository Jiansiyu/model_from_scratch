
```shell

# Build the Docker image
docker build -t resnet-classifier .

# Train the model (uses config.yaml by default)
docker run --gpus all --rm -v $(pwd)/data:/app/data resnet-classifier

# Or run tests against a saved checkpoint
docker run --gpus all --rm \
  -v $(pwd)/checkpoints:/app/checkpoints \
  resnet-classifier \
  python test.py --checkpoint ./checkpoints/best.pth

```