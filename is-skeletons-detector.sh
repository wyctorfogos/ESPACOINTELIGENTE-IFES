docker container run --rm -d \
  --memory=3g \
  --gpus=1 \
  --runtime=nvidia \
  --network=host \
  --name=sk1 \
  labviros/is-skeletons-detector:0.0.2-openpose ./rpc.bin #./stream.bin #./rpc.bin #

docker container run --rm -d \
  --memory=3g \
  --gpus=1 \
  --runtime=nvidia \
  --network=host \
  --name=sk2 \
  labviros/is-skeletons-detector:0.0.2-openpose ./rpc.bin #./stream.bin #./rpc.bin 
