echo 'is-tensorflow-gpu-jupyter-ppsus-labtef'
docker run -it --gpus all -p 8888:8888  --name tensorflow-gpu-jupyter-wyctor -v /home/julian/docker/ifes-2019-09-09/Modelos_para_treinamento:/tf/tensorflow-tutorials/ tensorflow/tensorflow:latest-gpu-jupyter
#docker run -it --rm --gpus all -d \
#    -v /home/julian/docker/ifes-2019-09-09/Modelos_para_treinamento:/tf/tensorflow-tutorials/  \
#    -p 8888:8888 \
#    --name tensorflow-gpu-jupyter \
#    tensorflow/tensorflow:latest-gpu-jupyter
