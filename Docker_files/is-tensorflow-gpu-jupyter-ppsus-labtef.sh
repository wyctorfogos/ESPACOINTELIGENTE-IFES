echo 'is-tensorflow-gpu-jupyter-ppsus-labtef'
docker run -it -p 8080:8080 -v /home/julian/docker/ifes-2019-09-09/Modelos_para_treinamento:/tf/tensorflow-tutorials/ tensorflow/tensorflow:latest-gpu-py3-jupyter
#docker run -it --rm -d \
#    -v /home/julian/docker/ifes-2019-09-09/Modelos_para_treinamento:/tf/tensorflow-tutorials/  \
#    -p 8888:8888 \
#    --name tensorflow-gpu-jupyter \
#    tensorflow/tensorflow:latest-gpu-jupyter
