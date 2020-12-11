echo 'cam 0'
docker container run -d --rm \
    -v /home/julian/docker/options/0.json/:/options.json \
    --memory=60M \
    --network=host \
    --name cam0 \
    viros/is-cameras:1.5 ./service.bin

echo 'cam 1'
docker container run -d --rm \
    -v /home/julian/docker/options/1.json/:/options.json \
    --memory=60M \
    --network=host \
    --name cam1 \
    viros/is-cameras:1.5 ./service.bin

echo 'cam 2'
docker container run -d --rm \
    -v /home/julian/docker/options/2.json/:/options.json \
    --memory=60M \
    --network=host \
    --name cam2 \
    viros/is-cameras:1.5 ./service.bin

echo 'cam 3'
docker container run -d --rm \
    -v /home/julian/docker/options/3.json/:/options.json \
    --memory=60M \
    --network=host \
    --name cam3 \
    viros/is-cameras:1.5 ./service.bin



