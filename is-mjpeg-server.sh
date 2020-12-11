echo 'viros/is-mjpeg-server:1'

docker container run -d --rm \
    -p 3000:3000 \
    -e IS_URI=amqp://127.0.0.1:5672 \
    --memory=60M \
    --network=host \
    --name mjpeg-server \
    viros/is-mjpeg-server:1

echo 'google-chrome http://localhost:3000/0'
