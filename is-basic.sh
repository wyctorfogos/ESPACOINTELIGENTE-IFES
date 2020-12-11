echo 'rabbitmq:3.7.6-management'
docker container run --rm -d \
    -v /home/julian/docker/is-k8s-deployments/ifes:/etc/rabbitmq/  \
    -p 5672:5672 \
    -p 15672:15672 \
    --network=host \
    --name rabbitmq \
    rabbitmq:3.7.6-management


echo ' openzipkin/zipkin'
docker container run --rm -d \
    -p 9411:9411 \
    --network=host \
    --name zipkin \
    openzipkin/zipkin


