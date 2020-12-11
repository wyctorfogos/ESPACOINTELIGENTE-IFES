docker container run -d --rm \
	-v /home/julian/docker/is-skeletons-grouper/options.json:/opt/is/options.json \
	--memory=2g \
	--cpus=1 \
	--network=host \
	--name=grouper \
	mendonca/is-skeletons-grouper:0.0.4-debug  ./rpc.bin /options.json  #./service.bin /options.json #./rpc.bin /options.json 
