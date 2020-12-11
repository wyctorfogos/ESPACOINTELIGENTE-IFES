from is_wire.core import Channel, Message, Logger
from is_wire.rpc import ServiceProvider, LogInterceptor
from is_msgs.image_pb2 import Image, ObjectAnnotations, ObjectAnnotation
from utils import load_options
import time
from random import randint

mean_time = 100 # milliseconds
var_time = 20

def detect(image, ctx):
    # simulate error
    time.sleep(randint(mean_time - var_time, mean_time + var_time) / 1000.0)
    reply = ObjectAnnotations(
        frame_id=randint(0, 4), objects=[ObjectAnnotation()] * randint(1, 3))
    return reply

options = load_options(print_options=False)

channel = Channel(options.broker_uri)
provider = ServiceProvider(channel)
provider.add_interceptor(LogInterceptor())

provider.delegate(
    topic='Skeletons.Detect',
    function=detect,
    request_type=Image,
    reply_type=ObjectAnnotations)

provider.run()