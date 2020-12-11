from is_wire.core import Channel, Subscription, Message, Logger
from utils import load_options

log = Logger(name='ConfigureCameras')

options = load_options()
c = Channel(options.broker_uri)
sb = Subscription(c)

cids = {}
for camera in options.cameras:
  log.info("Camera: {}\nConfiguration: {}", camera.id, camera.config)
  msg = Message()
  msg.pack(camera.config)
  msg.reply_to = sb
  msg.topic = 'CameraGateway.{}.SetConfig'.format(camera.id)
  c.publish(msg)
  cids[msg.correlation_id] = { 'camera': camera.id, 'ok': False }

while True:
    msg = c.consume()
    if msg.correlation_id in cids:
      camera = cids[msg.correlation_id]['camera']
      cids[msg.correlation_id]['ok'] = True
      log.info('Camera: {} Reply: {}', camera, msg.status)
    if all(map(lambda x: x[1]['ok'], cids.items())):
      break