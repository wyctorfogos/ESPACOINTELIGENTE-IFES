#!/usr/bin/env python
import pika
import time
import json


connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='192.168.1.107'))
channel = connection.channel()

channel.queue_declare(queue='Receive_information')
print(' [*] Waiting for messages. To exit press CTRL+C')



def callback(ch, method, properties, body):
    print(type(body))
    data_out=json.loads(body)
    print(type(data_out))
    print(data_out)

channel.basic_consume(queue='Receive_information', on_message_callback=callback,auto_ack=True)

channel.start_consuming()