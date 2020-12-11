import socket
from enum import Enum
from is_wire.core import Subscription, Message, Logger, Tracer
from is_wire.core.utils import now


class RequestManager:
    def __init__(self,
                 channel,
                 max_requests,
                 min_requests=None,
                 log_level=Logger.INFO,
                 zipkin_exporter=None):

        if min_requests is None:
            min_requests = max_requests
        if min_requests < 0:
            min_requests = 0
        if min_requests > max_requests:
            raise Exception("'min_requests' must be lower than 'max_requests'")

        self._channel = channel
        self._subscription = Subscription(self._channel)

        self._do_tracing = zipkin_exporter is not None
        self._zipkin_exporter = zipkin_exporter

        self._log = Logger(name='RequestManager')
        self._log.set_level(level=log_level)

        self._min_requests = min_requests
        self._max_requests = max_requests
        self._can_request = True

        self._requests = {}

    def can_request(self):
        return self._can_request

    def all_received(self):
        return len(self._requests) == 0

    def request(self, content, topic, timeout_ms, metadata=None):

        if not self.can_request():
            raise Exception("Can't request more than {}. Use 'RequestManager.can_request' "
                            "method to check if you can do requests.")

        tracer = Tracer(exporter=self._zipkin_exporter) if self._do_tracing else None
        span = tracer.start_span(name='request') if self._do_tracing else None

        msg = Message(content=content)
        msg.topic = topic
        msg.reply_to = self._subscription
        msg.timeout = timeout_ms / 1000.0

        self._log.debug("[Sending] metadata={}, cid={}", metadata, msg.correlation_id)

        if self._do_tracing:
            for key, value in (metadata or {}).items():
                span.add_attribute(key, value)
            tracer.end_span()
            msg.inject_tracing(span)

        self._publish(msg, metadata)

        if len(self._requests) >= self._max_requests:
            self._can_request = False

    def consume_ready(self, timeout=1.0):
        received_msgs = []

        # wait for new message
        try:
            stated_at = now()
            while True:
                _timeout = max(0.0, stated_at + timeout - now())
                msg = self._channel.consume(timeout=_timeout)

                if msg.status.ok() and msg.has_correlation_id():
                    cid = msg.correlation_id
                    if cid in self._requests:
                        received_msgs.append((msg, self._requests[cid]["metadata"]))
                        del self._requests[cid]

        except socket.timeout:
            pass

        # check for timeouted requests
        for cid in self._requests.keys():
            timeouted_msg = self._requests[cid]["msg"]

            if timeouted_msg.deadline_exceeded():
                msg = Message()
                msg.body = timeouted_msg.body
                msg.topic = timeouted_msg.topic
                msg.reply_to = self._subscription
                msg.timeout = timeouted_msg.timeout

                metadata = self._requests[cid]["metadata"]

                del self._requests[cid]

                self._log.debug("[Retring] metadata={}, cid={}", metadata, msg.correlation_id)
                self._publish(msg, metadata)

        if not self._can_request and len(self._requests) <= self._min_requests:
            self._can_request = True

        return received_msgs

    def _publish(self, msg, metadata):
        self._channel.publish(message=msg)
        self._requests[msg.correlation_id] = {
            "msg": msg,
            "metadata": metadata,
        }