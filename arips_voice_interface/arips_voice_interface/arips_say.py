import os
import queue
import threading

import rclpy
import sounddevice as sd
from ament_index_python.packages import get_package_share_directory
from piper import PiperVoice
from rclpy.node import Node
from std_msgs.msg import String


class AripsSay(Node):
    def __init__(self):
        super().__init__('arips_say')
        default_model = os.path.join(
            get_package_share_directory('arips_voice_interface'),
            'models', 'en_US-joe-medium.onnx')
        self.declare_parameter('model', default_model)
        model = self.get_parameter('model').value

        # Load the model once and keep the audio stream open (low latency)
        self.voice = PiperVoice.load(model)
        self.stream = sd.RawOutputStream(
            samplerate=self.voice.config.sample_rate,
            channels=1, dtype='int16')
        self.stream.start()

        # The first inference is slower than later ones
        for _ in self.voice.synthesize('Ready.'):
            pass

        self.queue = queue.Queue()
        threading.Thread(target=self.worker, daemon=True).start()
        self.create_subscription(String, 'tts/say', self.on_text, 10)
        self.get_logger().info(
            f'Listening on topic {self.resolve_topic_name("tts/say")} '
            'for speaking')

    def on_text(self, msg):
        self.queue.put(msg.data)

    def worker(self):
        while True:
            text = self.queue.get()
            self.get_logger().info(f'Speaking: "{text}"')
            try:
                # Stream sentence by sentence
                for chunk in self.voice.synthesize(text):
                    self.stream.write(chunk.audio_int16_bytes)
            except Exception as e:
                self.get_logger().error(f'Speech synthesis failed: {e}')


def main():
    rclpy.init()
    node = AripsSay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
