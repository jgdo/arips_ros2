import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from rosetta_interfaces.action import RecordEpisode
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from std_srvs.srv import Trigger

START_BUTTON = 7
FINISH_BUTTON = 0
DISCARD_BUTTON = 1
NUM_BUTTONS = 16


class RecordControl(Node):
    """Control rosetta episode recordings from joystick buttons."""

    def __init__(self, prompt):
        super().__init__('record_control')
        self.prompt = prompt
        self.client = ActionClient(
            self, RecordEpisode, '/record_episode')
        self.delete_client = self.create_client(
            Trigger, '/episode_recorder/delete_last_bag')
        self.tts_publisher = self.create_publisher(String, '/tts/say', 10)
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.prev = [0] * NUM_BUTTONS
        # idle | starting | recording
        self.state = 'idle'
        self.goal_handle = None
        self.stop_requested = None  # None, 'save' or 'discard'
        self.get_logger().info(
            f'Ready. Prompt: "{prompt}". Button {START_BUTTON}: start, '
            f'{FINISH_BUTTON}: finish, {DISCARD_BUTTON}: discard.')

    def joy_callback(self, msg):
        buttons = (list(msg.buttons) + [0] * NUM_BUTTONS)[:max(
            NUM_BUTTONS, len(msg.buttons))]
        prev = self.prev + [0] * (len(buttons) - len(self.prev))

        def pressed(i):
            return bool(buttons[i]) and not prev[i]

        if pressed(START_BUTTON):
            self.start()
        if pressed(FINISH_BUTTON):
            self.stop('save')
        if pressed(DISCARD_BUTTON):
            self.stop('discard')
        self.prev = buttons

    def start(self):
        if self.state != 'idle':
            self.get_logger().info('Recording is already running')
            return
        if not self.client.server_is_ready():
            self.get_logger().error('Record action server not available')
            return
        goal = RecordEpisode.Goal()
        goal.prompt = self.prompt
        self.state = 'starting'
        self.stop_requested = None
        self.get_logger().info('Starting recording')
        self.client.send_goal_async(goal).add_done_callback(self.goal_response)

    def goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Recording goal rejected')
            self.reset()
            return
        self.goal_handle = handle
        self.state = 'recording'
        self.say('Episode started')
        handle.get_result_async().add_done_callback(self.result_callback)
        if self.stop_requested:
            self.cancel_goal()

    def stop(self, mode):
        if self.state == 'idle':
            self.get_logger().info('Not recording right now')
            return
        if self.stop_requested == 'discard' or (
                self.stop_requested == 'save' and mode == 'save'):
            return
        self.stop_requested = mode
        self.get_logger().info(
            'Finishing recording' if mode == 'save' else 'Discarding recording')
        if self.state == 'recording':
            self.cancel_goal()

    def cancel_goal(self):
        self.goal_handle.cancel_goal_async()

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f'Recording ended ({result.termination_reason}): '
            f'{result.bag_path}, {result.messages_written} messages')
        discard = self.stop_requested == 'discard'
        if self.stop_requested == 'save' or (
                self.stop_requested is None and
                result.termination_reason in ('stopped', 'timeout')):
            self.say('Episode finished')
        else:
            self.say('Episode aborted')
        self.reset()
        if discard:
            self.delete_last_bag()

    def say(self, message):
        msg = String()
        msg.data = message
        self.tts_publisher.publish(msg)

    def delete_last_bag(self):
        if not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('delete_last_bag service not available')
            return
        self.delete_client.call_async(Trigger.Request()).add_done_callback(
            self.delete_response)

    def delete_response(self, future):
        res = future.result()
        if res.success:
            self.get_logger().info(f'Recording discarded: {res.message}')
        else:
            self.get_logger().error(f'Discard failed: {res.message}')

    def reset(self):
        self.state = 'idle'
        self.goal_handle = None
        self.stop_requested = None


def main(args=None):
    argv = remove_ros_args(sys.argv if args is None else args)[1:]
    if not argv or not argv[0].strip():
        print('Usage: ros2 run arips_teleop record_control "<task prompt>"',
              file=sys.stderr)
        sys.exit(1)
    rclpy.init(args=args)
    node = RecordControl(' '.join(argv))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
