"""Action server that grabs the door handle by running a learned policy.

GrabDoorHandle sends a RunPolicy goal and watches the arm's target_reached signal
(published by arips_lerobot's CommandRateLimiter). Once the signal has been true for
reached_hold_time_s, the policy is cancelled and the goal succeeds. Cancelling or
preempting the GrabDoorHandle goal also cancels the policy.
"""

import threading
import time

import rclpy
from arips_action_msgs.action import GrabDoorHandle
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rosetta_interfaces.action import RunPolicy
from std_msgs.msg import Bool


class GrabDoorHandleServer(Node):
    def __init__(self):
        super().__init__('grab_door_handle_server')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('run_policy_action', '/run_policy'),
                ('target_reached_topic', '/arm_forward_position_controller/target_reached'),
                ('default_prompt', 'Grab the door handle'),
                ('server_wait_timeout_s', 5.0),
                ('cancel_timeout_s', 5.0),
                # Ignore the reached signal right after the policy started; its first targets
                # may coincide with the current arm pose.
                ('min_policy_duration_s', 2.0),
                # The reached signal must stay true this long before the goal succeeds.
                ('reached_hold_time_s', 0.3),
                # A reached message older than this is treated as false.
                ('reached_signal_timeout_s', 0.5),
                ('control_rate_hz', 20.0),
            ],
        )
        self._default_prompt = self.get_parameter('default_prompt').value
        self._server_wait_timeout = self.get_parameter('server_wait_timeout_s').value
        self._cancel_timeout = self.get_parameter('cancel_timeout_s').value
        self._min_policy_duration = self.get_parameter('min_policy_duration_s').value
        self._reached_hold_time = self.get_parameter('reached_hold_time_s').value
        self._reached_signal_timeout = self.get_parameter('reached_signal_timeout_s').value
        control_rate_hz = self.get_parameter('control_rate_hz').value
        if control_rate_hz <= 0.0:
            raise ValueError('control_rate_hz must be greater than zero')
        self._control_period = 1.0 / control_rate_hz

        # Reached signal state, written by the subscription callback (monotonic timestamps).
        self._state_lock = threading.Lock()
        self._reached = False
        self._reached_since = None
        self._reached_stamp = None
        self._published_actions = 0

        # Only one goal executes at a time; a new goal preempts the running one.
        self._execute_lock = threading.Lock()
        self._active_goal_id = None

        callback_group = ReentrantCallbackGroup()
        self.create_subscription(
            Bool,
            self.get_parameter('target_reached_topic').value,
            self._on_reached,
            10,
            callback_group=callback_group,
        )
        self._policy_client = ActionClient(
            self,
            RunPolicy,
            self.get_parameter('run_policy_action').value,
            callback_group=callback_group,
        )
        self._action_server = ActionServer(
            self,
            GrabDoorHandle,
            'grab_door_handle',
            execute_callback=self.execute_callback,
            goal_callback=self._goal_callback,
            handle_accepted_callback=self._handle_accepted,
            cancel_callback=self._cancel_callback,
            callback_group=callback_group,
        )
        self.get_logger().info('GrabDoorHandle server ready')

    # ---- callbacks -------------------------------------------------------------------------

    def _on_reached(self, msg):
        now = time.monotonic()
        with self._state_lock:
            if msg.data and not self._reached:
                self._reached_since = now
            elif not msg.data:
                self._reached_since = None
            self._reached = msg.data
            self._reached_stamp = now

    def _goal_callback(self, _goal_request):
        return GoalResponse.ACCEPT

    def _handle_accepted(self, goal_handle):
        # Mark the newest goal as active; a running execute loop sees this and stops.
        self._active_goal_id = bytes(goal_handle.goal_id.uuid)
        goal_handle.execute()

    def _cancel_callback(self, _goal_handle):
        return CancelResponse.ACCEPT

    def _on_policy_feedback(self, feedback_msg):
        with self._state_lock:
            self._published_actions = feedback_msg.feedback.published_actions

    # ---- execution -------------------------------------------------------------------------

    def execute_callback(self, goal_handle):
        goal_id = bytes(goal_handle.goal_id.uuid)
        with self._execute_lock:
            if self._active_goal_id != goal_id:
                return self._abort(goal_handle, 'Preempted by a newer goal before starting')
            return self._execute(goal_handle, goal_id)

    def _execute(self, goal_handle, goal_id):
        request = goal_handle.request
        prompt = request.prompt or self._default_prompt
        self.get_logger().info(f'Executing GrabDoorHandle goal (prompt: "{prompt}")')

        if not self._policy_client.wait_for_server(timeout_sec=self._server_wait_timeout):
            return self._abort(goal_handle, 'RunPolicy action server not available')

        policy_goal = RunPolicy.Goal()
        policy_goal.prompt = prompt
        policy_goal.max_duration_s = request.max_duration_s
        with self._state_lock:
            self._published_actions = 0
        send_future = self._policy_client.send_goal_async(
            policy_goal, feedback_callback=self._on_policy_feedback)

        if not self._wait_for(send_future, goal_handle, goal_id, self._server_wait_timeout):
            if send_future.done() and send_future.result().accepted:
                self._cancel_policy(send_future.result())
            if goal_handle.is_cancel_requested:
                return self._canceled(goal_handle)
            if self._active_goal_id != goal_id:
                return self._abort(goal_handle, 'Preempted by a newer goal')
            return self._abort(goal_handle, 'Timed out sending RunPolicy goal')

        policy_handle = send_future.result()
        if not policy_handle.accepted:
            return self._abort(goal_handle, 'RunPolicy goal rejected')
        result_future = policy_handle.get_result_async()
        policy_start = time.monotonic()

        while True:
            if goal_handle.is_cancel_requested:
                self._cancel_policy(policy_handle)
                return self._canceled(goal_handle)
            if self._active_goal_id != goal_id:
                self._cancel_policy(policy_handle)
                return self._abort(goal_handle, 'Preempted by a newer goal')

            now = time.monotonic()
            reached, held = self._reached_state(now)
            if now - policy_start >= self._min_policy_duration and held:
                self.get_logger().info('Door handle reached, cancelling policy')
                self._cancel_policy(policy_handle)
                result = GrabDoorHandle.Result()
                result.error_code = GrabDoorHandle.Result.SUCCESS
                goal_handle.succeed()
                return result

            if result_future.done():
                policy_result = result_future.result().result
                detail = f': {policy_result.message}' if policy_result.message else ''
                return self._abort(
                    goal_handle,
                    'RunPolicy ended before the door handle was reached '
                    f'({policy_result.termination_reason}){detail}')

            feedback = GrabDoorHandle.Feedback()
            feedback.handle_reached = reached
            with self._state_lock:
                feedback.published_actions = self._published_actions
            goal_handle.publish_feedback(feedback)
            time.sleep(self._control_period)

    def _reached_state(self, now):
        """Return (reached, held): current signal value and whether it held long enough."""
        with self._state_lock:
            fresh = (self._reached_stamp is not None
                     and now - self._reached_stamp <= self._reached_signal_timeout)
            reached = fresh and self._reached
            held = (reached and self._reached_since is not None
                    and now - self._reached_since >= self._reached_hold_time)
        return reached, held

    def _wait_for(self, future, goal_handle, goal_id, timeout):
        """Wait for a future; stop early if the goal is cancelled or preempted."""
        deadline = time.monotonic() + timeout
        while not future.done():
            if (goal_handle.is_cancel_requested or self._active_goal_id != goal_id
                    or time.monotonic() > deadline):
                return False
            time.sleep(0.01)
        return True

    def _cancel_policy(self, policy_handle):
        cancel_future = policy_handle.cancel_goal_async()
        deadline = time.monotonic() + self._cancel_timeout
        while not cancel_future.done():
            if time.monotonic() > deadline:
                self.get_logger().warning('Timed out waiting for RunPolicy cancel response')
                return
            time.sleep(0.01)

    def _abort(self, goal_handle, error_str):
        result = GrabDoorHandle.Result()
        result.error_code = GrabDoorHandle.Result.FAILURE
        result.error_str = error_str
        goal_handle.abort()
        self.get_logger().error(error_str)
        return result

    def _canceled(self, goal_handle):
        result = GrabDoorHandle.Result()
        result.error_code = GrabDoorHandle.Result.FAILURE
        result.error_str = 'GrabDoorHandle canceled'
        goal_handle.canceled()
        self.get_logger().info(result.error_str)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = GrabDoorHandleServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
