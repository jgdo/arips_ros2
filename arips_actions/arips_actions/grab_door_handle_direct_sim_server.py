"""GrabDoorHandle action server for the Gazebo lu world that uses the known handle pose.

Serves the same GrabDoorHandle action under the same name ('grab_door_handle') as
grab_door_handle_server, but instead of running a learned policy it computes the gripper
pose directly from the fixed position of the door handle in the Gazebo world (door closed)
and moves the arm there with MoveIt (move_group action).

The arm makes these moves:
1. Pre-grab: the door gripper points straight at the door, its rotation axis on the handle
   spindle (raised by PRE_GRAB_Z_OFFSET), and its front edge pre_grab_distance in front of
   the handle lever. If this pose has no IK solution from the robot's current base pose,
   the goal fails without moving the arm. Before the arm moves, the door gripper joint
   turns to pre_grab_gripper_rotation (lever slot horizontal).
2. Forward: a straight Cartesian line (compute_cartesian_path, then execute_trajectory) of
   forward_distance along the gripper's pointing direction, starting from the arm's actual
   pose after move 1, orientation unchanged. Fails unless the whole line is feasible.
3. Rotate: the door gripper joint turns to gripper_rotation (MoveIt, door gripper group).

The goal's prompt is ignored; max_duration_s > 0 limits the arm motion (planning and
execution) instead of the execution_timeout_s parameter.
"""

import math
import threading
import time

import rclpy
from arips_action_msgs.action import GrabDoorHandle
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

# ---- Door in the Gazebo lu world (arips_gazebo/worlds/lu.sdf), closed -----------------------
# Model "door": hinge axis at DOOR_HINGE (map = Gazebo world frame), yaw 0. Closed, the leaf
# extends from the hinge along -x and its handle side faces -y.
DOOR_HINGE_X = -0.723
DOOR_HINGE_Y = 1.98
# Handle in door_solid_door.glb mesh coordinates (x: thickness, handle side +x; y: along
# the leaf, hinge edge at y = 0.854). The leaf visual maps mesh (x, y) to
# (hinge_x + y - 0.854, hinge_y + 0.045 - x).
MESH_HINGE_EDGE_Y = 0.854
MESH_DOOR_FACE_X = 0.045           # handle side face of the leaf
MESH_SPINDLE_Y = 0.064             # handle spindle (neck between rosette and lever)
MESH_SPINDLE_Z = 1.0445
MESH_LEVER_FACE_X = 0.112          # outermost face of the lever

# Handle spindle axis and lever face in the map frame
HANDLE_SPINDLE_X = DOOR_HINGE_X + MESH_SPINDLE_Y - MESH_HINGE_EDGE_Y      # -1.513
HANDLE_SPINDLE_Z = MESH_SPINDLE_Z                                         # 1.0445
HANDLE_LEVER_FACE_Y = DOOR_HINGE_Y + MESH_DOOR_FACE_X - MESH_LEVER_FACE_X  # 1.913
# Direction from the handle side towards the door (map frame)
APPROACH_YAW = math.pi / 2.0
# The gripper axis sits this much above the spindle axis in the pre-grab pose
PRE_GRAB_Z_OFFSET = -0.001

# ---- Door gripper geometry (arips_description) ---------------------------------------------
# door_gripper sits at this offset in link_3 (the IK tip of the arm group) and turns about
# its y axis; the gripper points along its -y axis and reaches GRIPPER_REACH beyond its origin.
GRIPPER_OFFSET_IN_LINK_3 = (0.0, -0.0533, -0.0248)
GRIPPER_REACH = 0.01


def pre_grab_link_3_pose(pre_grab_distance):
    """Return (x, y, z, yaw) of link_3 in the map frame for the pre-grab pose."""
    # Gripper origin on the spindle axis, its front edge pre_grab_distance before the lever
    # face: step back from the lever face (on the spindle axis) against the approach direction.
    back = pre_grab_distance + GRIPPER_REACH
    gx = HANDLE_SPINDLE_X - back * math.cos(APPROACH_YAW)
    gy = HANDLE_LEVER_FACE_Y - back * math.sin(APPROACH_YAW)
    gz = HANDLE_SPINDLE_Z + PRE_GRAB_Z_OFFSET

    # Gripper -y points along the approach direction, so its (and link_3's) yaw is
    # APPROACH_YAW + 90 deg. link_3 = gripper pose minus the rotated gripper offset.
    yaw = APPROACH_YAW + math.pi / 2.0
    ox, oy, oz = GRIPPER_OFFSET_IN_LINK_3
    c, s = math.cos(yaw), math.sin(yaw)
    x = gx - (c * ox - s * oy)
    y = gy - (s * ox + c * oy)
    z = gz - oz
    return x, y, z, math.atan2(math.sin(yaw), math.cos(yaw))


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GrabDoorHandleDirectSimServer(Node):
    def __init__(self):
        super().__init__('grab_door_handle_direct_sim_server')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_frame', 'map'),
                # MoveIt planning frame (root link of the robot model)
                ('planning_frame', 'base_link'),
                ('planning_group', 'arm'),
                ('ik_link', 'link_3'),
                ('move_group_action', '/move_action'),
                ('compute_ik_service', '/compute_ik'),
                # Gap between the gripper front and the handle lever in the pre-grab pose
                ('pre_grab_distance', 0.05),
                # Straight move of the gripper along its pointing direction after pre-grab
                ('forward_distance', 0.045),
                ('forward_velocity_scaling', 0.1),
                ('forward_acceleration_scaling', 0.1),
                # Cartesian interpolation step and max joint change between two steps
                ('forward_max_step', 0.002),
                ('forward_revolute_jump_threshold', 0.3),
                ('forward_prismatic_jump_threshold', 0.05),
                ('compute_cartesian_path_service', '/compute_cartesian_path'),
                ('gripper_group', 'door_gripper_group'),
                ('gripper_joint', 'joint_door_gripper'),
                # Door gripper joint position for the pre-grab pose [rad]
                ('pre_grab_gripper_rotation', 0.0),
                # Final door gripper joint position after the forward move [rad]
                ('gripper_rotation', 0.16),
                ('execute_trajectory_action', '/execute_trajectory'),
                ('ik_timeout_s', 1.0),
                ('ik_avoid_collisions', True),
                ('planning_time_s', 5.0),
                ('planning_attempts', 5),
                ('max_velocity_scaling', 0.5),
                ('max_acceleration_scaling', 0.5),
                # OMPL may end anywhere within this tolerance of the IK solution (m for the
                # slider, rad otherwise); the lever slot leaves only ~1 mm, so keep it tight.
                ('joint_goal_tolerance', 0.0005),
                ('execution_timeout_s', 60.0),
                ('server_wait_timeout_s', 5.0),
                ('tf_timeout_s', 1.0),
                ('cancel_timeout_s', 5.0),
            ],
        )
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self._map_frame = p('map_frame')
        self._planning_frame = p('planning_frame')
        self._group = p('planning_group')
        self._ik_link = p('ik_link')
        self._pre_grab_distance = p('pre_grab_distance')
        self._forward_distance = p('forward_distance')
        self._forward_velocity_scaling = p('forward_velocity_scaling')
        self._forward_acceleration_scaling = p('forward_acceleration_scaling')
        self._forward_max_step = p('forward_max_step')
        self._forward_revolute_jump = p('forward_revolute_jump_threshold')
        self._forward_prismatic_jump = p('forward_prismatic_jump_threshold')
        self._gripper_group = p('gripper_group')
        self._gripper_joint = p('gripper_joint')
        self._pre_grab_gripper_rotation = p('pre_grab_gripper_rotation')
        self._gripper_rotation = p('gripper_rotation')
        self._ik_timeout = p('ik_timeout_s')
        self._ik_avoid_collisions = p('ik_avoid_collisions')
        self._planning_time = p('planning_time_s')
        self._planning_attempts = p('planning_attempts')
        self._velocity_scaling = p('max_velocity_scaling')
        self._acceleration_scaling = p('max_acceleration_scaling')
        self._joint_tolerance = p('joint_goal_tolerance')
        self._execution_timeout = p('execution_timeout_s')
        self._server_wait_timeout = p('server_wait_timeout_s')
        self._tf_timeout = p('tf_timeout_s')
        self._cancel_timeout = p('cancel_timeout_s')

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Only one goal executes at a time; a new goal preempts the running one.
        self._execute_lock = threading.Lock()
        self._active_goal_id = None

        callback_group = ReentrantCallbackGroup()
        self._ik_client = self.create_client(
            GetPositionIK, p('compute_ik_service'), callback_group=callback_group)
        self._move_client = ActionClient(
            self, MoveGroup, p('move_group_action'), callback_group=callback_group)
        self._cartesian_client = self.create_client(
            GetCartesianPath, p('compute_cartesian_path_service'), callback_group=callback_group)
        self._execute_client = ActionClient(
            self, ExecuteTrajectory, p('execute_trajectory_action'),
            callback_group=callback_group)
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
        x, y, z, yaw = pre_grab_link_3_pose(self._pre_grab_distance)
        self.get_logger().info(
            f'GrabDoorHandle (direct sim) server ready, pre-grab {self._ik_link} in '
            f'{self._map_frame}: ({x:.3f}, {y:.3f}, {z:.3f}), yaw {math.degrees(yaw):.1f} deg')

    # ---- callbacks -------------------------------------------------------------------------

    def _goal_callback(self, _goal_request):
        return GoalResponse.ACCEPT

    def _handle_accepted(self, goal_handle):
        # Mark the newest goal as active; a running execute loop sees this and stops.
        self._active_goal_id = bytes(goal_handle.goal_id.uuid)
        goal_handle.execute()

    def _cancel_callback(self, _goal_handle):
        return CancelResponse.ACCEPT

    # ---- execution -------------------------------------------------------------------------

    def execute_callback(self, goal_handle):
        goal_id = bytes(goal_handle.goal_id.uuid)
        with self._execute_lock:
            if self._active_goal_id != goal_id:
                return self._abort(goal_handle, 'Preempted by a newer goal before starting')
            return self._execute(goal_handle, goal_id)

    def _execute(self, goal_handle, goal_id):
        self.get_logger().info('Executing GrabDoorHandle goal (direct sim)')
        self._publish_feedback(goal_handle, False)

        target = self._pre_grab_target()
        if target is None:
            return self._abort(
                goal_handle,
                f'No transform {self._map_frame} -> {self._planning_frame} for the pre-grab pose')

        if not self._ik_client.wait_for_service(timeout_sec=self._server_wait_timeout):
            return self._abort(goal_handle, 'compute_ik service not available')
        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = self._group
        ik_request.ik_request.ik_link_name = self._ik_link
        ik_request.ik_request.pose_stamped = target
        ik_request.ik_request.avoid_collisions = self._ik_avoid_collisions
        ik_request.ik_request.timeout = Duration(seconds=self._ik_timeout).to_msg()
        ik_request.ik_request.robot_state.is_diff = True
        ik_future = self._ik_client.call_async(ik_request)
        if not self._wait_for(ik_future, goal_handle, goal_id,
                              self._ik_timeout + self._server_wait_timeout):
            return self._stopped_or_abort(goal_handle, goal_id, 'Timed out waiting for IK')
        ik_response = ik_future.result()
        if ik_response.error_code.val != MoveItErrorCodes.SUCCESS:
            p = target.pose.position
            return self._abort(
                goal_handle,
                f'Pre-grab pose not reachable by IK (error code {ik_response.error_code.val}), '
                f'{self._ik_link} target in {self._planning_frame}: '
                f'({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')

        solution = dict(zip(ik_response.solution.joint_state.name,
                            ik_response.solution.joint_state.position))
        self.get_logger().info(
            'Pre-grab IK solution: '
            + ', '.join(f'{name}={value:.3f}' for name, value in solution.items()))

        if not self._move_client.wait_for_server(timeout_sec=self._server_wait_timeout):
            return self._abort(goal_handle, 'move_group action server not available')
        arm_solution = {name: position for name, position in solution.items()
                        if name in ('joint_slider', 'joint_2', 'joint_3', 'joint_4')}
        if not arm_solution:
            return self._abort(goal_handle, 'IK solution contains no joints of the arm group')
        deadline = None
        if goal_handle.request.max_duration_s > 0.0:
            deadline = time.monotonic() + goal_handle.request.max_duration_s

        # The gripper and the arm are separate groups and joint_trajectory_server runs one
        # command at a time, so the gripper turns first (this also releases the lever if a
        # previous goal left the gripper turned), then the arm moves.
        failure = self._run_move(
            self._joint_goal(
                self._gripper_group, {self._gripper_joint: self._pre_grab_gripper_rotation}),
            goal_handle, goal_id, deadline,
            f'pre-grab gripper rotation {self._pre_grab_gripper_rotation:.3f} rad')
        if failure is not None:
            return failure

        move_goal = self._joint_goal(self._group, arm_solution)
        failure = self._run_move(move_goal, goal_handle, goal_id, deadline, 'the pre-grab pose')
        if failure is not None:
            return failure
        self.get_logger().info('Arm is at the pre-grab pose')

        trajectory = self._forward_path(goal_handle, goal_id)
        if not isinstance(trajectory, ExecuteTrajectory.Goal):
            return trajectory
        if not self._execute_client.wait_for_server(timeout_sec=self._server_wait_timeout):
            return self._abort(goal_handle, 'execute_trajectory action server not available')
        failure = self._run_move(
            trajectory, goal_handle, goal_id, deadline,
            f'the pose {self._forward_distance:.3f} m forward', self._execute_client)
        if failure is not None:
            return failure
        self.get_logger().info(f'Moved {self._forward_distance:.3f} m forward')

        failure = self._run_move(
            self._joint_goal(self._gripper_group, {self._gripper_joint: self._gripper_rotation}),
            goal_handle, goal_id, deadline,
            f'gripper rotation {self._gripper_rotation:.3f} rad')
        if failure is not None:
            return failure
        self.get_logger().info(f'Rotated the gripper to {self._gripper_rotation:.3f} rad')

        self._publish_feedback(goal_handle, True)
        result = GrabDoorHandle.Result()
        result.error_code = GrabDoorHandle.Result.SUCCESS
        goal_handle.succeed()
        return result

    def _run_move(self, move_goal, goal_handle, goal_id, deadline, what, client=None):
        """Send a MoveGroup (or ExecuteTrajectory) goal and wait for it.

        Return None on success, else the action result to return.
        """
        client = client or self._move_client
        send_future = client.send_goal_async(move_goal)
        if not self._wait_for(send_future, goal_handle, goal_id, self._server_wait_timeout):
            if send_future.done() and send_future.result().accepted:
                self._cancel_move(send_future.result())
            return self._stopped_or_abort(
                goal_handle, goal_id, 'Timed out sending the move_group goal')
        move_handle = send_future.result()
        if not move_handle.accepted:
            return self._abort(goal_handle, 'move_group goal rejected')

        timeout = self._execution_timeout if deadline is None else deadline - time.monotonic()
        result_future = move_handle.get_result_async()
        if not self._wait_for(result_future, goal_handle, goal_id, max(timeout, 0.0)):
            self._cancel_move(move_handle)
            return self._stopped_or_abort(
                goal_handle, goal_id, f'Arm did not reach {what} in time')

        error_code = result_future.result().result.error_code.val
        if error_code != MoveItErrorCodes.SUCCESS:
            return self._abort(
                goal_handle, f'Moving to {what} failed (MoveIt error code {error_code})')
        return None

    def _forward_path(self, goal_handle, goal_id):
        """Plan the straight forward move of the IK link along the gripper direction.

        Starts from the actual pose of the IK link (TF) and keeps its yaw; the gripper points
        along the link's -y axis. Returns an ExecuteTrajectory goal, or the action result to
        return on failure.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                self._planning_frame, self._ik_link, Time(),
                timeout=Duration(seconds=self._tf_timeout))
        except TransformException as ex:
            return self._abort(
                goal_handle,
                f'No transform {self._planning_frame} -> {self._ik_link} for the forward '
                f'move: {ex}')
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = yaw_of(q)
        # -y axis of a frame with this yaw (roll/pitch are zero for this arm)
        dx, dy = math.sin(yaw), -math.cos(yaw)

        waypoint = PoseStamped().pose
        waypoint.position.x = t.x + self._forward_distance * dx
        waypoint.position.y = t.y + self._forward_distance * dy
        waypoint.position.z = t.z
        waypoint.orientation.z = math.sin(yaw / 2.0)
        waypoint.orientation.w = math.cos(yaw / 2.0)

        if not self._cartesian_client.wait_for_service(timeout_sec=self._server_wait_timeout):
            return self._abort(goal_handle, 'compute_cartesian_path service not available')
        request = GetCartesianPath.Request()
        request.header.frame_id = self._planning_frame
        request.start_state.is_diff = True
        request.group_name = self._group
        request.link_name = self._ik_link
        request.waypoints = [waypoint]
        request.max_step = self._forward_max_step
        request.jump_threshold = 0.0
        request.revolute_jump_threshold = self._forward_revolute_jump
        request.prismatic_jump_threshold = self._forward_prismatic_jump
        request.avoid_collisions = self._ik_avoid_collisions
        request.max_velocity_scaling_factor = self._forward_velocity_scaling
        request.max_acceleration_scaling_factor = self._forward_acceleration_scaling
        future = self._cartesian_client.call_async(request)
        if not self._wait_for(future, goal_handle, goal_id,
                              self._planning_time + self._server_wait_timeout):
            return self._stopped_or_abort(
                goal_handle, goal_id, 'Timed out planning the forward move')
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS or response.fraction < 0.999:
            return self._abort(
                goal_handle,
                f'Forward move of {self._forward_distance:.3f} m not feasible: only '
                f'{100.0 * response.fraction:.0f}% of the line planned '
                f'(MoveIt error code {response.error_code.val})')

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        return goal

    def _pre_grab_target(self):
        """Return the pre-grab pose of the IK link in the planning frame, or None.

        The arm can only set yaw about the robot's z axis, so the orientation is given as a
        pure yaw in the planning frame. A slightly tilted base does not make it unreachable.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                self._planning_frame, self._map_frame, Time(),
                timeout=Duration(seconds=self._tf_timeout))
        except TransformException as ex:
            self.get_logger().error(str(ex))
            return None

        x, y, z, yaw = pre_grab_link_3_pose(self._pre_grab_distance)
        t, q = tf.transform.translation, tf.transform.rotation
        # Rotate the map point into the planning frame (full rotation) and translate.
        qv = (q.x, q.y, q.z)
        uv = _cross(qv, (x, y, z))
        uuv = _cross(qv, uv)
        px = x + 2.0 * (q.w * uv[0] + uuv[0]) + t.x
        py = y + 2.0 * (q.w * uv[1] + uuv[1]) + t.y
        pz = z + 2.0 * (q.w * uv[2] + uuv[2]) + t.z
        local_yaw = yaw + yaw_of(q)

        target = PoseStamped()
        target.header.frame_id = self._planning_frame
        target.pose.position.x = px
        target.pose.position.y = py
        target.pose.position.z = pz
        target.pose.orientation.z = math.sin(local_yaw / 2.0)
        target.pose.orientation.w = math.cos(local_yaw / 2.0)
        return target

    def _joint_goal(self, group, positions):
        """Return a MoveGroup goal moving the joints of group to positions (name -> value)."""
        constraints = Constraints()
        for name, position in positions.items():
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = position
            joint.tolerance_above = self._joint_tolerance
            joint.tolerance_below = self._joint_tolerance
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)

        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = group
        request.start_state.is_diff = True
        request.goal_constraints.append(constraints)
        request.num_planning_attempts = self._planning_attempts
        request.allowed_planning_time = self._planning_time
        request.max_velocity_scaling_factor = self._velocity_scaling
        request.max_acceleration_scaling_factor = self._acceleration_scaling
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    def _publish_feedback(self, goal_handle, reached):
        feedback = GrabDoorHandle.Feedback()
        feedback.handle_reached = reached
        feedback.published_actions = 0
        goal_handle.publish_feedback(feedback)

    def _wait_for(self, future, goal_handle, goal_id, timeout):
        """Wait for a future; stop early if the goal is cancelled or preempted."""
        deadline = time.monotonic() + timeout
        while not future.done():
            if (goal_handle.is_cancel_requested or self._active_goal_id != goal_id
                    or time.monotonic() > deadline):
                return False
            time.sleep(0.01)
        return True

    def _cancel_move(self, move_handle):
        cancel_future = move_handle.cancel_goal_async()
        deadline = time.monotonic() + self._cancel_timeout
        while not cancel_future.done():
            if time.monotonic() > deadline:
                self.get_logger().warning('Timed out waiting for move_group cancel response')
                return
            time.sleep(0.01)

    def _stopped_or_abort(self, goal_handle, goal_id, error_str):
        if goal_handle.is_cancel_requested:
            return self._canceled(goal_handle)
        if self._active_goal_id != goal_id:
            return self._abort(goal_handle, 'Preempted by a newer goal')
        return self._abort(goal_handle, error_str)

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


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def main(args=None):
    rclpy.init(args=args)
    node = GrabDoorHandleDirectSimServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
