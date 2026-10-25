"""Analytic kinematics of the arips SCARA arm, derived from the URDF.

Expected chain (top-down view, z up):
    base_link -joint_slider (prismatic, z)-> link_0
              -joint_2 (revolute, z)-> link_1
              -joint_3 (revolute, z)-> link_2
              -joint_4 (revolute, z)-> link_3

joint_slider only moves along z, the other joints operate in the xy plane.
"""
import math

from urdf_parser_py.urdf import URDF

SLIDER, JOINT_2, JOINT_3, JOINT_4 = 'joint_slider', 'joint_2', 'joint_3', 'joint_4'
KINEMATIC_JOINTS = (SLIDER, JOINT_2, JOINT_3, JOINT_4)
TOL = 1e-6


class KinematicsError(RuntimeError):
    """The URDF does not describe the expected SCARA arm."""


class IKError(ValueError):
    """No valid joint solution exists for the requested pose."""


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi


def _rot(angle, x, y):
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


class ScaraKinematics:
    def __init__(self, urdf_xml, base_link='base_link', tip_link='link_3'):
        try:
            robot = URDF.from_xml_string(urdf_xml)
        except Exception as exc:
            raise KinematicsError(f'cannot parse robot description: {exc}') from exc
        self.base_link, self.tip_link = base_link, tip_link

        joints = self._chain(robot, base_link, tip_link)
        names = tuple(j.name for j in joints)
        if names != KINEMATIC_JOINTS:
            raise KinematicsError(
                f'kinematic chain {base_link} -> {tip_link} is {names}, '
                f'expected {KINEMATIC_JOINTS}; this is not the arips SCARA arm')
        slider, j2, j3, j4 = joints

        if slider.type != 'prismatic':
            raise KinematicsError(f'{SLIDER} must be prismatic, is {slider.type}')
        for j in (j2, j3, j4):
            if j.type != 'revolute':
                raise KinematicsError(f'{j.name} must be revolute, is {j.type}')

        self.axis_sign = {}
        for j in joints:
            axis = j.axis if j.axis is not None else [1.0, 0.0, 0.0]
            if abs(axis[0]) > TOL or abs(axis[1]) > TOL or abs(abs(axis[2]) - 1) > TOL:
                raise KinematicsError(f'{j.name} axis {axis} is not along z')
            self.axis_sign[j.name] = 1.0 if axis[2] > 0 else -1.0
        if self.axis_sign[SLIDER] < 0:
            raise KinematicsError(f'{SLIDER} must move along +z')

        self.limits = {}
        for j in joints:
            if j.limit is None or j.limit.lower is None or j.limit.upper is None:
                raise KinematicsError(f'{j.name} has no position limits')
            self.limits[j.name] = (j.limit.lower, j.limit.upper)

        origin = {}
        for j in joints:
            xyz = list(j.origin.xyz) if j.origin is not None else [0.0] * 3
            rpy = list(j.origin.rpy) if j.origin is not None else [0.0] * 3
            if abs(rpy[0]) > TOL or abs(rpy[1]) > TOL:
                raise KinematicsError(f'{j.name} origin is rotated about x/y; arm is not planar')
            origin[j.name] = (xyz[0], xyz[1], xyz[2], rpy[2])

        # Slider origin: base_link -> link_0, with fixed yaw and the z offset
        self.base_x, self.base_y, z0, self.yaw0 = origin[SLIDER]
        _, _, z2, _ = origin[JOINT_2]
        self.o2 = origin[JOINT_2][:2]
        self.yaw2 = origin[JOINT_2][3]
        self.link1 = origin[JOINT_3][0]
        self.yaw3 = origin[JOINT_3][3]
        self.link2 = origin[JOINT_4][0]
        self.yaw4 = origin[JOINT_4][3]
        for name, off in ((JOINT_3, origin[JOINT_3][1]), (JOINT_4, origin[JOINT_4][1])):
            if abs(off) > TOL:
                raise KinematicsError(f'{name} is not located along the x axis of its parent link')
        for name, length in ((JOINT_3, self.link1), (JOINT_4, self.link2)):
            if abs(length) < TOL:
                raise KinematicsError(f'link length of {name} is zero')
        self.z_offset = z0 + z2 + origin[JOINT_3][2] + origin[JOINT_4][2]

    @staticmethod
    def _chain(robot, base_link, tip_link):
        joints = []
        link = tip_link
        while link != base_link:
            if link not in robot.parent_map:
                raise KinematicsError(f'link {tip_link} is not connected to {base_link}')
            joint_name, parent = robot.parent_map[link]
            joints.append(robot.joint_map[joint_name])
            link = parent
        joints.reverse()
        return joints

    @property
    def joint_names(self):
        return KINEMATIC_JOINTS

    def fk(self, q):
        """Return (x, y, z, yaw) of the tip link in base_link. q: dict of joint positions."""
        s = self.axis_sign
        phi1 = self.yaw0 + self.yaw2 + s[JOINT_2] * q[JOINT_2]
        phi2 = phi1 + self.yaw3 + s[JOINT_3] * q[JOINT_3]
        phi3 = phi2 + self.yaw4 + s[JOINT_4] * q[JOINT_4]
        x, y = self._joint_2_position()
        dx, dy = _rot(phi1, self.link1, 0.0)
        x, y = x + dx, y + dy
        dx, dy = _rot(phi2, self.link2, 0.0)
        return x + dx, y + dy, self.z_offset + q[SLIDER], wrap_angle(phi3)

    def _joint_2_position(self):
        dx, dy = _rot(self.yaw0, *self.o2)
        return self.base_x + dx, self.base_y + dy

    def ik(self, x, y, z, yaw):
        """Joint positions reaching the tip link position (x, y, z) with the given yaw.

        The elbow side is chosen by the sign of q3: positive when the y axis of the tip link
        points towards -x of base_link, negative when it points towards +x.
        Raises IKError if unreachable or outside the joint limits.
        """
        s = self.axis_sign
        px, py = self._joint_2_position()
        dx, dy = x - px, y - py
        a, b = self.link1, self.link2
        cos_b = (dx * dx + dy * dy - a * a - b * b) / (2 * a * b)
        if abs(cos_b) > 1 + 1e-9:
            raise IKError(f'position ({x:.3f}, {y:.3f}) is out of reach')
        beta = math.acos(max(-1.0, min(1.0, cos_b)))

        want_positive = -math.sin(yaw) <= 0
        q3 = None
        for cand in (beta, -beta):
            q = wrap_angle(s[JOINT_3] * (cand - self.yaw3))
            if (q >= 0) == want_positive or abs(q) < 1e-12:
                q3, beta_sel = q, cand
                break
        if q3 is None:
            q3, beta_sel = wrap_angle(s[JOINT_3] * (beta - self.yaw3)), beta

        wx, wy = a + b * math.cos(beta_sel), b * math.sin(beta_sel)
        phi1 = math.atan2(dy, dx) - math.atan2(wy, wx)
        q2 = wrap_angle(s[JOINT_2] * (phi1 - self.yaw0 - self.yaw2))
        phi2 = phi1 + self.yaw3 + s[JOINT_3] * q3
        q4 = wrap_angle(s[JOINT_4] * (yaw - phi2 - self.yaw4))
        q = {SLIDER: z - self.z_offset, JOINT_2: q2, JOINT_3: q3, JOINT_4: q4}

        for name, value in q.items():
            lo, hi = self.limits[name]
            if value < lo - 1e-9 or value > hi + 1e-9:
                raise IKError(f'{name}={value:.3f} outside limits [{lo:.3f}, {hi:.3f}]')
        return q
