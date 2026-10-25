import math
import random

import pytest

from arips_teleop.scara_kinematics import (
    IKError, JOINT_2, JOINT_3, JOINT_4, KinematicsError, ScaraKinematics, SLIDER)

URDF = """<robot name="scara">
  <link name="base_link"/><link name="link_0"/><link name="link_1"/>
  <link name="link_2"/><link name="link_3"/>
  <joint name="joint_slider" type="prismatic">
    <origin xyz="0.14 0.135 0.292" rpy="0 0 1.5708"/><axis xyz="0 0 1"/>
    <parent link="base_link"/><child link="link_0"/>
    <limit effort="1" velocity="1" lower="0" upper="0.871"/>
  </joint>
  <joint name="joint_2" type="revolute">
    <origin xyz="-0.0962 0 0.06555" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <parent link="link_0"/><child link="link_1"/>
    <limit effort="1" velocity="1" lower="-2.26893" upper="2.26893"/>
  </joint>
  <joint name="joint_3" type="revolute">
    <origin xyz="-0.17 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <parent link="link_1"/><child link="link_2"/>
    <limit effort="1" velocity="1" lower="-2.44346" upper="2.44346"/>
  </joint>
  <joint name="joint_4" type="revolute">
    <origin xyz="-0.15 0 -0.0516" rpy="0 0 -1.5708"/><axis xyz="0 0 -1"/>
    <parent link="link_2"/><child link="link_3"/>
    <limit effort="1" velocity="1" lower="-2.0944" upper="2.0944"/>
  </joint>
</robot>"""


@pytest.fixture
def kin():
    return ScaraKinematics(URDF)


def test_home_pose(kin):
    x, y, z, yaw = kin.fk({SLIDER: 0.0, JOINT_2: 0.0, JOINT_3: 0.0, JOINT_4: 0.0})
    assert x == pytest.approx(0.14, abs=1e-4)
    assert y == pytest.approx(0.135 - 0.0962 - 0.17 - 0.15, abs=1e-4)
    assert z == pytest.approx(0.292 + 0.06555 - 0.0516)
    assert yaw == pytest.approx(0.0, abs=1e-3)


def test_ik_fk_round_trip_and_sign_rule(kin):
    rng = random.Random(1)
    checked = 0
    for _ in range(2000):
        q = {
            SLIDER: rng.uniform(0, 0.871),
            JOINT_2: rng.uniform(-2.2, 2.2),
            JOINT_3: rng.uniform(-2.4, 2.4),
            JOINT_4: rng.uniform(-2.0, 2.0),
        }
        pose = kin.fk(q)
        try:
            sol = kin.ik(*pose)
        except IKError:
            # The rule picks the other elbow side, which may violate limits
            continue
        checked += 1
        back = kin.fk(sol)
        assert back[:3] == pytest.approx(pose[:3], abs=1e-9)
        assert math.cos(back[3] - pose[3]) == pytest.approx(1.0, abs=1e-9)
        if -math.sin(pose[3]) > 1e-9:
            assert sol[JOINT_3] <= 0
        elif -math.sin(pose[3]) < -1e-9:
            assert sol[JOINT_3] >= 0
    assert checked > 200


def test_unreachable(kin):
    with pytest.raises(IKError):
        kin.ik(2.0, 2.0, 0.3, 0.0)


def test_joint_limit_rejected(kin):
    with pytest.raises(IKError):
        kin.ik(0.14, -0.3, 5.0, 0.0)


def test_closest_reports_true_error(kin):
    rng = random.Random(2)
    for _ in range(200):
        target = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.2, 1.2),
                  rng.uniform(-math.pi, math.pi))
        q, err = kin.ik_closest(*target)
        x, y, z, _ = kin.fk(q)
        assert math.dist((x, y, z), target[:3]) == pytest.approx(err, abs=1e-9)
        for name, value in q.items():
            lo, hi = kin.limits[name]
            assert lo - 1e-9 <= value <= hi + 1e-9


def test_closest_is_exact_when_reachable(kin):
    q = {SLIDER: 0.3, JOINT_2: 0.4, JOINT_3: -0.9, JOINT_4: 0.2}
    pose = kin.fk(q)
    sol, err = kin.ik_closest(*pose)
    assert err == 0.0
    assert kin.fk(sol)[:3] == pytest.approx(pose[:3], abs=1e-9)


def test_closest_beats_grid_search(kin):
    target = (0.1, -0.9, 0.5, 0.0)  # beyond reach
    _, err = kin.ik_closest(*target)
    px, py = kin._joint_2_position()
    best = float('inf')
    lo2, hi2 = kin.limits[JOINT_2]
    lo3, hi3 = kin.limits[JOINT_3]
    for i in range(201):
        for j in range(201):
            q = {SLIDER: 0.2, JOINT_2: lo2 + (hi2 - lo2) * i / 200,
                 JOINT_3: lo3 + (hi3 - lo3) * j / 200, JOINT_4: 0.0}
            x, y, _, _ = kin.fk(q)
            best = min(best, math.hypot(x - target[0], y - target[1]))
    assert err <= best + 2e-3


@pytest.mark.parametrize('old, new', [
    ('name="joint_2" type="revolute"', 'name="joint_2" type="fixed"'),
    ('<axis xyz="0 0 -1"/>', '<axis xyz="0 1 0"/>'),
    ('<origin xyz="-0.17 0 0" rpy="0 0 0"/>', '<origin xyz="-0.17 0.05 0" rpy="0 0 0"/>'),
    ('<origin xyz="-0.0962 0 0.06555" rpy="0 0 0"/>',
     '<origin xyz="-0.0962 0 0.06555" rpy="0.5 0 0"/>'),
    ('<parent link="link_2"/>', '<parent link="link_1"/>'),
])
def test_wrong_chain_aborts(old, new):
    assert old in URDF
    with pytest.raises(KinematicsError):
        ScaraKinematics(URDF.replace(old, new))
