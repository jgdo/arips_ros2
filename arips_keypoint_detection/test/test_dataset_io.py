import json

import numpy as np
import pytest

from arips_keypoint_detection import dataset_io as dio


def test_frame_name_round_trip_and_sorting():
    a = dio.frame_name('episode_0000', 21_850_000_000, 'rgb', 'jpg')
    b = dio.frame_name('episode_0000', 100_000_000_001, 'rgb', 'jpg')
    assert a == 'episode_0000_0000000021_850000000_rgb.jpg'
    assert dio.parse_frame_name(a) == ('episode_0000', 21_850_000_000, 'rgb')
    assert dio.parse_frame_name(b) == ('episode_0000', 100_000_000_001, 'rgb')
    assert sorted([b, a]) == [a, b]
    assert dio.parse_frame_name('frame_123.jpg') is None
    assert dio.sanitize_name('my bag/1') == 'my_bag_1'


def test_depth_round_trip(tmp_path):
    depth = np.array([[0.0, 1.2345], [np.nan, 70.0]], dtype=np.float32)
    dio.save_depth_m(tmp_path / 'd.png', depth)
    loaded = dio.load_depth_m(tmp_path / 'd.png')
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, [[0.0, 1.234], [0.0, 0.0]], atol=1e-3)


def test_depth_association(tmp_path):
    for name in [dio.frame_name('ep0', 1_000_000_000, 'depth', 'png'),
                 dio.frame_name('ep0', 1_100_000_000, 'depth', 'png'),
                 dio.frame_name('ep1', 1_010_000_000, 'depth', 'png')]:
        dio.save_depth_m(tmp_path / name, np.ones((2, 2)))
    rgb = tmp_path / dio.frame_name('ep0', 1_020_000_000, 'rgb', 'jpg')
    rgb.write_bytes(b'')
    assert dio.find_depth_for_rgb(rgb).name == dio.frame_name('ep0', 1_000_000_000, 'depth', 'png')
    far = tmp_path / dio.frame_name('ep0', 2_000_000_000, 'rgb', 'jpg')
    assert dio.find_depth_for_rgb(far) is None
    assert dio.list_rgb_images(tmp_path) == [rgb]


def test_annotation_and_config(tmp_path):
    rgb = tmp_path / 'frame_123.jpg'
    assert dio.load_annotation(rgb, 'door_handle') is None
    path = dio.save_annotation(rgb, 'door_handle', {1: (3.04, 4.0), 0: (1, 2)})
    assert path.name == 'frame_123.jpg_door_handle.json'
    assert json.loads(path.read_text()) == {'0': [1.0, 2.0], '1': [3.0, 4.0]}
    assert dio.load_annotation(rgb, 'door_handle') == {0: (1.0, 2.0), 1: (3.0, 4.0)}
    dio.save_annotation(rgb, 'empty', {})
    assert dio.load_annotation(rgb, 'empty') == {}

    cfg = dio.save_config(tmp_path, 'door_handle', ['handle_start', 'handle_end'])
    assert cfg.name == 'config_door_handle.json'
    assert json.loads(cfg.read_text()) == {
        'tag': 'door_handle', 'keypoints': {'0': 'handle_start', '1': 'handle_end'}}
    assert dio.list_configs(tmp_path) == [('door_handle', ['handle_start', 'handle_end'])]
    with pytest.raises(ValueError):
        dio.save_config(tmp_path, 'bad tag', ['a'])


def test_split_files_and_create_split(tmp_path):
    from arips_keypoint_detection import create_split
    names = []
    for i in range(10):
        rgb = tmp_path / dio.frame_name('ep0', (i + 1) * 1_000_000_000, 'rgb', 'jpg')
        rgb.write_bytes(b'')
        names.append(rgb.name)
        if i < 9:   # one image not annotated -> not in any split
            dio.save_annotation(rgb, 'door_handle', {0: (1, 2)})
    cfg = dio.save_config(tmp_path, 'door_handle', ['a', 'b'])

    assert dio.split_path(tmp_path, 'door_handle', 'foobar', 'train').name == 'split_door_handle_foobar_train.json'
    assert dio.split_path(tmp_path, 'door_handle', None, 'test').name == 'split_door_handle_test.json'

    assert create_split.main([str(cfg), 'foobar', '--test_ratio', '0.3', '--seed', '1']) == 0
    train = dio.load_split(tmp_path / 'split_door_handle_foobar_train.json')
    test = dio.load_split(tmp_path / 'split_door_handle_foobar_test.json')
    assert len(test) == 3 and len(train) == 6
    assert not set(train) & set(test)
    assert names[9] not in {p.name for p in train + test}
    assert json.loads((tmp_path / 'split_door_handle_foobar_test.json').read_text()).keys() == {'images'}
    # refuses to overwrite
    assert create_split.main([str(cfg), 'foobar', '--test_ratio', '0.3']) == 1

    assert create_split.main([str(cfg), '--all']) == 0
    assert len(dio.load_split(tmp_path / 'split_door_handle_all.json')) == 9
    assert len(dio.list_splits(tmp_path)) == 3


def test_create_split_by_bag(tmp_path):
    from arips_keypoint_detection import create_split
    for b, n in (('ep0', 10), ('ep1', 10), ('ep2', 5), ('ep3', 15)):
        for i in range(n):
            rgb = tmp_path / dio.frame_name(b, (i + 1) * 1_000_000_000, 'rgb', 'jpg')
            rgb.write_bytes(b'')
            dio.save_annotation(rgb, 't', {})
    cfg = dio.save_config(tmp_path, 't', ['a'])
    for seed in range(5):
        assert create_split.main([str(cfg), f's{seed}', '--test_ratio', '0.25', '--by-bag', '--seed', str(seed)]) == 0
        train = {dio.parse_frame_name(p)[0] for p in dio.load_split(tmp_path / f'split_t_s{seed}_train.json')}
        test_paths = dio.load_split(tmp_path / f'split_t_s{seed}_test.json')
        test = {dio.parse_frame_name(p)[0] for p in test_paths}
        assert train and test and not train & test
        assert len(test_paths) in (10, 15)   # closest to 10 = 25 % of 40 images
