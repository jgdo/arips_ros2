from pathlib import Path

from arips_keypoint_detection.presets import list_presets, load_preset

SOURCE_PRESETS = Path(__file__).resolve().parents[1] / 'config' / 'presets'


def test_door_handle_preset():
    p = load_preset(SOURCE_PRESETS / 'door_handle.yaml')
    assert p.tag == 'door_handle'
    assert p.keypoints == ['handle_start', 'handle_end']


def test_invalid_presets_are_skipped(tmp_path):
    (tmp_path / 'good.yaml').write_text('tag: a_b\nkeypoints: [x, y, z]\n')
    (tmp_path / 'notag.yaml').write_text('keypoints: [p]\n')          # tag defaults to file name
    (tmp_path / 'empty.yaml').write_text('tag: e\n')                  # no keypoints
    (tmp_path / 'dup.yaml').write_text('keypoints: [p, p]\n')         # duplicate names
    (tmp_path / 'badtag.yaml').write_text('tag: "a b"\nkeypoints: [p]\n')
    presets = list_presets(tmp_path)
    assert [(p.name, p.tag, p.keypoints) for p in presets] == [('good', 'a_b', ['x', 'y', 'z']),
                                                                 ('notag', 'notag', ['p'])]
