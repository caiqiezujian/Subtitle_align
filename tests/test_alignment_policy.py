from app.alignment_policy import has_spoken_content, interpolate_missing_ranges


def test_punctuation_only_line_uses_full_neighbor_gap():
    ranges, flags = interpolate_missing_ranges(
        ["第十三句", "。", "第十五句"],
        [(10.0, 12.5), (12.2, 12.3), (15.75, 17.0)],
        total_duration=20.0,
    )
    assert ranges[1] == (12.5, 15.75)
    assert flags == [False, True, False]


def test_multiple_unresolved_lines_share_gap_without_overlap():
    ranges, flags = interpolate_missing_ranges(
        ["before", "无法识别一", "…", "after"],
        [(1.0, 2.0), None, None, (6.0, 7.0)],
        total_duration=8.0,
    )
    assert ranges[1] == (2.0, 4.0)
    assert ranges[2] == (4.0, 6.0)
    assert flags == [False, True, True, False]


def test_spoken_content_detection_handles_multiple_languages():
    assert has_spoken_content("Hello!")
    assert has_spoken_content("你好。")
    assert has_spoken_content("123")
    assert not has_spoken_content("……？！")
    assert not has_spoken_content("♪")
