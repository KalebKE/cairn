from cairn.hooks.pre_add_guard import _is_broad_add


def test_is_broad_add_blocks_real_commit_all_variants():
    assert _is_broad_add("git commit -a")
    assert _is_broad_add('git commit -am "msg"')
    assert _is_broad_add('git commit -va -m "msg"')
    assert _is_broad_add("git commit --all")


def test_is_broad_add_allows_commit_text_and_value_args_with_a_like_text():
    assert not _is_broad_add('git commit -m "fix: learned-the-hard-way"')
    assert not _is_broad_add('git commit -m "support -a parsing"')
    assert not _is_broad_add('git commit --amend -m "x"')
    assert not _is_broad_add("git commit -F path/to/some-area/msg.txt")
    assert not _is_broad_add("git commit --author 'A Name <a@example.com>' -m x")
    assert not _is_broad_add('git commit -m "support -a parsing')
