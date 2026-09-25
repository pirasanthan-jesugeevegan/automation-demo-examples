from ai_eval_lab.evaluation.normalization import normalize_claim


def test_case_whitespace_and_trailing_full_stop_do_not_matter() -> None:
    assert (
        normalize_claim("  Acme  Holdings\treceived a penalty. ")
        == "acme holdings received a penalty"
    )


def test_different_wording_is_still_different() -> None:
    assert normalize_claim("Acme was fined") != normalize_claim("Acme received a fine")
