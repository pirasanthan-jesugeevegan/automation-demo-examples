from ai_eval_lab.evaluation.gate import Gate, check_gate


def test_a_run_that_meets_the_gate_has_no_failures() -> None:
    assert check_gate(1.0, 0, Gate()) == []


def test_a_pass_rate_below_the_minimum_fails() -> None:
    (failure,) = check_gate(0.8, 0, Gate(min_pass_rate=0.9))

    assert "80%" in failure
    assert "90%" in failure


def test_the_minimum_itself_passes() -> None:
    assert check_gate(0.9, 0, Gate(min_pass_rate=0.9)) == []


def test_fabricated_citations_are_ignored_unless_a_limit_is_set() -> None:
    assert check_gate(1.0, 7, Gate()) == []


def test_fabricated_citations_over_the_limit_fail_even_with_a_perfect_pass_rate() -> None:
    (failure,) = check_gate(1.0, 1, Gate(max_invalid_citations=0))

    assert "1 fabricated citation" in failure


def test_every_broken_rule_is_reported() -> None:
    failures = check_gate(0.2, 3, Gate(min_pass_rate=1.0, max_invalid_citations=0))

    assert len(failures) == 2
