from balatro_sim.economy import calculate_interest


def test_interest_scales_by_five_dollar_chunks():
    assert calculate_interest(0) == 0
    assert calculate_interest(4) == 0
    assert calculate_interest(5) == 1
    assert calculate_interest(17) == 3


def test_interest_caps_at_five():
    assert calculate_interest(25) == 5
    assert calculate_interest(100) == 5
