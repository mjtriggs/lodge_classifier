from lodge_classifier.normalise import normalise_lodge_name


def test_normalise_removes_lodge_token() -> None:
    n = normalise_lodge_name("St. Cuthbert's Lodge")
    assert "lodge" not in n.tokens


def test_normalise_apostrophes() -> None:
    n = normalise_lodge_name("Old Westminsters’ Lodge")
    assert "westminsters'" not in n.normalised  # apostrophe standardised but punctuation removed
    assert "old" in n.tokens
