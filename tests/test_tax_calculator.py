import pytest

from app.tools.tax_calculator import FILING_STATUSES, estimate_tax


def test_single_85k_2024():
    r = estimate_tax(filing_status="single", gross_income=85_000, tax_year=2024)
    assert r["standard_deduction"] == 14_600
    assert r["taxable_income"] == 70_400
    # 2024 single: 10% to 11,600; 12% to 47,150; 22% above.
    # 1160 + 4266 + (70400-47150)*0.22 = 1160 + 4266 + 5115 = 10541
    assert r["total_tax"] == pytest.approx(10_541, abs=1.0)
    assert r["marginal_rate"] == pytest.approx(0.22)
    assert 0 < r["effective_rate"] < 0.22


def test_income_below_standard_deduction_is_zero_tax():
    r = estimate_tax(filing_status="single", gross_income=10_000, tax_year=2024)
    assert r["taxable_income"] == 0
    assert r["total_tax"] == 0
    assert r["effective_rate"] == 0


def test_all_filing_statuses_supported():
    for fs in FILING_STATUSES:
        r = estimate_tax(filing_status=fs, gross_income=120_000, tax_year=2024)
        assert r["total_tax"] > 0


def test_unknown_status_raises():
    with pytest.raises(ValueError):
        estimate_tax(filing_status="alien", gross_income=50_000)


def test_unsupported_year_raises():
    with pytest.raises(ValueError):
        estimate_tax(filing_status="single", gross_income=50_000, tax_year=1990)


def test_negative_income_raises():
    with pytest.raises(ValueError):
        estimate_tax(filing_status="single", gross_income=-1)
