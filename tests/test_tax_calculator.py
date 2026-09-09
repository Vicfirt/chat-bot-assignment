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


def test_single_85k_2025():
    r = estimate_tax(filing_status="single", gross_income=85_000, tax_year=2025)
    assert r["standard_deduction"] == 15_750
    assert r["taxable_income"] == 69_250
    # 2025 single: 10% to 11,925; 12% to 48,475; 22% above.
    # 1192.5 + 4386 + (69250-48475)*0.22 = 1192.5 + 4386 + 4570.5 = 10149
    assert r["total_tax"] == pytest.approx(10_149, abs=1.0)
    assert r["marginal_rate"] == pytest.approx(0.22)


def test_2025_is_the_default_year():
    assert estimate_tax(filing_status="single", gross_income=85_000)["tax_year"] == 2025


def test_no_dependents_means_no_child_tax_credit():
    r = estimate_tax(filing_status="single", gross_income=85_000, tax_year=2025)
    assert r["child_tax_credit"] == 0
    assert r["tax_after_credits"] == r["total_tax"]


def test_child_tax_credit_reduces_tax_after_credits():
    r = estimate_tax(filing_status="married_joint", gross_income=150_000,
                     tax_year=2025, dependents=2)
    # 2025 CTC is $2,200/child; MAGI $150k is well under the $400k MFJ phase-out.
    assert r["child_tax_credit"] == 4_400
    assert r["tax_after_credits"] == pytest.approx(r["total_tax"] - 4_400, abs=1.0)
    assert r["effective_rate"] == pytest.approx(r["tax_after_credits"] / 150_000, abs=1e-4)


def test_child_tax_credit_phases_out_above_threshold():
    r = estimate_tax(filing_status="single", gross_income=240_500,
                     tax_year=2025, dependents=1)
    # $40,500 over $200k -> 41 steps of $1,000 -> 41 * $50 = $2,050 reduction,
    # which wipes out the $2,200 credit down to $150.
    assert r["child_tax_credit"] == pytest.approx(150, abs=1.0)


def test_child_tax_credit_is_capped_at_tax_before_credits():
    r = estimate_tax(filing_status="head_of_household", gross_income=30_000,
                     tax_year=2025, dependents=3)
    assert r["child_tax_credit"] == r["total_tax"]
    assert r["tax_after_credits"] == 0


def test_income_below_standard_deduction_is_zero_tax():
    r = estimate_tax(filing_status="single", gross_income=10_000, tax_year=2025)
    assert r["taxable_income"] == 0
    assert r["total_tax"] == 0
    assert r["effective_rate"] == 0


def test_all_filing_statuses_supported():
    for year in (2024, 2025):
        for fs in FILING_STATUSES:
            r = estimate_tax(filing_status=fs, gross_income=120_000, tax_year=year)
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
