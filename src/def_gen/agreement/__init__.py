"""Inter-rater agreement metrics for definition judgements."""

from enum import StrEnum


class RaterMetric(StrEnum):
    """Enumeration of the implemented rater agreement metrics."""

    COHEN_KAPPA = "cohen_kappa"
    FLEISS_KAPPA = "fleiss_kappa"
    KRIPPENDORFF_ALPHA = "krippendorff_alpha"
    PERCENT_AGREEMENT = "percent_agreement"
    GWET_AC1 = "gwet_ac1"
