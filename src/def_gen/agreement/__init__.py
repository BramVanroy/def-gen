from enum import StrEnum


class RaterMetric(StrEnum):
    """Enumeration of rater agreement metrics."""

    COHEN_KAPPA = "cohen_kappa"
    KRIPPENDORFF_ALPHA = "krippendorff_alpha"
    PEARSON_CORRELATION = "pearson_correlation"
    SPEARMAN_CORRELATION = "spearman_correlation"
