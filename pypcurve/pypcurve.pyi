from typing import Union, Iterable, Dict, List, Any, Set, Tuple, Optional, NoReturn, Pattern
from matplotlib.pyplot import Axes
from pandas.core.frame import DataFrame
from numpy import array


def adjust_spines(ax, spines):
    ax: Axes
    spines: List[str]


class PCurve(object):
    REGEX_STAT_TEST: Pattern

    POWER_LEVELS = List[float]

    _cached_chi_ncp_ests = Dict[str, float]

    _sig_chi_tests = List[List[float]]
    _sig_f_tests = List[List[float]]
    df_tests = DataFrame
    n_tests = Dict[str, float]
    stouffer_p = Dict[str, float]
    stouffer_z = Dict[str, float]

    # Static Methods
    @staticmethod
    def _bound_pvals(pvals: array) -> array:
        ...

    @staticmethod
    def _format_pval(p: float) -> Union[str, float]:
        ...

    @staticmethod
    def _compute_prop_lower_33(pcrit: float, family: array, df1: array, df2: array, p: array,
                               ncp33: array) -> array: ...

    @staticmethod
    def _compute_stouffer_z(pp: array) -> float: ...

    def _solve_ncp_f(self, df1: float, df2: float, power: float) -> float: ...

    def _solve_ncp_chi(self, df: float, power: float, use_cache: bool) -> float: ...

    def _solve_ncp(self, family: str, df1: float, df2: float, power: float) -> float: ...

    def _compute_ncp_f(self, df1: array, df2: array, power:float) -> array: ...

    def _compute_ncp_chi(self, df1: array, power:float, use_cache:bool) -> array: ...

    def _compute_ncp_all(self, power: float) -> array: ...

    def _parse_test(self, test_str: str) -> Tuple[str, str, float, float, float]: ...

    def _compute_pvals(self) -> array: ...

    def _compute_stouffer_z_at_power(self, power: float) -> float: ...

    def _solve_power_for_pct(self, pct) -> float: ...

    def _compute_ppvals_null(self, pcurvetype: str) -> array: ...

    def _compute_ppvals_33(self, pcurvetype: str) -> array: ...

    def _get_33_power_curve(self) -> array: ...

    def _run_binom_test(self, alternative: str) -> float: ...

    @property
    def has_evidential_value(self) -> bool: ...
    @property
    def has_inadequate_evidence(self) -> bool: ...
    def estimate_power(self) -> Tuple[float, float, float]: ...
