import numpy as np
import pandas as pd
import re
import warnings

import scipy.optimize as opt
from scipy.stats import norm, f, chi2, ncf, ncx2, binom
from scipy.special import ncfdtrinc, chndtrinc

import matplotlib.pyplot as plt
import seaborn as sns

from poibin import PoiBin

warnings.filterwarnings("ignore")


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
    return None


# noinspection PyProtectedMember,PyProtectedMember,PyProtectedMember,PyProtectedMember,PyProtectedMember,PyProtectedMember,PyProtectedMember,PyProtectedMember
class PCurve(object):
    __version__ = "0.1.1"

    __pcurve_app_version__ = "4.06"

    _REGEX_STAT_TEST = re.compile(
        """
        ^ # Beginning of string
        (?P<testtype>chi2|F|t|r|z|p)  # Matches the type of test statistic
        (\((?P<df1>\d+)(,)?(?P<df2>\d+)?\))? #Matches the degrees of freedom
        =(-?) #Matches an equal sign with a potential minus sign
        (?P<stat>(\d*)\.(\d+)) # Matches the test statistics
        """,
        re.IGNORECASE | re.VERBOSE)

    _POWER_LEVELS = [0.051, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
                     0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36,
                     0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53,
                     0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,
                     0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86,
                     0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

    @staticmethod
    def _bound_pvals(pvals):
        """
        Bound a p-value to avoid precision error
        :param pvals: The p-values
        :return: The bounded p-values 2.2e-16 < p < 1-2.2e-16
        """
        return np.where(pvals < 2.2e-16, 2.2e-16, np.where(pvals > 1 - 2.2e-16, 1 - 2.2e-16, pvals))

    @staticmethod
    def _format_pval(p):
        if p < .0001:
            return "< .0001"
        elif p < .9999:
            pstring = f"= {p:.4f}".strip("0")
            return pstring
        else:
            return "> .9999"

    @staticmethod
    def _compute_prop_lower_33(pcrit, family, df1, df2, p, ncp33):
        """
        Compute the proportion of p-values that is expected to be smaller than `pcrit` under 33% power.
        :param pcrit: The p-values
        :return: The proportion
        """
        # Transform the critical p-value `pcrit` into the corresponding critical value for the dist.
        critval_f = f.ppf(1 - pcrit, df1, df2)
        critval_chi = chi2.ppf(1 - pcrit, df1)

        # Compute the proportion of values that are expected to be larger than this under 33% power
        exp_smaller_f = ncf._sf(critval_f, df1, df2, ncp33)
        exp_smaller_chi = ncx2._sf(critval_chi, df1, ncp33)

        # Return the appropriate stats for the family
        return np.where(p > .05, np.nan, np.where(family == "F", exp_smaller_f, exp_smaller_chi))

    @staticmethod
    def _compute_stouffer_z(pp):
        isnotnan = ~np.isnan(pp)
        pp_notnan = pp[isnotnan]
        return np.sum(norm._ppf(pp_notnan)) / np.sqrt(isnotnan.sum())

    @staticmethod
    def _compute_ncp_f(df1, df2, power=1 / 3):
        """
        Uses the inverse function of the non-central F distribution with regard to NCP to recover the NCP corresponding
        to a given level of power for a given F test.
        :param df1: Numerator degrees of freedom
        :param df2: Denominator degrees of freedom
        :param power: Desired level of power
        :return:
        """
        critval = f._ppf(.95, df1, df2)
        return ncfdtrinc(df1, df2, 1 - power, critval)

    @staticmethod
    def _compute_ncp_chi(df, power=1 / 3):
        """
        Uses the inverse function of the non-central Chi2 distribution with regard to NCP to recover the NCP
        corresponding to a given level of power for a given Chi2 test.
        :param df: Degrees of freedom
        :param power: Desired level of power
        :return:
        """
        critval = chi2._ppf(.95, df)
        return chndtrinc(critval, df, 1 - power)

    def _compute_ncp_all(self, power=1 / 3):
        family = self._df_results["family"].values
        df1, df2 = self._df_results[["df1", "df2"]].to_numpy().T
        return np.where(family == "F", self._compute_ncp_f(df1, df2, power),
                        self._compute_ncp_chi(df1, power))

    def _parse_result(self, result_str):
        """
        Parse a statistical result, entered as a string
        :param result_str: A statistical result (e.g., F(1, 234) = 12.32)
        :return:
        """
        result_str_replaced = (
            result_str
                .replace(" ", "")  # Whitespaces
                .replace("\u2009", "")  # Thin whitespaces
                .replace("\u2212", "-")  # All possible symbols for 'minus'
                .replace("\u2013", "-")
                .replace("\uFE63", "-")
                .replace("\u002D", "-")
                .replace("\uFF0D", "-")
        )
        match = self._REGEX_STAT_TEST.match(result_str_replaced)  # See regex above
        if match is None:  # Statistical test not recognized
            raise ValueError(f"The input {result_str} is not recognized. Please correct it")

        test_props = match.groupdict()  # Test recognized, accessing properties
        test_type = test_props["testtype"]
        df1_raw = test_props["df1"]
        df2_raw = test_props["df2"]
        stat_raw = test_props["stat"]

        # Testing that degrees of freedom are correctly entered
        if (test_type == "F") and ((test_props["df1"] is None) or (test_props["df2"] is None)):
            raise ValueError(
                f"Error in {result_str}: The test statistics {test_type} requires you to specify the numerator \
                                and denominator degrees of freedom.")
        if (test_type not in ["z", "p"]) and (test_props["df1"] is None):
            raise ValueError(
                f"Error in {result_str}: The test statistics {test_type} requires you to specify the degrees of \
                        freedom.")

        stat_raw = float(stat_raw)
        if test_type == "F":
            family = "F"
            df1 = float(df1_raw)
            df2 = float(df2_raw)
            stat = stat_raw
        elif test_type == "t":
            family = "F"
            df1 = 1
            df2 = float(df1_raw)
            stat = stat_raw ** 2
        elif test_type == "r":
            family = "F"
            df1 = 1
            df2 = float(df1_raw)
            stat = (stat_raw / (np.sqrt((1 - stat_raw ** 2) / df2))) ** 2
        elif test_type == "chi2":
            family = "Chi2"
            df1 = float(df1_raw)
            df2 = None
            stat = stat_raw
        elif test_type == "z":
            family = "Chi2"
            df1 = 1
            df2 = None
            stat = stat_raw ** 2
        else:
            family = "Chi2"
            df1 = 1
            df2 = None
            stat = norm.ppf(1 - stat_raw / 2) ** 2

        return result_str_replaced, family, df1, df2, stat

    def _compute_pvals(self):
        family = self._df_results["family"].values
        df1, df2, stat = self._df_results[["df1", "df2", "stat"]].to_numpy().T
        pval = np.where(family == "F", f._sf(stat, df1, df2), chi2._sf(stat, df1))
        return self._bound_pvals(pval)

    def _compute_stouffer_z_at_power(self, power):
        # NCP and pp-values of F tests
        df1_f, df2_f, stat_f = self._sig_f_tests
        ncp_f_est = self._compute_ncp_f(df1_f, df2_f, power)
        pp_f_est = (ncf._cdf(stat_f, df1_f, df2_f, ncp_f_est) - (1 - power)) / power

        # NCP and pp-values of Chi2 tests
        df1_chi, stat_chi = self._sig_chi_tests
        ncp_chi_est = self._compute_ncp_chi(df1_chi, power)
        pp_chi_est = (ncx2._cdf(stat_chi, df1_chi, ncp_chi_est) - (1 - power)) / power

        # pp-values for all tests
        pp_est = self._bound_pvals(np.hstack([pp_f_est, pp_chi_est]))
        stouffer_at_power = self._compute_stouffer_z(pp_est)
        return stouffer_at_power

    def _solve_power_for_pct(self, pct):
        z = norm._ppf(pct)
        error = lambda est: self._compute_stouffer_z_at_power(est) - z
        return opt.brentq(error, .05, .999999)

    def _compute_ppvals_null(self, pcurvetype="full"):
        """
        Compute the pp-value of the p-values under the null. It simply stretches the p-value over the interval [0, 1]
        :param pcurvetype: The type of p-curve (full or half)
        :return:
        """
        p = self._df_results["p"].values
        if pcurvetype == "full":
            return np.array([self._bound_pvals(x * 20) if x < .05 else np.nan for x in p])
        else:
            return np.array([self._bound_pvals(x * 40) if x < .025 else np.nan for x in p])

    def _compute_ppvals_33(self, pcurvetype="full"):
        family = self._df_results["family"].values
        df1, df2, stat, pvals, ncp33 = self._df_results[["df1", "df2", "stat", "p", "ncp33"]].to_numpy().T
        if pcurvetype == "full":
            pthresh = .05  # Only keep p-values smaller than .05
            propsig = 1 / 3  # Under 33% power, 1/3 of p-values should be lower than .05
        else:
            pthresh = .025  # Only keep p-values smaller than .025
            # We don't know which prop of p-values should be smaller than .025 under 33% power, so compute it
            propsig = 3 * self._compute_prop_lower_33(.025, family, df1, df2, pvals, ncp33)

        # We then stretch the ppval on the [0, 1] interval.
        pp_33_f = (1 / propsig) * (ncf.cdf(stat, df1, df2, ncp33) - (1 - propsig))
        pp_33_chi = (1 / propsig) * (ncx2.cdf(stat, df1, ncp33) - (1 - propsig))
        pp_33 = np.where(family == "F", pp_33_f, pp_33_chi)
        return np.array([self._bound_pvals(pp) if p < pthresh else np.nan for (p, pp) in zip(pvals, pp_33)])

    def _get_33_power_curve(self):
        family = self._df_results["family"].values
        df1, df2, p, ncp33 = self._df_results[["df1", "df2", "p", "ncp33"]].to_numpy().T
        cprop = lambda x: self._compute_prop_lower_33(x, family, df1, df2, p, ncp33)
        propsig = np.array([cprop(c) for c in [.01, .02, .03, .04, .05]])
        diffs = np.diff(propsig, axis=0,
                        prepend=0)  # Difference of CDFs: Likelihood of p-values falling between each value
        props = np.nanmean(3 * diffs, axis=1)
        return props

    def _run_binom_test(self, alternative="null"):
        family = self._df_results["family"].values
        df1, df2, p, ncp33 = self._df_results[["df1", "df2", "p", "ncp33"]].to_numpy().T
        k_below_25 = self._n_tests['p025']
        if alternative == "null":
            return binom(n=self._n_tests['p05'], p=.5).sf(k_below_25 - 1)
        else:
            prop_below_25_33 = 3 * self._compute_prop_lower_33(.025, family, df1, df2, p, ncp33)
            prop_below_25_33_filtered = prop_below_25_33[p < .05]
            return PoiBin(prop_below_25_33_filtered).cdf(k_below_25)

    def __init__(self, test_arr):
        """
        Initialize a p-curve object. The only argument is a list (or an array) of statistical tests in text format.
        As with the p-curve app, pypcurve accepts the following formats of statistical tests:
            F(dfn, dfd)=XXX
            t(df)=XXX
            r(N)=XXX
            z=XXX
            chi2(df)=XXX
        If this is all the information you have, you can also specify the p-value of a test directly: p=X
        :param test_arr: A list (of array of statistical tests).
        """
        self._df_results = (pd.DataFrame(np.array([self._parse_result(t) for t in test_arr]),
                                         columns=["result_str", "family", "df1", "df2", "stat"])
                          .astype({'result_str': str,
                                   'family': str,
                                   "df1": np.float64,
                                   "df2": np.float64,
                                   "stat": np.float64
                                   })
                          )

        self._df_results["p"] = self._compute_pvals()
        n_tests = self._df_results.shape[0]
        n_tests_p05 = self._df_results[self._df_results.p <= .05].shape[0]
        n_tests_p025 = self._df_results[self._df_results.p <= .025].shape[0]
        n_tests_ns = n_tests - n_tests_p05
        self._n_tests = {"all": n_tests, "p05": n_tests_p05, "p025": n_tests_p025, "ns": n_tests_ns}

        # Unpacking the significant F tests
        sig_f_tests = self._df_results.query("p < .05 & family == 'F'")
        df1_f, df2_f, stat_f = sig_f_tests[["df1", "df2", "stat"]].to_numpy().T

        # Then significant Chi2 tests
        sig_chi_tests = self._df_results.query("p < .05 & family == 'Chi2'")
        df1_chi, stat_chi = sig_chi_tests[["df1", "stat"]].to_numpy().T

        self._sig_f_tests = [df1_f, df2_f, stat_f]
        self._sig_chi_tests = [df1_chi, stat_chi]

        self._df_results["ncp33"] = self._compute_ncp_all(1 / 3)
        self._df_results["pp-null-full"] = self._compute_ppvals_null("full")
        self._df_results["pp-null-half"] = self._compute_ppvals_null("half")
        self._df_results["pp-33-full"] = self._compute_ppvals_33("full")
        self._df_results["pp-33-half"] = self._compute_ppvals_33("half")

        self._stouffer_z = dict()
        self._stouffer_p = dict()
        # For all p-curve types and alternatives:
        for s in ["null-full", "null-half", "33-full", "33-half"]:
            # Convert pp-values to z-scores
            pp = self._df_results[f"pp-{s}"]
            self._df_results[f"z-{s}"] = norm.ppf(pp)
            # Compute the stouffer Z
            z = self._compute_stouffer_z(pp)
            self._stouffer_z[s] = z
            self._stouffer_p[s] = norm.cdf(z)

        levels = self._POWER_LEVELS  # Levels of power on which to run the estimation
        stouffer_at_power_levels = np.array([self._compute_stouffer_z_at_power(p) for p in levels])
        self._z_at_power = np.abs(stouffer_at_power_levels)

    @property
    def has_evidential_value(self):
        """
        Combines the half and full-pcurve to make inferences about evidential value (test of flatness against null).
        If the half p-curve test is right-skewed with p<.05 or both the half and full test are right-skewed with p<.1,
        then p-curve analysis indicates the presence of evidential value.
        :return: True if it contains evidential value, False otherwise
        """
        ps = self._stouffer_p
        p_half, p_full = ps["null-half"], ps["null-full"]
        return any([p_half < .05, p_full < .05]) or all([p_half < .1, p_full < .1])

    @property
    def has_inadequate_evidence(self):
        """
        Combines the half and full-pcurve to make inferences about the evidential value presented (test of flatness
        against 33% power).
        If the half p-curve test is right-skewed with p<.05 or both the half and full test are right-skewed with p<.1,
        then p-curve analysis indicates the presence of evidential value.
        :return: True if it contains evidential value, False otherwise
        """
        ps = self._stouffer_p
        p_half, p_full = ps["33-half"], ps["33-full"]
        return any([p_half < .05, p_full < .05]) or all([p_half < .1, p_full < .1])

    def get_binomial_tests(self):
        """
        Return the p-values of the binomial tests against the p-curve of null power and the p-curve of 33% power.
        :return: A dictionary of p-values
        """
        return {"null": self._run_binom_test("null"), "33%": self._run_binom_test("33")}

    def get_stouffer_tests(self):
        """
        Return the Z and p-values of the Stouffer (continuous) tests against the p-curve of null power and the p-curve
        of 33% power, for the full and half p-curve
        :return: A dictionary of statistical results: {"p": p, "z": z}
        """
        keys = ["null-full", "null-half", "33%-full", "33%-half"]
        stouffer_p = [self._stouffer_p[t] for t in ["null-full", "null-half", "33-full", "33-half"]]
        stouffer_z = [self._stouffer_z[t] for t in ["null-full", "null-half", "33-full", "33-half"]]
        return {k: {"p": p, "z": z} for (k, p, z) in zip(keys, stouffer_p, stouffer_z)}

    def get_results_entered(self):
        """
        Return a DataFrame of the statistical tests entered by the user, and their associated properties.
        :return:
        """
        headers = pd.MultiIndex.from_tuples((("p-value", "", ""),
                                             ("pp-values", "Full", "Null"), ("pp-values", "Full", "33%"),
                                             ("pp-values", "Half", "Null"), ("pp-values", "Half", "33%"),
                                             ("Z scores", "Full", "Null"), ("Z scores", "Full", "33%"),
                                             ("Z scores", "Half", "Null"), ("Z scores", "Half", "33%")))
        index = pd.Index(self._df_results.result_str, name="Test entered by user")
        values = self._df_results[['p', 'pp-null-full', 'pp-33-full', 'pp-null-half', 'pp-33-half',
                                 'z-null-full', 'z-33-full', 'z-null-half', 'z-33-half']].values
        return pd.DataFrame(values, index=index, columns=headers)

    def estimate_power(self):
        """
        Estimate the power of the tests entered in the-pcurve, correcting for publication bias.
        :return: p, lbp, ubp: The power, and (lower bound, upper bound) of the 95% CI for power.
        """
        try:
            p = self._solve_power_for_pct(.50)
        except:
            # If the power of the tests is exactly 5%, the power estimation will fail. In this case, we will use the
            # default power of 5%.
            p = .05
        p05 = norm._cdf(self._compute_stouffer_z_at_power(.051))
        p99 = norm._cdf(self._compute_stouffer_z_at_power(.99))
        if p05 <= .95:
            lbp = .05
        elif p99 >= .95:
            lbp = .99
        else:
            lbp = self._solve_power_for_pct(.95)

        if p05 <= .05:
            ubp = .05
        elif p99 >= .05:
            ubp = .99
        else:
            ubp = self._solve_power_for_pct(.05)
        return p, (lbp, ubp)

    def pcurve_analysis_summary(self):
        """
        Print the summary of the p-curve analysis: The binomial tests, continuous tests of full p-curve, and continuous
        tests of half p-curve against the flat p-curve/the p-curve of 33% power.
        :return: A DataFrame containing the results.
        """
        header = pd.MultiIndex.from_tuples(
            (('Binomial Test', ''), ('Continuous Test', 'Full'), ('Continuous Test', 'Half')))
        index = ["Test of Right-Skewness", "Test of Flatness vs. 33% Power"]
        stouffer_p = [self._stouffer_p[t] for t in ["null-full", "null-half", "33-full", "33-half"]]
        stouffer_z = [self._stouffer_z[t] for t in ["null-full", "null-half", "33-full", "33-half"]]
        stouffer_results = np.reshape([f"Z={z:.2f}, p {self._format_pval(p)}" for z, p in zip(stouffer_z, stouffer_p)],
                                      (2, 2))
        binom_results = np.reshape([f"p {self._format_pval(self._run_binom_test('null'))}",
                                    f"p {self._format_pval(self._run_binom_test('33'))}"], (2, 1))
        results = np.hstack([binom_results, stouffer_results])
        return pd.DataFrame(results, index=index, columns=header)

    def summary(self):
        """
        Print a summary, like the one you'd see using the web version of the P-Curve app.
        :return:
        """

        init_str = (
            f"pypcurve v. {self.__version__} is based on Uri Simonsohn's "
            f"P-Curve's app v. {self.__pcurve_app_version__}.\n"
        )
        print(init_str)
        self.plot_pcurve(dpi=100)
        plt.show()
        summary_str = ("------------- Summary of p-curve tests -------------\n\n"
                       + self.pcurve_analysis_summary().to_string())
        print(summary_str)
        return None

    def plot_power_estimate(self, dpi=100):
        """
        Plot the power estimate, as it appears on the online web-app.
        :return: A matplotlib ax containing the power estimate.
        """
        fig, ax = plt.subplots(1, dpi=dpi)
        x = self._POWER_LEVELS
        y = self._z_at_power
        ax.scatter(x, y)
        ax.set_xlabel("Power")
        ax.set_ylabel("Fit of Observed P-Curve")
        ax.set_xticks(np.arange(.1, 1.1, .1))
        power_est, _ = self.estimate_power()
        stouffer_at_power_est = self._compute_stouffer_z_at_power(power_est)
        ax.scatter(power_est, stouffer_at_power_est, color="red")
        ax.set_title(f"Best Power Estimate = {power_est * 100:.0f}%"
                     "\n(do not trust if plot is not V-shaped or smooth line to 99%)")
        sns.despine(ax=ax)
        return ax

    def plot_pcurve(self, dpi=100):
        """
        Plot the p-curve, as it appears on the online web-app
        :return: A matplotlib ax containing the p-curve
        """
        sns.set_context("paper")

        # Creating the figure containing all axes
        fig = plt.figure(figsize=(6.38, 6.38), dpi=dpi)

        # Generating the p-curve first

        ##Subsetting significant p-values
        bins = np.arange(0, 0.06, .01)
        pvals = self._df_results.p.values
        pvals_sig = pvals[pvals < .05]

        ## Proportion of p-values in each bin
        count, _ = np.histogram(pvals_sig, bins)
        prop = count / len(pvals_sig)
        maxprop = max(prop)

        ## Adding main ax to plot p-curves
        gs = fig.add_gridspec(5, 1)
        main_ax = fig.add_subplot(gs[:-1])

        ## Cleaning the axes and ticks
        main_ax.set_ylim(-.01, 1.1)
        main_ax.set_yticks(np.linspace(0, 1, 5))
        main_ax.set_xticks(bins[1:])
        sns.despine(trim=True)
        adjust_spines(main_ax, ['left', 'bottom'])
        main_ax.set_xlabel("p-value")
        main_ax.set_ylabel("Percentage of significant tests")

        ## Plotting the observed p-curve
        main_ax.scatter(bins[1:], prop, color="#1c86ee")
        main_ax.plot(bins[1:], prop, color="#1c86ee")
        for x, p in zip(bins[1:], prop):
            main_ax.annotate(f"{p * 100:.0f}%", (x, p + .05), ha="center", va="center")

        ## Plotting the uniform null
        main_ax.axhline(.20, ls=":", color="#ee2c2c")

        ## Plot the null of 33% power
        prop33 = self._get_33_power_curve()
        main_ax.plot(bins[1:], prop33, color="#008b45", ls="--")

        # Adding the "pseudo-legend": A new ax overlayed on top of the main ax
        if maxprop < .65:
            leg_ax = main_ax.inset_axes([0, .65, .975, .3])
        else:
            leg_ax = main_ax.inset_axes([0, maxprop, .975, .3])

        ## Formatting the pseudo legend
        sns.despine(ax=leg_ax, left=True, bottom=True)
        leg_ax.set_xticks([])
        leg_ax.set_yticks([])
        leg_ax.set_xlim(-.01, 1.01)
        leg_ax.set_ylim(-.01, 1.01)

        ## Next we add the legend and their labels
        ### Observed p-curve
        leg_ax.plot([.05, .15], [.85, .85], color="#1c86ee")
        leg_ax.annotate("Observed p-curve", (.18, .85), va="center", fontsize=12)
        p, (lp, up) = self.estimate_power()
        leg_ax.annotate(f"Est. Power = {p * 100:.0f}%, CI({lp * 100:.0f}%, {up * 100:.0f}%)",
                        (.22, .73), va="center", fontsize=8,
                        color="grey")
        leg_ax.plot([.05, .15], [.55, .55], color="#ee2c2c", ls=":")

        ### Null of no effect
        leg_ax.annotate("Null of no effect", (.18, .55), va="center", fontsize=12)
        p_null_full = self._stouffer_p["null-full"]
        p_null_half = self._stouffer_p["null-half"]
        leg_ax.annotate(
            f"Tests for right-skewness: $p_{{\mathrm{{Full}}}}$ {self._format_pval(p_null_full)}, "
            f"$p_{{\mathrm{{Half}}}}$ {self._format_pval(p_null_half)}",
            (.22, .43), va="center", fontsize=8, color="grey")
        leg_ax.plot([.05, .15], [.25, .25], color="#008b45", ls="--")

        ### Null of 33% power
        leg_ax.annotate("Null of 33% power", (.18, .25), va="center", fontsize=12)
        p_33_full = self._stouffer_p["33-full"]
        p_33_half = self._stouffer_p["33-half"]
        p_binomial = self._run_binom_test("33")
        leg_ax.annotate(
            f"Tests for flatness: $p_{{\mathrm{{Full}}}}$ {self._format_pval(p_33_full)}, "
            f"$p_{{\mathrm{{Half}}}}$ {self._format_pval(p_33_half)}, "
            f"$p_{{\mathrm{{Binom}}}}$ {self._format_pval(p_binomial)}",
            (.22, .13), va="center", fontsize=8, color="grey")

        ## Bounding box
        leg_ax.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], lw=1, color="grey", alpha=.8)

        # Adding the "pseudo-footnote": A new ax overlayed at bottom of the main ax
        footnote_ax = fig.add_subplot(gs[-1])

        ## Formatting the ax
        footnote_ax.set_xticks([])
        footnote_ax.set_yticks([])
        footnote_ax.set_xlim(0, 1)
        footnote_ax.set_ylim(0, 1)
        sns.despine(ax=footnote_ax, left=True, bottom=True)

        ## Adding the footnote
        footnote = (
            f"Note: The observed p-curve includes {self._n_tests['p05']} statistically significant (p < .05) results, "
            f"of which {self._n_tests['p025']} are p < .025."
        )
        if self._n_tests['ns'] == 0:
            footnote += "\nThere were no non-significant results entered."
        elif self._n_tests['ns'] == 1:
            footnote += "\nThere was one result entered but excluded from p-curve because it was p > .05."
        else:
            footnote += (f"\nThere were {self._n_tests['ns']} additional results entered but excluded from p-curve "
                         f"because they were p > .05.")
        footnote_ax.annotate(footnote, (0, 1), va="center", fontsize=7)

        # We tighten the figure and return it
        plt.tight_layout()
        return main_ax
