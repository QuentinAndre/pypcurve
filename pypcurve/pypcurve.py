import numpy as np
import scipy.optimize as opt
from scipy.stats import norm, f, chi2, ncf, ncx2, binom
import pandas as pd
import re
from functools import partial
from poibin import PoiBin
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import ncfdtrinc, chndtrinc


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


class PCurve(object):
    REGEX_STAT_TEST = re.compile(
        """
        ^ # Beginning of string
        (?P<testtype>chi2|F|t|r|z|p)  # Matches the type of test statistic
        (\((?P<df1>\d+)(,)?(?P<df2>\d+)?\))? #Matches the degrees of freedom
        =(-?) #Matches an equal sign with a potential minus sign
        (?P<stat>(\d*)\.(\d+)) # Matches the test statistics
        """,
        re.IGNORECASE | re.VERBOSE)

    POWER_LEVELS = [0.051, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21,
                    0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38,
                    0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55,
                    0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72,
                    0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                    0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

    @staticmethod
    def _bound_pvals(pvals):
        """
        Bound a p-value to avoid precision error
        :param pvals: The p-values
        :return: The bounded p-values 2.2e-16 < p < 1-2.2e-16
        """
        return np.where(pvals < 2.2e-16, 2.2e-16, np.where(pvals > 1 - 2.2e-16, 1 - 2.2e-16, pvals))

    @staticmethod
    def _format_pvals(p):
        if p < .0001:
            return "< .0001"
        elif p < .9999:
            pstring = f"{p:.4f}".strip("0")
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
        exp_larger_f = ncf.cdf(critval_f, df1, df2, ncp33)
        exp_larger_chi = ncx2.cdf(critval_chi, df1, ncp33)

        # Return the appropriate stats for the family
        return np.where(p > .05, np.nan, np.where(family == "F", 1 - exp_larger_f, 1 - exp_larger_chi))

    @staticmethod
    def _compute_stouffer_z(pp):
        isnotnan = ~np.isnan(pp)
        pp_notna = pp[isnotnan]
        return np.sum(norm._ppf(pp_notna)) / np.sqrt(isnotnan.sum())

    def _solve_ncp_f(self, df1, df2, power):
        """
        Find the non-centrality parameter of the F distribution
        that corresponds to a given level of power and a given alpha
        :param df1: Numerator degrees of freedom
        :param df2: Denominator degrees of freedom
        :param power: Desired level of power
        :return: The non-centrality parameter
        """
        x = f._ppf(.95, df1, df2)  # Critical value of the distribution
        error = lambda est: ncf._sf(x, df1, df2, est) - power
        return opt.brentq(error, 0, 1000, maxiter=1000)

    def _solve_ncp_chi(self, df, power, use_cache=True):
        """
        Find the non-centrality parameter of the Chi2 distribution
        that corresponds to a given level of power for alpha = .05.
        Since it is computationally intensive, it can cache previous estimates
        and recover them if `use_cache` is True.
        :param df: Degrees of freedom
        :param power: Desired level of power
        :param use_cache: Whether to search the cache for a pre-computed value
        :return: The non-centrality parameter
        """
        if use_cache:
            cache_key = f"{power}-{df}"
            ncp_est = self._cached_chi_ncp_ests.get(cache_key)
            if ncp_est:
                return ncp_est
            x = chi2._ppf(.95, df)  # Critical value of the distribution
            # What is the NCP that is required to achieve a level of `power`?
            error = lambda est: ncx2._cdf(x, df, nc=est) - (1 - power)
            root = opt.brentq(error, 0, 1000, maxiter=1000)
            self._cached_chi_ncp_ests[cache_key] = root
            return root
        else:
            x = chi2._ppf(.95, df)  # Critical value of the distribution
            # What is the NCP that is required to achieve a level of `power`?
            error = lambda est: ncx2._cdf(x, df, nc=est) - (1 - power)
            root = opt.brentq(error, 0, 1000, maxiter=1000)
            return root

    def _solve_ncp(self, family, df1, df2, power=1 / 3):
        return self._solve_ncp_chi(df1, power, use_cache=False) if family == "Chi" else self._solve_ncp_f(df1, df2, power)

    def _compute_ncp_f(self, df1, df2, power=1 / 3):
        return np.array(list(map(partial(self._solve_ncp_f, power=power), df1, df2)))

    def _compute_ncp_chi(self, df1, power=1 / 3, use_cache=True):
        return np.array(list(map(partial(self._solve_ncp_chi, power=power, use_cache=use_cache), df1)))

    def _compute_ncp_all(self, power=1 / 3):
        family = self.df_tests["family"].values
        df1, df2 = self.df_tests[["df1", "df2"]].to_numpy().T
        return np.array(list(map(partial(self._solve_ncp, power=power), family, df1, df2)))

    def _compute_ncp_f_faster(self, df1, df2, power=1 / 3):
        critval = f._ppf(.95, df1, df2)
        return ncfdtrinc(df1, df2, 1 - power, critval)

    def _compute_ncp_chi_faster(self, df1, power=1 / 3):
        critval = chi2._ppf(.95, df1)
        return chndtrinc(critval, df1, 1 - power)

    def _compute_ncp_all_faster(self, power=1 / 3):
        family = self.df_tests["family"].values
        df1, df2 = self.df_tests[["df1", "df2"]].to_numpy().T
        return np.where(family == "F", self._compute_ncp_f_faster(df1, df2, power),
                        self._compute_ncp_chi_faster(df1, power))

    def _parse_test(self, test_str):
        """
        Parse a statistical test, entered as a string
        :param test_str: A statistical test (e.g., F(1, 234) = 12.32)
        :return:
        """
        test_str_replaced = (
            test_str
                .replace(" ", "")  # Whitespaces
                .replace("\u2009", "")  # Thin whitespaces
                .replace("\u2212", "-")  # All possible symbols for 'minus'
                .replace("\u2013", "-")
                .replace("\uFE63", "-")
                .replace("\u002D", "-")
                .replace("\uFF0D", "-")
        )
        match = self.REGEX_STAT_TEST.match(test_str_replaced)  # See regex above
        if match is None:  # Statistical test not recognized
            raise ValueError(f"The input {test_str} is not recognized. Please correct it")

        test_props = match.groupdict()  # Test recognized, accessing properties
        test_type = test_props["testtype"]
        df1_raw = test_props["df1"]
        df2_raw = test_props["df2"]
        stat_raw = test_props["stat"]

        # Testing that degrees of freedom are correctly entered
        if (test_type == "F") and ((test_props["df1"] is None) or (test_props["df2"] is None)):
            raise ValueError(
                f"Error in {test_str}: The test statistics {test_type} requires you to specify the numerator \
                                and denominator degrees of freedom.")
        if (test_type not in ["z", "p"]) and (test_props["df1"] is None):
            raise ValueError(
                f"Error in {test_str}: The test statistics {test_type} requires you to specify the degrees of \
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
            family = "Chi"
            df1 = float(df1_raw)
            df2 = None
            stat = stat_raw
        elif test_type == "z":
            family = "Chi"
            df1 = 1
            df2 = None
            stat = stat_raw ** 2
        elif test_type == "p":
            family = "Chi"
            df1 = 1
            df2 = None
            stat = norm.ppf(1 - stat_raw / 2) ** 2

        return test_str_replaced, family, df1, df2, stat

    def _compute_pvals(self):
        family = self.df_tests["family"].values
        df1, df2, stat = self.df_tests[["df1", "df2", "stat"]].to_numpy().T
        pval = np.where(family == "F", 1 - f.cdf(stat, df1, df2), 1 - chi2.cdf(stat, df1))
        return self._bound_pvals(pval)

    def _compute_stouffer_z_at_power(self, power):
        # NCP and pp-values of F tests
        df1_f, df2_f, stat_f = self._sig_f_tests
        ncp_f_est = self._compute_ncp_f_faster(df1_f, df2_f, power)
        pp_f_est = (ncf._cdf(stat_f, df1_f, df2_f, ncp_f_est) - (1 - power)) / power

        # NCP and pp-values of Chi2 tests
        df1_chi, stat_chi = self._sig_chi_tests
        ncp_chi_est = self._compute_ncp_chi_faster(df1_chi, power, use_cache=True)
        pp_chi_est = (ncx2._cdf(stat_chi, df1_chi, ncp_chi_est) - (1 - power)) / power

        # pp-values for all tests
        pp_est = self._bound_pvals(np.hstack([pp_f_est, pp_chi_est]))
        stouffer_at_power = self._compute_stouffer_z(pp_est)
        return stouffer_at_power

    def _solve_power_for_pct(self, pct):
        z = norm._ppf(pct)
        error = lambda est: self._compute_stouffer_z_at_power(est) - z
        return opt.brentq(error, .0501, .99)

    def _compute_ppvals_null(self, pcurvetype="full"):
        """
        Compute the pp-value of the p-value under the null. It simply stretches the p-value over the interval [0, 1]
        :param pval: The p-value
        :param pcurvetype: The type of p-curve (full or half)
        :return:
        """
        p = self.df_tests["p"].values
        if pcurvetype == "full":
            return np.where(p < .05, self._bound_pvals(p * 20), np.nan)
        else:
            return np.where(p < .025, self._bound_pvals(p * 40), np.nan)

    def _compute_ppvals_33(self, pcurvetype="full"):
        family = self.df_tests["family"].values
        df1, df2, stat, p, ncp33 = self.df_tests[["df1", "df2", "stat", "p", "ncp33"]].to_numpy().T
        if pcurvetype == "full":
            pthresh = .05  # Only keep p-values smaller than .05
            propsig = 1 / 3  # Under 33% power, 1/3 of p-values should be lower than .05
        else:
            pthresh = .025  # Only keep p-values smaller than .025
            # We don't know which prop of p-values should be smaller than .025 under 33% power, so compute it
            propsig = 3 * self._compute_prop_lower_33(.025, family, df1, df2, p, ncp33)

        # We then stretch the ppval on the [0, 1] interval.
        pp_33_f = (1 / propsig) * (ncf.cdf(stat, df1, df2, ncp33) - (1 - propsig))
        pp_33_chi = (1 / propsig) * (ncx2.cdf(stat, df1, ncp33) - (1 - propsig))
        pp = np.where(p > pthresh, np.nan,
                      self._bound_pvals(np.where(family == "F", pp_33_f, pp_33_chi)))
        return pp

    def _get_33_power_curve(self):
        family = self.df_tests["family"].values
        df1, df2, p, ncp33 = self.df_tests[["df1", "df2", "p", "ncp33"]].to_numpy().T
        cprop = lambda x: self._compute_prop_lower_33(x, family, df1, df2, p, ncp33)
        propsig = np.array([cprop(c) for c in [.01, .02, .03, .04, .05]])
        diffs = np.diff(propsig, axis=0,
                        prepend=0)  # Difference of CDFs: Likelihood of p-values falling between each value
        props = np.nanmean(3 * diffs, axis=1)
        return props

    def _run_binom_test(self, alternative="null"):
        family = self.df_tests["family"].values
        df1, df2, p, ncp33 = self.df_tests[["df1", "df2", "p", "ncp33"]].to_numpy().T
        k_below_25 = self.n_tests['p025']
        if alternative == "null":
            return 1 - binom(n=self.n_tests['p05'], p=.5).cdf(k_below_25 - 1)
        else:
            prop_below_25_33 = 3 * self._compute_prop_lower_33(.025, family, df1, df2, p, ncp33)
            prop_below_25_33_filtered = prop_below_25_33[p < .05]
            return PoiBin(prop_below_25_33_filtered).cdf(k_below_25)

    def __init__(self, test_arr):
        self.df_tests = (pd.DataFrame(np.array([self._parse_test(t) for t in test_arr]),
                                      columns=["test_str", "family", "df1", "df2", "stat"])
                         .astype({'test_str': str,
                                  'family': str,
                                  "df1": np.float64,
                                  "df2": np.float64,
                                  "stat": np.float64
                                  })
                         )
        self.n_tests = self.df_tests.shape[0]

        self._cached_chi_ncp_ests = dict()  # Generate an empty cache of Chi NCP
        self.df_tests["p"] = self._compute_pvals()
        n_tests = self.df_tests.shape[0]
        n_tests_p05 = self.df_tests[self.df_tests.p <= .05].shape[0]
        n_tests_p025 = self.df_tests[self.df_tests.p <= .025].shape[0]
        n_tests_ns = n_tests - n_tests_p05
        self.n_tests = {"all": n_tests, "p05": n_tests_p05, "p025": n_tests_p025, "ns": n_tests_ns}

        # Unpacking the significant F tests
        sig_f_tests = self.df_tests.query("p < .05 & family == 'F'")
        df1_f, df2_f, stat_f = sig_f_tests[["df1", "df2", "stat"]].to_numpy().T

        # Then significant Chi2 tests
        sig_chi_tests = self.df_tests.query("p < .05 & family != 'F'")
        df1_chi, stat_chi = sig_chi_tests[["df1", "stat"]].to_numpy().T

        self._sig_f_tests = [df1_f, df2_f, stat_f]
        self._sig_chi_tests = [df1_chi, stat_chi]

        import time
        now = time.time()
        self.df_tests["ncp33"] = self._compute_ncp_all(1 / 3)
        print(time.time()-now)
        now = time.time()
        self.df_tests["ncp33"] = self._compute_ncp_all_faster(1/3)
        print(time.time()-now)
        self.df_tests["pp-null-full"] = self._compute_ppvals_null("full")
        self.df_tests["pp-null-half"] = self._compute_ppvals_null("half")
        self.df_tests["pp-33-full"] = self._compute_ppvals_33("full")
        self.df_tests["pp-33-half"] = self._compute_ppvals_33("half")

        self.stouffer_z = dict()
        self.stouffer_p = dict()
        # For all p-curve types and alternatives:
        for s in ["null-full", "null-half", "33-full", "33-half"]:
            # Convert pp-values to z-scores
            pp = self.df_tests[f"pp-{s}"]
            self.df_tests[f"z-{s}"] = norm.ppf(pp)
            # Compute the stouffer Z
            z = self._compute_stouffer_z(pp)
            print(z)
            p = norm.cdf(z)
            self.stouffer_z[s] = z
            self.stouffer_p[s] = p

        levels = self.POWER_LEVELS  # Levels of power on which to run the estimation
        stouffer_at_power_levels = np.array([self._compute_stouffer_z_at_power(p) for p in levels])
        self.z_at_power = np.abs(stouffer_at_power_levels)

    @property
    def has_evidential_value(self):
        ps = self.stouffer_p
        p_half, p_full = ps["null-half"], ps["null-full"]
        return any([p_half < .05, p_full < .05]) or all([p_half < .1, p_full < .1])

    @property
    def has_inadequate_evidence(self):
        ps = self.stouffer_p
        p_half, p_full = ps["33-half"], ps["33-full"]
        return any([p_half < .05, p_full < .05]) or all([p_half < .1, p_full < .1])

    def plot_power_estimate(self):
        """
        self
        :return:
        """
        x = self.POWER_LEVELS
        y = self.z_at_power
        plt.scatter(x, y)
        plt.show()

    def estimate_power(self):
        """
        Estimate the power of the tests entered in the-pcurve, correcting for publication bias.
        :return: lbp, p, ubp: The lower bound, power, and upper bound of the 95% CI for power.
        """
        p = self._solve_power_for_pct(.50)
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
        return lbp, p, ubp

    def plot_pcurve(self):
        """
        Plot the p-curve, as it appears on the online web-app
        :return: A matplotlib ax containing the p-curve
        """
        sns.set_context("paper")

        # Creating the figure containing all axes
        fig = plt.figure(figsize=(6.38, 6.38), dpi=400)

        # Generating the p-curve first

        ##Subsetting significant p-values
        bins = np.arange(0, 0.06, .01)
        pvals = self.df_tests.p.values
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
        lp, p, up = self.estimate_power()
        leg_ax.annotate(f"Est. Power = {p * 100:.0f}%, CI({lp * 100:.0f}%, {up * 100:.0f}%)",
                        (.22, .73), va="center", fontsize=9,
                        color="grey")
        leg_ax.plot([.05, .15], [.55, .55], color="#ee2c2c", ls=":")

        ### Null of no effect
        leg_ax.annotate("Null of no effect", (.18, .55), va="center", fontsize=12)
        p_null_full = self.stouffer_p["null-full"]
        p_null_half = self.stouffer_p["null-half"]
        leg_ax.annotate(
            f"Tests for right-skewness: $p_{{\mathrm{{Full}}}}$ {self._format_pvals(p_null_full)}, "
            f"$p_{{\mathrm{{Half}}}}$ {self._format_pvals(p_null_half)}",
            (.22, .43), va="center", fontsize=9, color="grey")
        leg_ax.plot([.05, .15], [.25, .25], color="#008b45", ls="--")

        ### Null of 33% power
        leg_ax.annotate("Null of 33% power", (.18, .25), va="center", fontsize=12)
        p_33_full = self.stouffer_p["33-full"]
        p_33_half = self.stouffer_p["33-half"]
        p_binomial = self._run_binom_test("33")
        leg_ax.annotate(
            f"Tests for flatness: $p_{{\mathrm{{Full}}}}$ {self._format_pvals(p_33_full)}, "
            f"$p_{{\mathrm{{Half}}}}$ {self._format_pvals(p_33_half)}, "
            f"$p_{{\mathrm{{Binom}}}}$ {self._format_pvals(p_binomial)}",
            (.22, .13), va="center", fontsize=9, color="grey")

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
            f"Note: The observed p-curve includes {self.n_tests['p05']} statistically significant (p < .05) results, "
            f"of which {self.n_tests['p025']} are p < .025."
        )
        if self.n_tests['ns'] == 0:
            footnote += "\nThere were no non-significant results entered."
        elif self.n_tests['ns'] == 1:
            footnote += "\nThere was one result entered but excluded from p-curve because it was p > .05."
        else:
            footnote += (f"\nThere were {self.n_tests['ns']} additional results entered but excluded from p-curve "
                         f"because they were p > .05.")
        footnote_ax.annotate(footnote, (0, 1), va="center", fontsize=7)

        # We tighten the figure and return it
        plt.tight_layout()
        return main_ax
