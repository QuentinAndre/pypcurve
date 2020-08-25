pypcurve: A Python Implementation of Simonsohn, Simmons and Nelson's 'p-curve'
============================================================================

# Installation

You can install pypcurve with pip:

    pip install pypcurve

# Using pypcurve

## 1. Compulsory Reading

First and foremost, [read the user guide to the p-curve](http://p-curve.com/guide.pdf). It is crucial that users 
understand what p-curve can and cannot do, that they know which statistical results to select, and that they properly
 prepare the disclosure table. 

## 2. Formatting the statistical results

pypcurve only requires a list of statistical results, stored in a list (or an array). Similar to the p-curve app, 
pypcurve accepts the following formats of statistical tests:
* F(1, 302)=3.273
* t(103)=4.23
* r(76)=.42
* z=1.98
* chi2(1)=7.1

In addition, pypcurve will accept raw p-values:
* p = .0023

This is not recommended though: p-values are often weirdly rounded, so enter the statistical result instead if 
 it is reported in the paper.

## 3. Using pypcurve

### A. Initialization

For this example, I will assume that your tests have been properly formatted, and stored in a column
called "Tests" of a .csv file.

````python
from pypcurve import PCurve
import pandas as pd
df = pd.read_csv("mydata.csv")
pc = PCurve(df.Tests)
````

If all your tests are properly formatted, there will be no error, and pcurve will be initialized properly.

### B. Printing the p-curve output

Next, you can print the summary of the p-curve, as you would see it using the web-app:

````python
pc.summary()
````

This will output the p-curve plot, as well as the table summarizing the binomial and Stouffer tests of the 
p-curve analysis. You can get the plot alone, or the table alone, using the methods `pc.plot_pcurve()` and 
`pc.pcurve_analysis_summary()`.

### C. Power Estimation

You can use pycurve to estimate the power of the design that generated the statistical tests:
 * `pc.estimate_power()` will return the power estimate, and the (lower, upper) bounds of 90% confidence interval.
 * `pc.plot_power_estimate()` will plot the power estimate (as the webapp does).
 
### D. Accessing the results of the p-curve analysis

You can directly access the results of the p-curve analysis using three methods:
* `pc.get_stouffer_tests()` will recover the Z and p-values of the Stouffer tests
* `pc.get_binomial_tests()` will recover the p-values of the binomial tests
* `pc.get_results_entered()` will recover the statistical results entered in the p-curve, and the pp-values and z scores
associated with the different alternatives to which they are compared.

You can also directly check if the p-curve passes the cutoff for evidential value, and the cutoff for 
inadequate evidence (as defined in [Better P-Curve](http://p-curve.com/paper/Better%20p-curves%202015%2011%2026.pdf)), 
using the properties `pc.has_evidential_value` and `pc.has_inadequate_evidence`

# Version History

The app is still in beta, so please take care when interpreting the results. I have tested pypcurve against the 
p-curve app using multiple examples: There are occasional minor deviations between the two, because of the way R (vs.
Python) compute the non-central F distribution.

## Beta

### 0.1.0
First beta release.