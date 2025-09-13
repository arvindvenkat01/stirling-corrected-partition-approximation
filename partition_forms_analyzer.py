# MIT License
# Copyright (c) 2025 Arvind N. Venkat
# Permission is hereby granted, free of charge...


import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gammaln


class TheoreticalHybridTester:
    def __init__(self, max_n_to_compute=80000):
        self.method_names = {
            'p_tilde_theoretical': 'Theoretical (Fixed)', 
            'p_tilde_adaptive': 'Adaptive Scaling', 
            'hardy_ramanujan': 'Hardy-Ramanujan',
        }
        print("="*60)
        print("THEORETICAL HYBRID FORMS: EXTENDED COMPARISON")
        print("="*60)
        self._precompute_pentagonals(max_n_to_compute)
        self._precompute_partitions_iteratively(max_n_to_compute)

    def _precompute_pentagonals(self, N):
        G, S = [], []
        k = 1
        while True:
            g1 = k * (3 * k - 1) // 2
            g2 = k * (3 * k + 1) // 2
            if g1 > N and g2 > N:
                break
            sign = 1 if (k % 2 == 1) else -1
            if g1 <= N:
                G.append(g1)
                S.append(sign)
            if g2 <= N:
                G.append(g2)
                S.append(sign)
            k += 1
        self._G = G
        self._S = S

    def _precompute_partitions_iteratively(self, max_n):
        print(f"Pre-computing partition values up to n={max_n}...")
        t0 = time.time()
        p = [0] * (max_n + 1)
        p[0] = 1
        G, S = self._G, self._S
        for n in range(1, max_n + 1):
            total = 0
            for j in range(len(G)):
                g = G[j]
                if g > n:
                    break
                total += S[j] * p[n - g]
            p[n] = total
        self._p = p
        print(f"Pre-computation done in {time.time() - t0:.2f} seconds.")

    def exact_partition(self, n):
        return self._p[n]

    @staticmethod
    def log_central_binomial(n):
        k = n // 2
        return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))

    @staticmethod
    def log_hardy_ramanujan(n):
        return (-math.log(4) - math.log(n) - 0.5 * math.log(3) +
                math.pi * math.sqrt(2 * n / 3))

    @staticmethod
    def lambda_n(n):
     """
        Calculates the value of Lambda(n) based on the derived asymptotic form.

        This function corresponds to the logarithmic form of L(n), which is
        the cube root of the ratio of the central binomial coefficient to the
        partition function.

        Args:
            n (int): The integer for which to calculate Lambda.

        Returns:
            float: The calculated value of Lambda(n).
        """
        ln2_div_3 = math.log(2) / 3
        pi_sqrt6_div_9 = math.pi * math.sqrt(6) / 9
        one_sixth = 1 / 6
        c0 = (1 / 3) * math.log((4 * math.sqrt(3)) / math.sqrt(math.pi / 2))
        return c0 + ln2_div_3 * n - pi_sqrt6_div_9 * math.sqrt(n) + one_sixth * math.log(n)

    def log_p_tilde_theoretical(self, n):
        log_binom = self.log_central_binomial(n)
        return log_binom - 3 * self.lambda_n(n)

    # Method for the adaptive scaling formula
    def log_p_tilde_adaptive(self, n):
        log_binom = self.log_central_binomial(n)
        alpha_n = 3 + (1 / (120 * n))
        return log_binom - alpha_n * self.lambda_n(n)

    def run_experiment(self, test_points):
        print("\n" + "="*80)
        print("STARTING EXPERIMENT")
        methods = {
            'p_tilde_theoretical': self.log_p_tilde_theoretical,
            'p_tilde_adaptive': self.log_p_tilde_adaptive, # <<< ADDED
            'hardy_ramanujan': self.log_hardy_ramanujan,
        }
        results = {k: {'errors': [], 'n_values': []} for k in methods}

        for n in test_points:
            exact_val = self.exact_partition(n)
            if exact_val <= 0:
                continue
            log_exact = math.log(exact_val)
            for key, f in methods.items():
                log_pred = f(n)
                rel_pct = abs(math.exp(log_pred - log_exact) - 1) * 100
                results[key]['errors'].append(rel_pct)
                results[key]['n_values'].append(n)
        print("Experiment finished.")
        return results

    def generate_error_table(self, results, ranges):
        import numpy as np
        rows = []
        formula_order = ['p_tilde_theoretical', 'p_tilde_adaptive', 'hardy_ramanujan']
        for r0, r1 in ranges:
            for key in formula_order:
                data = results[key]
                errs = [e for n_val, e in zip(data['n_values'], data['errors']) if r0 <= n_val <= r1]
                if errs:
                    rows.append({
                        "Range": f"{r0}-{r1}",
                        "Formula": self.method_names[key],
                        "Mean Error (%)": f"{np.mean(errs):.6f}",
                        "Median Error (%)": f"{np.median(errs):.6f}",
                        "Max Error (%)": f"{np.max(errs):.6f}",
                        "Min Error (%)": f"{np.min(errs):.6f}",
                    })
        df = pd.DataFrame(rows)
        print("\nError Summary Table:")
        print(df.to_string(index=False))
        df.to_csv("partition_error_summary_adaptive.csv", index=False) # <<< CHANGED
        print("\nSaved 'partition_error_summary_adaptive.csv'")
        return df

    def analyze_and_plot(self, results):
        colors = ['blue', 'green', 'black']
        markers = ['o', 's', 'x']
        linestyles = ['-', '--', '-.']

        # Main percent error plot, log y-scale
        plt.figure(figsize=(12,8))
        plot_keys = ['p_tilde_theoretical', 'p_tilde_adaptive', 'hardy_ramanujan']
        for i, key in enumerate(plot_keys):
            plt.plot(results[key]['n_values'], results[key]['errors'],
                     label=self.method_names[key],
                     color=colors[i], marker=markers[i], linestyle=linestyles[i], alpha=0.9)
        plt.xlabel("n")
        plt.ylabel("Absolute Percent Error (%)")
        plt.title("Error vs n (Log Scale Y-Axis)")
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", linestyle="--")
        plt.tight_layout()
        plt.savefig("partition_performance_adaptive.png", dpi=300) # <<< CHANGED
        plt.show()

def main():
    tester = TheoreticalHybridTester(max_n_to_compute=80000)
    test_points = list(range(100, 1001, 100)) + list(range(2000, 80001, 1000))
    start_time = time.time()
    results = tester.run_experiment(test_points)
    print(f"\nExperiment completed in {time.time() - start_time:.2f} seconds")
    tester.generate_error_table(results, [(100, 1000), (1001, 10000), (10001, 80000)])
    tester.analyze_and_plot(results)


if __name__ == "__main__":
    main()