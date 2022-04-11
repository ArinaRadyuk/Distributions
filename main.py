import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def normal(n):
    print("Enter mu:")
    mu = float(input())
    print("Enter sigma:")
    sigma = float(input())

    # Selection
    x = np.random.normal(mu, sigma, n)

    # Grid for plotting density
    grid = np.linspace(mu - 3*sigma, mu + 3*sigma, n)

    # Plot size
    plt.figure(figsize=(14, 8))

    # Density
    plt.plot(grid, st.norm.pdf(grid, mu, sigma))

    # Histogram
    plt.hist(x, bins=50, density=True, color='royalblue', alpha=0.7)

    plt.show()

    # Information
    print("Information")
    print("Math expectation:", mu)
    print("Mean:", x.mean())
    print("Diff:", abs(mu - x.mean()))


def binomial(n):
    print("Enter p:")
    p = float(input())
    # Selection
    x = st.binom.rvs(n, p, size=n)

    # Plot size
    plt.figure(figsize=(14, 8))

    # Histogram
    plt.hist(x, bins=30, density=True, color='royalblue', alpha=0.7)
    plt.show()

    # Information
    print("Information")
    print("Math expectation:", n*p)
    print("Mean:", x.mean())
    print("Diff:", abs(n*p - x.mean()))


def exponential(n):
    print("Enter lambda:")
    la = int(input())
    scale = 1 / la

    # Selection
    x = st.expon.rvs(scale=scale, size=n)

    # Grid for plotting density
    grid = np.linspace(st.expon.ppf(scale=scale, q=0.001),
                       st.expon.ppf(scale=scale, q=0.999), n)

    # Plot size
    plt.figure(figsize=(14, 8))

    # Density
    plt.plot(grid, st.expon.pdf(grid, scale=scale))

    # Histogram
    plt.hist(x, bins=30, density=True, color='royalblue', alpha=0.7)
    plt.show()

    # Information
    print("Information")
    print("Math expectation:", scale)
    print("Mean:", x.mean())
    print("Diff:", abs(scale - x.mean()))


def uniform(n):
    print("Enter a:")
    a = int(input())
    print("Enter b:")
    b = int(input())

    if a > b:
        a, b = b, a

    # Selection
    x = st.uniform.rvs(loc=a, scale=b-a, size=n)

    # Grid for plotting density
    grid = np.linspace(st.uniform.ppf(loc=a, scale=b-a, q=0.001),
                       st.uniform.ppf(loc=a, scale=b-a, q=0.999), n)

    # Plot size
    plt.figure(figsize=(14, 8))

    # Density
    plt.plot(grid, st.uniform.pdf(grid, loc=a, scale=b-a))

    # Histogram
    plt.hist(x, bins=5, density=True, color='royalblue', alpha=0.7)
    plt.show()

    # Information
    print("Information")
    print("Math expectation:", (a + b)/2)
    print("Mean:", x.mean())
    print("Diff:", abs((a + b)/2 - x.mean()))


def main():
    print("Enter N:")
    n = int(input())
    print("Enter distribution:", "1 - normal", "2 - uniform", "3 - binomial", "4 - exponential", sep="\n")
    x = int(input())
    if x == 1:
        normal(n)
    elif x == 2:
        uniform(n)
    elif x == 3:
        binomial(n)
    elif x == 4:
        exponential(n)


if __name__ == '__main__':
    main()

