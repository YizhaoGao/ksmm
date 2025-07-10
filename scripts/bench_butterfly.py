from typing import List, Tuple
import subprocess


def create_butterfly_patterns(n: int) -> List[Tuple[int, int, int, int]]:
    """
    Generates the chain of Kronecker-Sparse patterns for a butterfly
    decomposition of a 2^n x 2^n square matrix.

    A butterfly matrix W of size 2^n x 2^n can be factored into n
    Kronecker-Sparse matrices: W = K_1 * K_2 * ... * K_n.

    The ℓ-th factor, K_ℓ, has a pattern corresponding to (a, b, c, d) where:
    a = 2**(ℓ-1)
    b = 2
    c = 2
    d = 2**(n-ℓ)

    Args:
        n: The power of 2 defining the matrix dimension (2^n x 2^n).
           Must be a positive integer.

    Returns:
        A list of n tuples, where each tuple is an (a, b, c, d) pattern.
        The list is ordered for direct use in libraries like ksmm,
        meaning patterns[0] corresponds to the right-most matrix factor (K_n)
        and patterns[-1] corresponds to the left-most factor (K_1).
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input 'n' must be a positive integer.")

    patterns = []
    # The loop variable 'l' corresponds to the index in the matrix product K_1 * ... * K_n
    for l in range(1, n + 1):
        a = 2**(l - 1)
        b = 2
        c = 2
        d = 2**(n - l)
        pattern = (a, b, c, d)
        patterns.append(pattern)

    # Reverse the list to match the convention where the first pattern in the list
    # corresponds to the right-most matrix in the product.
    # W = K_1 * K_2 * ... * K_n
    # patterns = [pattern_for_Kn, ..., pattern_for_K2, pattern_for_K1]
    return patterns[::-1]



if __name__ == "__main__":
    for n in range(6, 11):
        patterns = create_butterfly_patterns(n)
        print(f"Butterfly patterns for size {2**n} n={n}: {patterns}")
        ## run python bench.py --patterns "{patterns}" --output_file "results/butterfly_{n}.json"
        subprocess.run(
            [
                "python", "bench.py",
                "--patterns", str(patterns),
                "--output_file", f"results/butterfly_{n}.json"
            ],
            check=True
        )



