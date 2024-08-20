import os
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--id', type=int, default=0)
    parser.add_argument('-m', type=int, default=2048)
    parser.add_argument('-n', type=int, default=14336)
    parser.add_argument('-k', type=int, default=4096)
    parser.add_argument('-e', type=int, default=8, help='number of experts')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--topk', type=int, default=2)

    parser.add_argument('-b', type=int, default=1, help='batch generate problem')

    args = parser.parse_args()
    return args

def divide_number_randomly(n, parts):
    # Generate 7 random non-negative integers
    numbers = [random.randint(1, n) for _ in range(parts)]
    
    # Calculate the 8th number so that the sum of all numbers is n
    numbers.append(n - sum(numbers))
    
    # Check if any number is negative or if the 8th number is out of bounds
    # If so, recursively call the function until we get a valid set
    while min(numbers) < 0 or max(numbers) > n:
        numbers = [random.randint(1, n) for _ in range(parts)]
        numbers.append(n - sum(numbers))

    return numbers

def main():
    args = parse_args()

    n, k = args.n, args.k
    total = args.topk * args.m

    for pid in range(args.b):
        ms = divide_number_randomly(total, args.e)

        text = ""
        for i, m in enumerate(ms):
            line = f"{i} {m}x{n}x{k}\n"
            text += line

        with open(f'benchmarks/workspace/problem{pid}.txt', 'w') as file:
            file.write(text)


if __name__ == '__main__':
    main()
