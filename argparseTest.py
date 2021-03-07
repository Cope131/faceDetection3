import argparse


# parse arguments from the terminal

def fib(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a


def main():
    parser = argparse.ArgumentParser()
    # positional argument - only name
    parser.add_argument('num', help='fibonacci number to calculate', type=int)
    # optional argument - shortcut, full
    parser.add_argument('-o', '--output', help='Output result', action='store_true')
    # get arguments from the parser
    args = parser.parse_args()

    # output
    result = fib(args.num)
    print('result: ', result)
    print(type(args))

    if args.output:
        print('result stored (example): ', result)


if __name__ == '__main__':
    main()
