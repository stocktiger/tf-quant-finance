


def here():
    print(x)


def main():
    global x
    x = 10
    here()

if __name__ == '__main__':
    main()