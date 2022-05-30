import sys

def main(nparts : int):
    box_dim = (128**3 / (1e6/nparts))**(1/3)
    print(box_dim)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Enter number of particle for which to find sizes of box')
        exit(1)
    else:
        nparts = int(sys.argv[1])
    main(nparts)
