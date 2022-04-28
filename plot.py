import matplotlib.pyplot as plt
import sys

def read_threads_time(filename: str):
    x = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            line_split = line.split(':')
            x.append(float(line_split[0].strip()))
            y.append(float(line_split[1].strip()))
    return x, y


def main(filename):
    x, y = read_threads_time(filename)

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', markersize=4, linewidth=2, color='#f97b04')
    ax.title.set_text('RDF with dr = 0.1')

    ax.set_xlabel('r')
    ax.set_ylabel('rdf(r)')

    ax.set_ylim(bottom=0)

    ax.grid(which='major', color='#cccccc', linestyle=':')

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Source filename not specified, defaulting to data.txt')
        filename = 'data.txt'
    else:
        filename = sys.argv[1]
    main(filename)
