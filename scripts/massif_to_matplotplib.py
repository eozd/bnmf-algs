'''
Parse memory usage files created using ms_print from valgrind massif and plot
the graphs using matplotlib.

This script reads (alg_name, filepath) pairs from STDIN, parses ms_print file
of each algorithm, and draws a memory consumption graph for each algorithm.
ms_print graphs are parsed by only considering '#' and '@' columns.

Input : (From STDIN) alg_name filepath for each line. For example:

bld_add bld_add/memory.massif
nmf nmf/memory.massif
...

Output : Plots memory usage graph of all algorithms in a single matplotlib plot.
This plot is shown to user and also saved to a file called
'memory_matplotlib.pdf'.
'''
import sys
import argparse
import re
import matplotlib.pyplot as plt


def read_graph(file):
    '''
    Read a ms_print graph from the given open file object and return it as
    an str.

    The given open file object should contain a sequence of empty lines,
    the graph and another sequence of empty lines. This is the default format
    written by ms_print.

    Parameters
    ----------
    file : file object
        Open file object that contains a sequence of empty lines ('\n'), the
        graph to be parsed and another sequence of empty lines ('\n').

    Returns
    -------
    graph_str : str
        ms_print memory graph as a single string.
    '''
    for line in file:
        if line.rstrip():
            break

    graph_str = line
    for line in file:
        if not line.rstrip():
            break
        graph_str += line

    return graph_str


def parse_ms_print_file(filepath):
    '''
    Parse ms_print memory file in the given filepath and return the memory usage
    graph in the file as an str.

    Parameters
    ----------
    filepath : str
        Path to the ms_print file containing memory usage graph. File in the
        given filepath must be in ms_print format

    Returns
    -------
    graph_str : str
        ms_print memory graph as a single string.
    '''
    with open(filepath, 'r') as f:
        for line in f:
            if not line.rstrip():
                graph_str = read_graph(f)
                break

    graph_str = graph_str.replace(':', ' ').strip()
    return graph_str


def get_ylabel(graph_lines):
    '''
    Get the ylabel of the graph from an ms_print graph in a list of str format.

    Parameters
    ----------
    graph_lines : list of str
        Each element of graph_lines must contain a single line of the ms_print
        graph. These lines should not contain '\n' characters.

    Returns
    -------
    ylabel : str
        Vertical (y) label of the graph. In ms_print format, this is generally
        a string such as KB, MB, GB, etc.
    '''
    return graph_lines[0]


def get_xlabel(graph_lines):
    '''
    Get the xlabel of the graph from an ms_print graph in a list of str format.

    Parameters
    ----------
    graph_lines : list of str
        Each element of graph_lines must contain a single line of the ms_print
        graph. These lines should not contain '\n' characters.

    Returns
    -------
    xlabel : str
        Horizontal (x) label of the graph. In ms_print format, this is generally
        a string such as Gi.
    '''
    return graph_lines[-2][graph_lines[-2].find('>') + 1:]


def get_ymax(graph_lines):
    '''
    Get the maximum y value from an ms_print graph in a list of str format.

    Parameters
    ----------
    graph_lines : list of str
        Each element of graph_lines must contain a single line of the ms_print
        graph. These lines should not contain '\n' characters.

    Returns
    -------
    ymax : float
        Maximum y value in the graph. In ms_print format, this value is written
        next to the vertical axis.
    '''
    return float(graph_lines[1][:graph_lines[1].find('^')])


def get_xmax(graph_lines):
    '''
    Get the maximum x value from an ms_print graph in a list of str format.

    Parameters
    ----------
    graph_lines : list of str
        Each element of graph_lines must contain a single line of the ms_print
        graph. These lines should not contain '\n' characters.

    Returns
    -------
    xmax : float
        Maximum x value in the graph. In ms_print format, this value is written
        next to the horizontal axis.
    '''
    return float(graph_lines[-1].split(' ')[-1])


def get_graph_mat(graph_lines):
    '''
    Construct and return graph matrix from an ms_print graph in a list of str
    format.

    A graph matrix is a list of strings of equal length. Each string contains
    only the characters used to represent the ms_print graph. These characters
    are ' ', '@', '#', ':'.

    Parameters
    ----------
    graph_lines : list of str
        Each element of graph_lines must contain a single line of the ms_print
        graph. These lines should not contain '\n' characters.

    Returns
    -------
    graph_mat : list of str
        Graph matrix.
    '''
    beg_index = graph_lines[2].find('|') + 1
    end_index = graph_lines[-2].find('>') + 1

    graph_mat = []
    for line in graph_lines[1:-2]:
        graph_mat.append(line[beg_index:end_index])

    return graph_mat


def get_x_y(graph_mat, xmax, ymax):
    '''
    Find the x and y points of each column drawn using '@' and '#' in an ms_print
    graph matrix.

    The return (x, y) points are the tip of the columns drawn using '@' and '#'.
    x values of these columns are calculated by proportionalizing the maximum x
    value to the index of the column. y values are calculated by proportionalizing
    the maximum y value to the height of the column.

    Parameters
    ----------
    graph_mat : list of str
        Graph matrix. Each line of graph matrix is of equal length and contains
        the characters used to represent the ms_print graph.

    xmax : float
        Maximum x value in the ms_print graph.

    ymax : float
        Maximum y value in the ms_print graph.

    Returns
    -------
    x : list of float
        x values of the columns drawn using '@' and '#'.

    y : list of float
        y values of the columns drawn using '@' and '#'.
    '''
    n_rows = len(graph_mat)
    indices = []
    coeffs = []
    for j, char in enumerate(graph_mat[-1]):
        if char in '@#':
            indices.append(j)
            for i in range(n_rows - 1, -1, -1):
                if graph_mat[i][j] not in '@#':
                    coeffs.append((n_rows - i - 1))
                    break
            else:
                coeffs.append(n_rows)

    coeffs = [coeff / max(coeffs) for coeff in coeffs]

    x = [index * xmax / indices[-1] for index in indices]
    y = [coeff * ymax for coeff in coeffs]

    return x, y


def main():
    '''
    Read the algorithm name and its ms_print filepath from STDIN, calculate the
    x and y values of each detailed/max column in the ms_print graph and draw
    the found points on a matplotlib graph. Each algorithm and its points are
    drawn on the same graph for comparison reasons.

    After each algorithm is plotted, a final plot is constructed by setting the
    limits of the plot to the global maximum x and y values. A pdf version of
    this plot is saved in memory_matplotlib.pdf file.
    '''
    global_xlim = -1000.0
    global_ylim = -1000.0

    alg_filepath = []
    for line in sys.stdin:
        alg_filepath.append(line.strip().split(' '))

    fig = plt.figure()
    for alg, filepath in alg_filepath:
        graph_str = parse_ms_print_file(filepath)
        graph_lines = [line.strip('\n') for line in graph_str.split('\n')]

        x_label = get_xlabel(graph_lines)
        y_label = get_ylabel(graph_lines)
        xmax = get_xmax(graph_lines)
        ymax = get_ymax(graph_lines)

        global_ylim = max(global_ylim, ymax)

        graph_mat = get_graph_mat(graph_lines)
        x, y = get_x_y(graph_mat, xmax, ymax)

        global_xlim = max(global_xlim, max(x))

        plt.plot(x, y, label=alg, linewidth=2.5)

    plt.xlabel('Number of instructions ({})'.format(x_label))
    plt.ylabel('Memory ({})'.format(y_label))
    plt.xlim([0, global_xlim])
    plt.ylim([0, global_ylim])
    plt.title('Memory usage of different algorithms')
    plt.legend()
    fig.savefig('memory_matplotlib.pdf', format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
