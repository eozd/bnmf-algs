'''
This is a script to convert Celero tables LaTeX tables.

Input: (Command-line arguments)
    filepath : Path to file containing Celero output table in the following
    format:

    Celero
    Timer resolution: ...
    <TABLE>
    Complete.

    caption : LaTeX table caption string.
    label : LaTeX table label string.

Output: (STDOUT)
    LaTeX table
'''
import sys


def split_celero_row(celero_row):
    '''
    Split a Celero table row to a list of strings.

    A Celero table has the following format:

    | word1 | word2 | ... | wordN |

    Parameters
    ----------
    celero_row : str
        Celero table row.

    Returns
    -------
    word_list : list
        List of words in the given table row.
    '''
    return [word.strip() for word in celero_row.split('|')[1:-1]]


def space(string, count=4):
    '''
    Add count many spaces to the beginning of the given string.

    Parameters
    ----------
    string : str
        String to shift with spaces.

    count : int
        Number of spaces to put.

    Returns
    -------
    shifted_str : str
        String shifted with count many spaces to the right.
    '''
    return ' ' * count + string


def indent(string, level=1, num_spaces=4):
    '''
    Indent the given string given number of levels.

    Parameters
    ----------
    string : str
        String to indent.

    level : int
        Indentation level.

    num_spaces : int
        Number of spaces to put for an indent level.

    Returns
    -------
        Indented string.
    '''
    return space(string, level * num_spaces)


def newline(string, count=1):
    '''
    Add count many newline characters to the end of the given string.

    Parameters
    ----------
    string : str
        String to concatenate newline characters.

    count : int
        Number of newline characters to put.

    Returns
    -------
    newlined_str : str
        String that contains count many more newlines at the end.
    '''
    return string + '\n' * count


def bold(string):
    '''
    Return bold version of the given string in LaTeX syntax.

    Parameters
    ----------
    string : str
        String to return in LaTeX bold format.

    Returns
    -------
    bold_str : str
        String in LaTeX bold format.
    '''
    return r'\textbf{' + string + r'}'


def construct_latex_table(col_names, table, caption, label, resolution):
    '''
    Construct a LaTeX table from the column names and values, caption, label
    and Celero resolution string.

    Parameters
    ----------
    col_names : list of str
        Names of columns. These will be the headings of each column.

    table : list of list of str
        Table containing the value of each row. Each row must be represented as
        a list of strings.

    caption : str
        Caption of the table.

    label : str
        Label of the table.

    resolution : str
        Celero resolution line.

    Returns
    -------
    latex : str
        LaTeX table version of the given parameters representing a Celero table.
    '''
    latex = newline(r'\begin{table}[H]')
    latex += newline(indent(r'\centering'))
    latex += newline(
        indent(r'\caption{' + caption + ' ({}) '.format(resolution) + r'}'))
    latex += newline(indent(r'\label{' + label + r'}'))
    latex += newline(
        indent(r'\begin{tabular}{|' + r'c|' * len(col_names) + r'}'))
    latex += newline(indent(r'\hline', level=2))

    latex += indent('', level=2)
    for name in col_names[:-1]:
        latex += bold(name) + ' &'
    latex += newline(bold(col_names[-1]) + r' \\')

    latex += newline(indent(r'\hline', level=2))
    for row in table:
        latex += indent('', level=2)
        for val in row[:-1]:
            latex += val.replace('_', r'\_') + r' & '
        latex += newline(row[-1] + r' \\')
        latex += newline(indent(r'\hline', level=2))

    latex += newline(indent(r'\end{tabular}'))
    latex += newline(r'\end{table}')

    return latex


def main():
    '''
    Read filepath, caption and label from command-line, read the Celero table
    in the given filepath, construct the corresponding LaTeX table and print
    it out to STDOUT.
    '''
    filepath = sys.argv[1]
    caption = sys.argv[2]
    label = sys.argv[3]

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    lines = [line for line in lines if line]
    lines = lines[1:-1]

    resolution = lines[0]
    col_names = split_celero_row(lines[1])

    table = []
    for row in lines[3:]:
        table.append(split_celero_row(row))

    latex_table = construct_latex_table(col_names, table, caption, label,
                                        resolution)
    print(latex_table, end='')


if __name__ == '__main__':
    main()
