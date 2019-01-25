# if ' ' == '\t':
#     print('true')
import tensorflow as tf
import csv

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

lines = _read_tsv('data/baike.train.tsv')
# print(lines[0])
with open('data/baike_train.tsv','w',encoding='utf-8') as f_:
    for line in lines:
        # print(line)
        split_line = line[1].split(',',2)
        # print(split_line)
        new_line = line[0]+ '\t' + split_line[1] + '\t' + split_line[0] + split_line[2] + '\n'
        f_.write(new_line)