import re

accuracy_matcher = re.compile('accuracy: (.*)')


def get_avg(data):
    sum_data = 0
    for d in data:
        sum_data += float(d)
    return sum_data / len(data)


def read_data(file_path):
    f = open(file_path, 'r')
    data = f.read()
    f.close()

    accuracies = accuracy_matcher.findall(data)
    avg = get_avg(accuracies)
    print(avg)


read_data('outputs/run1.txt')
