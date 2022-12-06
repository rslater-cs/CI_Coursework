import csv
import matplotlib.pyplot as plt

def get_stats(filename, headers):
    stats = []

    with open(filename, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            infos = []
            for header in headers:
                infos.append(row[header])
            stats.append(infos)

        file.close()

    return stats

def display_axis2d(files, labels, headers, casts, join_axis=0):
    if len(headers) != 2:
        return

    data = []

    i = 0
    for file in files:
        stats = get_stats(file, headers)

        if i != 0:
            temp = []
            for stat in stats:
                index = (join_axis+1)%2
                temp.append(casts[index](stat[index]))
            data.append(temp)
        else:
            for j in range(2):
                temp = []
                for stat in stats:
                    temp.append(casts[j](stat[j]))
                data.append(temp)

        i += 1

    x_data = data[join_axis]
    y_data = data[:join_axis] + data[join_axis+1:]

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    for i in range(len(y_data)):
        plt.plot(x_data, y_data[i], label=labels[i])

    plt.legend(loc='lower right')

    plt.axvline(x=49, label="PSO start", color="red", ls="--")
    plt.text(46,10,'SL-PSO start',rotation=90)

    plt.savefig(f'./models/results/{headers[(join_axis+1)%2]}.png')
    plt.show()

if __name__ == "__main__":
    display_axis2d(["./models/logs/pso.csv", "./models/logs/sgd.csv", "./models/logs/padam_p_0.csv"], ["SGD+SL-PSO", "SGD", "Padam"],["Epoch", "Valid_Accuracy"], [int, float, float])
    display_axis2d(["./models/logs/pso.csv", "./models/logs/sgd.csv", "./models/logs/padam_p_0.csv"], ["SGD+SL-PSO", "SGD", "Padam"],["Epoch", "Train_Accuracy"], [int, float, float])
            
