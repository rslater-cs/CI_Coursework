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

def get_stats_tuning(filename, headers):
    stats = {}

    for header in headers:
        stats[header] = []

    with open(filename, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            for header in headers:
                stats[header].append(row[header])

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
    plt.ylim([0,100])

    for i in range(len(y_data)):
        plt.plot(x_data, y_data[i], label=labels[i])

    plt.legend(loc='lower right')

    plt.axvline(x=49, label="PSO start", color="red", ls="--")
    plt.text(46,10,'SL-PSO start',rotation=90)

    plt.savefig(f'./models/results/{headers[(join_axis+1)%2]}.png')
    plt.show()

def display_tuning(file_path, name, headers):
    stats = get_stats_tuning(file_path, headers)

    for key, value in stats.items():
        if key != "Epoch":
            stats[key] = [float(entry) for entry in value]
        else:
            stats[key] = [int(entry) for entry in value]
    

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([0,100])


    for header in headers[1:]:
        plt.plot(stats["Epoch"], stats[header], label=header)

    plt.legend(loc='lower right')

    plt.savefig(f'./models/results/{name}.png')
    plt.show()


if __name__ == "__main__":
    display_axis2d(["./models/logs/pso.csv", "./models/logs/sgd.csv", "./models/logs/padam_best.csv"], ["SGD+SL-PSO", "SGD", "Padam"],["Epoch", "Valid_Accuracy"], [int, float, float])
    display_axis2d(["./models/logs/pso.csv", "./models/logs/sgd.csv", "./models/logs/padam_best.csv"], ["SGD+SL-PSO", "SGD", "Padam"],["Epoch", "Train_Accuracy"], [int, float, float])
    display_tuning("./models/logs/padam_tuning.csv", "padam_hyperparams_train", ["Epoch","TA0","TA0.125","TA0.2","TA0.25","TA0.375","TA0.5"])
    display_tuning("./models/logs/padam_tuning.csv", "padam_hyperparams_valid", ["Epoch","VA0","VA0.125","VA0.2","VA0.25","VA0.375","VA0.5"])
            
