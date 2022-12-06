import csv 

with open("./models/logs/padam_p_0.csv", "r") as main_file:
    with open("./models/logs/padam_p_0_cleaned.csv", "w", newline="") as output_file:
        reader = csv.reader(main_file)
        writer = csv.writer(output_file)

        progress = 0

        for row in reader:
            if progress > 0:
                acc = row[2]
                acc = acc.replace("tensor(", "")
                acc = acc.replace(", device='cuda:0')", "")
                print(acc)
                row[2] = float(acc)*100/45000

            writer.writerow(row)

            progress += 1
