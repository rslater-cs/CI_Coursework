import csv

class Train_Logger():

    def __init__(self, filename):
        self.file = open(file=f"./models/logs/{filename}.csv", mode="w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Epoch", "Train_Loss", "Train_Accuracy", "Valid_Accuracy"])

    def put(self, epoch, tloss, taccuracy, vaccuracy):
        self.writer.writerow([epoch, tloss, taccuracy, vaccuracy])

    def close(self):
        self.file.close()

class NSGA_Logger():

    def __init__(self, filename):
        self.file = open(file=f"./models/logs/{filename}.csv", mode="w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Epoch", "Train_Complexity", "Train_Accuracy", "Valid_Complexity", "Valid_Accuracy"])

    def put(self, epoch, tcomp, tacc, vcomp, vacc):
        self.writer.writerow([epoch, tcomp, tacc, vcomp, vacc])

    def close(self):
        self.file.close()