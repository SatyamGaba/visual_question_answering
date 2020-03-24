import os, csv
import numpy as np
import matplotlib.pyplot as plt

file_name = "san4"
train_prename = "san4-train-log-epoch-"
val_prename = "san4-valid-log-epoch-"
dirx = "logs/"
num_epochs = 30

data = [[],[]]
try:
    for e in range(1, num_epochs+1):
        filenames = [train_prename+"%02d"%e+".txt", val_prename+"%02d"%e+".txt"]
        for i in range(len(filenames)):
            filex = filenames[i]
            filex = os.path.join(dirx, filex)
            with open(filex,"r") as csv_file:
                    reader = csv.reader(csv_file, delimiter='\t')
                    for l in reader:
                        data[i].append(np.array([int(l[0]), float(l[1]), float(l[2]), float(l[3])]))
except:
    print("Total Epochs {:d}".format(e))

data = np.array(data)

train_log = data[0]
val_log = data[1]

epochs = train_log[:,0]
train_loss = train_log[:,1]
train_acc1 = train_log[:,2]
train_acc2 = train_log[:,3]

epochs = val_log[:,0]
val_loss = val_log[:,1]
val_acc1 = val_log[:,2]
val_acc2 = val_log[:,3]

plt.title("Loss with four layered Stacked Attention")
plt.xlabel("Number of Epochs")
plt.ylabel("Cross Entropy Loss")
plt.plot(epochs, train_loss, color='r', label='training loss')
plt.plot(epochs, val_loss, color = 'b', label = 'validation loss')
plt.grid(True)
plt.legend()
plt.savefig("./%s_loss.png"%file_name)
plt.show()


plt.title("Accuracy with four layered Stacked Attention")
plt.xlabel("Number of Epochs")
plt.ylabel("Prediction Accuracy")
plt.plot(epochs, train_acc1, color='r', label='Training Accuracy')
plt.plot(epochs, val_acc1, color = 'b', label = 'Validation Accuracy')
plt.grid(True)
plt.legend()
plt.savefig("./%s_acc.png"%file_name)
plt.show()
