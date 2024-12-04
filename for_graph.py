
import matplotlib.pyplot as plt

training_error=[0.020402631778056814, 0.020808617698235637, 0.06277891597655558, 0.07525138652074126, 0.044105667080154974, 0.08343474751348183, 0.03397972479524274, 0.04294115367739344, 0.11027233115512301, 0.09679697801218122]

testing_error=[0.02612121553738191, 0.023193333585749893, 0.07322832878786498, 0.07164806145472477, 0.05096692057982986, 0.08392585079415814, 0.03814195835761067, 0.055318122771812195, 0.11290150458606056, 0.10235608642266104]

classes = list(range(10)) 

plt.figure(figsize=(10, 6))
plt.bar(classes, training_error, width=0.4, label='Training Error', align='center', alpha=0.7)
plt.bar([c + 0.4 for c in classes], testing_error, width=0.4, label='Testing Error', align='center', alpha=0.7)

plt.xlabel('Class (Digit)', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Training vs Testing Error for Each Class', fontsize=14)
plt.xticks([c + 0.2 for c in classes], classes)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.show()