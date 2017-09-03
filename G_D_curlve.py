import matplotlib.pyplot as plt

y_p5 = []
y_ndcg5 = []

with open('ml-100k/gen_log.txt')as fin:
    for line in fin:
        line = line.split()
        epoch = int(line[0])
        p_5 = float(line[2])
        y_p5.append(p_5)
        ndcg_5 = float(line[5])
        y_ndcg5.append(ndcg_5)

# y_p5 = y_p5[:745]
# y_ndcg5 = y_ndcg5[:745]

x = range(len(y_p5))
plt.figure(figsize=(5, 4))
plt.plot(x, y_p5, label="Generator of IRGAN")
plt.plot([0, max(x)], [0.347368421053, 0.347368421053], linestyle='--', label="LambdaFM")
plt.xlabel('Generator Training Epoch', fontsize=18)
plt.ylabel('Precision@5', fontsize=18)
plt.legend(prop={'size': 12}, loc=2)
plt.grid()
plt.tight_layout()
plt.xlim(0, max(x))
plt.savefig('movielens_p_5_new_54.pdf')

x = range(len(y_p5))
plt.figure(figsize=(6, 5))
plt.plot(x, y_p5, label="Generator of IRGAN")
plt.plot([0, max(x)], [0.347368421053, 0.347368421053], linestyle='--', label="LambdaFM")
plt.xlabel('Generator Training Epoch', fontsize=18)
plt.ylabel('Precision@5', fontsize=18)
plt.legend(prop={'size': 12}, loc=2)
plt.grid()
plt.tight_layout()
plt.xlim(0, max(x))
plt.savefig('movielens_p_5_new_65.pdf')

x = range(len(y_p5))
plt.figure(figsize=(5, 4))
plt.plot(x, y_ndcg5, label="Generator of IRGAN")
plt.plot([0, max(x)], [0.374834165226, 0.374834165226], linestyle='--', label="LambdaFM")
plt.xlabel('Generator Training Epoch', fontsize=18)
plt.ylabel('NDCG@5', fontsize=18)
plt.legend(prop={'size': 12}, loc=2)
plt.grid()
plt.tight_layout()
plt.xlim(0, max(x))
plt.savefig('movielens_ndcg_5_new_54.pdf')

x = range(len(y_p5))
plt.figure(figsize=(6, 5))
plt.plot(x, y_ndcg5, label="Generator of IRGAN")
plt.plot([0, max(x)], [0.374834165226, 0.374834165226], linestyle='--', label="LambdaFM")
plt.xlabel('Generator Training Epoch', fontsize=18)
plt.ylabel('NDCG@5', fontsize=18)
plt.legend(prop={'size': 12}, loc=2)
plt.grid()
plt.tight_layout()
plt.xlim(0, max(x))
plt.savefig('movielens_ndcg_5_new_65.pdf')
