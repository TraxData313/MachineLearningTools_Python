# - takes predicted and real binary labes and gives prediction correctness:
def classification_rate(Y, P):
	return np.mean(Y == P)
