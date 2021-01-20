import matplotlib.pyplot as plt
import ml


def show_graphs(axs_x, theta, depend_std, result, loss_mem):
	fig, axs = plt.subplots(2, 1, figsize=(10, 6))
	axs[0].scatter(axs_x, result)
	axs[0].plot(axs_x, ml.apply_theta(theta, depend_std), color="red", linewidth=2)
	for idx, elem in enumerate(theta):
		axs[0].plot([], [], ' ', label=f"t_{idx}: {round(elem, 2):.2f}")
	axs[0].legend()

	axs[1].plot(range(len(loss_mem)), loss_mem, marker='o', color="magenta")
	axs[1].plot([], [], ' ', label=f"iter : {len(loss_mem) - 1}")
	axs[1].plot([], [], ' ', label=f"loss : {round(loss_mem[-1]):.2f}")
	if len(loss_mem) > 1:
		axs[1].plot([], [], ' ', label=f"delta: {abs(loss_mem[-1] - loss_mem[-2]):.4f}")
	axs[1].legend()
	plt.show()
