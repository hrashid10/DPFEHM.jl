import DataFrames
import CSV
import PyCall
import PyPlot

df = CSV.read("figure3.csv", DataFrames.DataFrame)


@PyCall.pyimport matplotlib.patches as mpatches
@PyCall.pyimport matplotlib.transforms as mtransforms

fig, ax = PyPlot.subplots(dpi=1200)

# Plot data
ax.plot(df.Epoch, df.Validation_errors, label="Validation errors", color="blue")
ax.plot(df.Epoch, df.Training_erros, label="Training errors", color="red")

# Set log scale on y-axis
ax.set_yscale("log")

# Axis labels and title
ax.set_xlabel("Epochs", fontsize=12)
ax.set_ylabel("Pressure Errors, MPa", fontsize=12)


# Enable minor ticks
ax.minorticks_on()
# Grid: Only on y-axis
ax.yaxis.grid(true, which="major", linestyle=":", linewidth=1)    # major y grid
ax.yaxis.grid(true, which="minor", linestyle=":", linewidth=0.5)  # minor y grid

ax.xaxis.grid(true, which="major", linestyle=":", linewidth=1)    # major y grid
# ax.xaxis.grid(false, which="minor")  # minor y grid

# Legend
ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95), borderaxespad=0.)
ax.tick_params(labelsize=12)
# Display plot
display(fig)
fig.savefig("Figure3.pdf", dpi=1200)
PyPlot.close(fig)

