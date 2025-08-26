import DataFrames
import CSV
import PyCall
import PyPlot

df = CSV.read("figure4.csv", DataFrames.DataFrame)

@PyCall.pyimport matplotlib.patches as mpatches
@PyCall.pyimport matplotlib.transforms as mtransforms

fig, ax = PyPlot.subplots(dpi=1200)
range1=1:4000
range2=4001:4200
# Plot data
ax.plot(df.Epoch[range1], df.Validation_errors[range1], label="Validation errors (Pretraining)", color="blue")
ax.plot(df.Epoch[range1], df.Training_errors[range1], label="Training errors (Pretraining)", color="red")
ax.plot(df.Epoch[range2], df.Validation_errors[range2], label="Validation errors (Finetuning)", color="purple")
ax.plot(df.Epoch[range2], df.Training_errors[range2], label="Training errors (Finetuning)", color="green")
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
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.05, 0.95), borderaxespad=0.)

ax.tick_params(labelsize=12)
# --- Zoom Region ---
last_n = 200
x1, x2 = df.Epoch[end - last_n + 1], df.Epoch[end]
y1 = min(minimum(df.Validation_errors[end - last_n + 1:end]), 
         minimum(df.Training_errors[end - last_n + 1:end]))
y2 = max(maximum(df.Validation_errors[end - last_n + 1:end]), 
         maximum(df.Training_errors[end - last_n + 1:end]))

# --- Draw rectangle on main plot ---
zoom_box = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=1.2, edgecolor="black",
                              facecolor="none", linestyle="--")
ax.add_patch(zoom_box)

# --- Inset Axes ---
inset_ax = fig.add_axes([0.6, 0.55, 0.29, 0.29])  # inset location
inset_ax.plot(df.Epoch[end - last_n + 1:end], df.Validation_errors[end - last_n + 1:end], color="purple")
inset_ax.plot(df.Epoch[end - last_n + 1:end], df.Training_errors[end - last_n + 1:end], color="green")
inset_ax.set_yscale("log")
inset_ax.set_title("Finetuning", fontsize=9)
inset_ax.tick_params(labelsize=8)
# Enable minor ticks
inset_ax.minorticks_on()
# Grid: Only on y-axis
inset_ax.yaxis.grid(true, which="major", linestyle=":", linewidth=1)    # major y grid
inset_ax.yaxis.grid(true, which="minor", linestyle=":", linewidth=0.5)  # minor y grid

inset_ax.xaxis.grid(true, which="major", linestyle=":", linewidth=1)    # major y grid

# --- Draw lines from rectangle to inset (ConnectionPatch) ---

# Lower left corner
con1 = mpatches.ConnectionPatch(xyA=(x1, y1), coordsA=ax.transData,
                                xyB=(0, 0), coordsB=inset_ax.transAxes,
                                color="black", linestyle="--", linewidth=0.8)

# Upper right corner
con2 = mpatches.ConnectionPatch(xyA=(x2, y2), coordsA=ax.transData,
                                xyB=(1, 1), coordsB=inset_ax.transAxes,
                                color="black", linestyle="--", linewidth=0.8)

fig.add_artist(con1)
fig.add_artist(con2)

# Display plot
display(fig)
fig.savefig("Figure4.pdf", dpi=1200)
PyPlot.close(fig)

