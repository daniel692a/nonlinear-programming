import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animar_puntos(puntos):
    fig, ax = plt.subplots()
    ax.set_xlim(min(p[0] for p in puntos) - 1, max(p[0] for p in puntos) + 1)
    ax.set_ylim(min(p[1] for p in puntos) - 1, max(p[1] for p in puntos) + 1)

    lines = []
    label = None

    def init():
        return lines + [label] if label else lines

    def update(i):
        nonlocal label

        if i == 0:
            return lines + [label] if label else lines

        xdata = [puntos[i-1][0], puntos[i][0]]
        ydata = [puntos[i-1][1], puntos[i][1]]
        line, = ax.plot(xdata, ydata, 'ro-', animated=False)
        lines.append(line)

        if label:
            label.remove()

        label = ax.text(puntos[i][0], puntos[i][1], f'({round(puntos[i][0], 2)}, {round(puntos[i][1], 2)},)', fontsize=12, ha='right')

        return lines + [label]

    ani = animation.FuncAnimation(fig, update, frames=len(puntos), init_func=init,
                                  blit=False, interval=1500, repeat=False)

    plt.show()