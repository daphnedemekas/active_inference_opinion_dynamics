import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import figure 
from ipywidgets import interact 
from matplotlib.widgets import Slider

df = pd.read_pickle("param_data.pkl")

def get_ratio_from_parameters(c, e_l, e_u, env_l, env_u, b_l, b_u):
    indices = (c, e_l, e_u, env_l, env_u, b_l, b_u)
    return df[indices]


connectedness_values = [0.2,0.4]
lower_bounds = [1,3]
upper_bounds = [i+j for i in lower_bounds for j in range(3)]

ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.2, top=0.75)

#adjustable range values
ax_range = fig.add_axes(upper_bounds)
ax_range.spines['top'].set_visible(True)
ax_range.spines['right'].set_visible(True)

#adjustable connectedness values
ax_c = fig.add_axes(connectedness_values)
ax_c.spines['top'].set_visible(True)
ax_c.spines['right'].set_visible(True)

#adjustable idea precision values
ax_env_l = fig.add_axes(lower_bounds)
ax_env_l.spines['top'].set_visible(True)
ax_env_l.spines['right'].set_visible(True)

#adjustable idea precision values
ax_env_u = fig.add_axes(upper_bounds)
ax_env_u.spines['top'].set_visible(True)
ax_env_u.spines['right'].set_visible(True)

#adjustable idea precision values
ax_b_l = fig.add_axes(lower_bounds)
ax_b_l.spines['top'].set_visible(True)
ax_b_l.spines['right'].set_visible(True)

#adjustable idea precision values
ax_b_u = fig.add_axes(upper_bounds)
ax_b_u.spines['top'].set_visible(True)
ax_b_u.spines['right'].set_visible(True)


s_u = Slider(ax=ax_range, label='precision upper bound', valmin=min(upper_bounds), valmax=max(upper_bounds),facecolor='#cc7000')

s_c = Slider(ax=ax_c, label='graph connectedness', valmin=0, valmax=0.4, facecolor='#cc7000')

s_env_l = Slider(ax=ax_env_l, label='idea precision lower bound', valmin=min(lower_bounds), valmax=max(lower_bounds), facecolor='#cc7000')
s_env_u = Slider(ax=ax_env_u, label='idea precision upper bound', valmin=min(upper_bounds), valmax=max(upper_bounds), facecolor='#cc7000')
s_b_l = Slider(ax=ax_b_l, label='belief precision', valmin=min(lower_bounds), valmax=max(lower_bounds), facecolor='#cc7000')
s_b_u = Slider(ax=ax_b_u, label='belief precision range', valmin=min(upper_bounds), valmax=max(upper_bounds), facecolor='#cc7000')


r_0 = 0
c_0 = 0.4
env_l_0 = lower_bounds[0]
env_u_0 = upper_bounds[-1]
b_l_0 = lower_bounds[0]
b_u_0 = upper_bounds[-1]

y_axis = []
for precision_l in lower_bounds:
    ratio = get_ratio_from_parameters(c_0, precision_l, r_0, env_l_0, env_u_0, b_l_0, b_u_0)
    y_axis.append(ratio)
print(y_axis)

graph, = ax.plot(lower_bounds, y_axis, linewidth = 2.5)

# Update values
def update(val):
    s_u = s_u.val
    s_c = s_c.val
    s_env_l = s_env_l.val
    s_env_u = s_env_u.val
    s_b_l = s_b_l.val
    s_b_u = s_b_u.val
    new_data = []
    for l in lower_bounds:
        new_data.append(get_ratio_from_parameters(c_0, lower_bounds, r_0, env_l_0, env_u_0, b_l_0, b_u_0))

    graph.set_data(lower_bounds, new_data)
    fig.canvas.draw_idle()

s_u.on_changed(update)
s_c.on_changed(update)
s_Ef.on_changed(update)
s_env_l.on_changed(update)
s_env_u.on_changed(update)
s_b_l.on_changed(update)
s_b_u.on_changed(update)

ax.show()