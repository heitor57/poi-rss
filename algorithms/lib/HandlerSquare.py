from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

class HandlerSquare(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        
        # center = xdescent + 0.5 * (width - height), ydescent
        center = xdescent, ydescent
        width, height = height, height
        p = mpatches.Rectangle(xy=center, width=width,
                               height=height, angle=0.0)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
