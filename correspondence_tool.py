import sys
import os
import wx
import numpy as np

import geometry

LEFT_SIDE  = -1
RIGHT_SIDE =  1
DRAG_RADIUS = 10

class CorrespondenceFrame(wx.Frame):
    MAX_IMAGE_SIZE = 800.

    def __init__(self, left_image_path, right_image_path, corrs_path='correspondences.txt', parent=None):
        self._left_image, self._left_bitmap, self._left_shape, self._left_scaling = \
            self.load_and_scale_image(left_image_path)
        self._right_image, self._right_bitmap, self._right_shape, self._right_scaling = \
            self.load_and_scale_image(right_image_path)

        height_diff = (self._right_shape[1] - self._left_shape[1]) / 2
        self._left_offset = (0, max(0, height_diff))
        self._right_offset = (int(1.1*self._left_shape[0]), max(0, -height_diff))

        self.clear_correspondences()

        self._corrs_path = corrs_path
        if os.path.exists(self._corrs_path):
            self.load_correspondences(self._corrs_path)

        frame_width = self._right_offset[0] + self._right_shape[0]
        frame_height = max(self._left_shape[1], self._right_shape[1]) + 25

        wx.Frame.__init__(self,
                          parent,
                          title='Correspondence Tool',
                          size=(frame_width, frame_height))

        self._panel = wx.Panel(self, wx.ID_ANY)
        self._panel.Bind(wx.EVT_PAINT, self.on_self_paint)
        self._panel.Bind(wx.EVT_LEFT_DOWN, self.on_self_left_down)
        self._panel.Bind(wx.EVT_LEFT_UP, self.on_self_left_up)
        self._panel.Bind(wx.EVT_MOTION, self.on_self_motion)
        self._panel.Bind(wx.EVT_KEY_DOWN, self.on_self_key_down)
        self.Bind(wx.EVT_CLOSE, self.on_self_close)

        self.Show(True)


    def load_and_scale_image(self, path):
        image = wx.Image(path)
        w = image.GetWidth()
        h = image.GetHeight()
        if max(w,h) > CorrespondenceFrame.MAX_IMAGE_SIZE:
            scaling = CorrespondenceFrame.MAX_IMAGE_SIZE / float(max(w,h))
            w = np.ceil(w*scaling)
            h = np.ceil(h*scaling)
            image = image.Scale(w, h, wx.IMAGE_QUALITY_HIGH)
        else:
            scaling = 1.

        bitmap = wx.BitmapFromImage(image)
        return image, bitmap, (w,h), scaling
            

    def locate_mouse(self, p):
        p = np.asarray(p)
        if geometry.in_bounds(p-self._left_offset, self._left_shape):
            return LEFT_SIDE, p-self._left_offset
        elif geometry.in_bounds(p-self._right_offset, self._right_shape):
            return RIGHT_SIDE, p-self._right_offset
        else:
            return None,None

    def clear_correspondences(self):
        self._left_points = []
        self._right_points = []
        self._drag_index = None
        self._drag_side = None
        self._drag_offset = None
        self._hover_point = None
        self._show_homography = False


    def load_correspondences(self, path):
        self._corrs_path = path
        try:
            X = np.atleast_2d(np.loadtxt(path))
            if X.ndim != 2 or X.shape[1] != 4:
                print '%s has invalid shape: %s' % (corrs_path, str(X.shape))
            else:
                self._left_points  = list(X[:,:2] * self._left_scaling)
                self._right_points = list(X[:,2:] * self._right_scaling)
        except ValueError as ex:
            print 'Failed to parse %s: %s' % (corrs_path, ex.message)


    def on_self_paint(self, event=None):
        dc = wx.PaintDC(self)
        dc.Clear()

        dc.DrawBitmapPoint(self._left_bitmap, self._left_offset)
        dc.DrawBitmapPoint(self._right_bitmap, self._right_offset)

        HOVER_ALPHA = 80
        LINE_ALPHA = 140
        HOMOGRAPHY_ALPHA = 160

        # Draw the correspondence arcs
        dc.SetPen(wx.Pen((255, 0, 0, LINE_ALPHA), 2))
        for i in range(min(len(self._left_points), len(self._right_points))):
            lxy = np.add(self._left_offset,  self._left_points[i])
            rxy = np.add(self._right_offset, self._right_points[i])
            dc.DrawLine(lxy[0], lxy[1], rxy[0], rxy[1])

        # Draw the points
        dc.SetPen(wx.Pen((255, 0, 0, LINE_ALPHA), 6))
        for lx,ly in self._left_points:
            dc.DrawPoint(self._left_offset[0]+lx, self._left_offset[1]+ly)
        for rx,ry in self._right_points:
            dc.DrawPoint(self._right_offset[0]+rx, self._right_offset[1]+ry)


        # Find the closest point
        if self._hover_point is not None:
            side,imxy = self.locate_mouse(self._hover_point)
            if side is not None:
                L = self._left_points if side == LEFT_SIDE else self._right_points
                hover_index,p = geometry.find_point_within_distance(imxy, L, DRAG_RADIUS)

        # Draw the hover arc
        grey = wx.Colour(0, 0, 0, HOVER_ALPHA)  # semi-transparent
        dc.SetPen(wx.Pen(grey, 2))
        if self._hover_point is not None:
            side,imxy = self.locate_mouse(self._hover_point)
            if side is not None:
                offset = self._right_offset if side == LEFT_SIDE else self._left_offset
                Lcur = self._left_points if side == LEFT_SIDE else self._right_points
                Lopp = self._right_points if side == LEFT_SIDE else self._left_points
                if len(Lopp) > len(Lcur):
                    hx,hy = np.add(Lopp[len(Lcur)], offset)
                    dc.DrawLine(hx, hy, self._hover_point[0], self._hover_point[1])

        # Estimate a homography
        if self._show_homography:
            try:
                n = min(len(self._left_points), len(self._right_points))
                if n >= 4:
                    # Estimate the homography
                    H = geometry.fit_homography(self._left_points[:n],
                                                self._right_points[:n])
                
                    # Project the left image bounds through the homography
                    quad = [(0., 0.),
                            (self._left_shape[0], 0.),
                            self._left_shape,
                            (0., self._left_shape[1])]
                    Hquad = geometry.prdot(H, quad)

                    # Draw the quad
                    dc.SetPen(wx.Pen(wx.Colour(0, 0, 255, HOMOGRAPHY_ALPHA), 4.))
                    for i in range(4):
                        p0 = self._right_offset + Hquad[i]
                        p1 = self._right_offset + Hquad[(i+1)%4]
                        dc.DrawLine(p0[0], p0[1], p1[0], p1[1])

            except np.linalg.LinAlgError:
                print 'Failed to fit homography'
            

    def on_self_left_down(self, event=None):
        self._hover_point = None
        xy = (event.X, event.Y)
        side,imxy = self.locate_mouse(xy)
        if side is not None:
            L = self._left_points if side == LEFT_SIDE else self._right_points
            i,p = geometry.find_point_within_distance(imxy, L, DRAG_RADIUS)
            if i is None:
                L.append(imxy)
            else:
                self._drag_side = side
                self._drag_index = i
                self._drag_offset = np.subtract(p, xy)

        self.Refresh()

    def on_self_left_up(self, event=None):
        self._hover_point = (event.X, event.Y)
        self._drag_side = None
        self._drag_index = None
        self._drag_offset = None
        self.Refresh()

    def on_self_motion(self, event=None):
        if self._drag_index is None:
            self._hover_point = (event.X, event.Y)

        else:
            if self._drag_side == LEFT_SIDE:
                L = self._left_points
                shape = self._left_shape
            else:
                L = self._right_points
                shape = self._right_shape

            pnew = np.add((event.X,event.Y), self._drag_offset)
            pnew = np.clip(pnew, (0,0), shape)
            L[ self._drag_index ] = pnew

        self.Refresh()

    def on_self_key_down(self, event=None):
        if event.KeyCode == wx.WXK_ESCAPE or event.KeyCode == ord('Q'):
            self.Close()
        elif event.KeyCode == ord('S') and event.CmdDown():
            self.save_correspondences()
        elif event.KeyCode == ord('H'):
            self._show_homography = ~self._show_homography
            self.Refresh()
        elif event.KeyCode == ord('C'):
            self.clear_correspondences()
            self.Refresh()

    def on_self_close(self, event=None):
        self.save_correspondences()
        event.Skip()        # make sure the window closes after we return

    def save_correspondences(self):
        n = min(len(self._left_points), len(self._right_points))
        if n > 0:
            left_scaled = np.asarray(self._left_points[:n]) / self._left_scaling
            right_scaled = np.asarray(self._right_points[:n]) / self._right_scaling
            print 'Wrote %d correspondences to %s' % (n, self._corrs_path)
            np.savetxt(self._corrs_path,
                       np.hstack((left_scaled, right_scaled)),
                       fmt='%10.2f')

    @classmethod
    def run(cls, *args, **kwargs):
        app = wx.App(False)
        frame = cls(*args, **kwargs)
        app.MainLoop()

if __name__ == '__main__':
    if len(sys.argv) not in (3,4):
        print 'Usage: python correspondence_tool.py <IMAGE1> <IMAGE2> <OUTPUT_CORRESPONDENCES>'
    else:
        CorrespondenceFrame.run(*sys.argv[1:])
