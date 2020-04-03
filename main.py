import typing
import argparse
import cv2
import numpy as np
from matplotlib.backend_bases import Event, KeyEvent
from matplotlib.image import AxesImage
from matplotlib.collections import PathCollection
from matplotlib.quiver import Quiver
import matplotlib.pyplot as plt

from typing import Any, List, Union

PATH_OUTPUT = 'out.png'


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input image path')
    parser.add_argument('-r', '--reference', type=str,
                        help='reference image path')
    parser.add_argument('-s', '--points_start', type=str,
                        help='input image keypoint .npy file (optional)')
    parser.add_argument('-e', '--points_end', type=str,
                        help='reference image keypoint .npy file (optional)')
    parser.add_argument('-o', '--output', type=str,
                        help='output image path (default="out.png")')
    args = parser.parse_args()
    assert args.input is not None
    assert args.reference is not None
    return args


def plot_df(axis: plt.Axes, df: np.ndarray, scale: int = 8) -> Quiver:
    return axis.quiver(df[::-scale, ::scale, 0], -df[::-scale, ::scale, 1],
               units='xy', scale=(1/scale), angles='xy')

def plot_image(axis: plt.Axes, image: np.ndarray) -> AxesImage:
    return axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def get_grid(width: int, height: int) -> np.ndarray:
    xs = np.stack([np.arange(0, width) for _ in range(height)]).astype(np.float32)
    ys = np.stack([np.arange(0, height) for _ in range(width)]).astype(np.float32).T
    return np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], 2)


def calc_df(
        width: int,
        height: int,
        ptrs_from: List[np.ndarray],
        ptrs_to: List[np.ndarray]):

    num_disps = min(len(ptrs_from), len(ptrs_to))
    grid = get_grid(width, height)

    tps = cv2.createThinPlateSplineShapeTransformer()
    arr_src = np.expand_dims(np.array(ptrs_from), 0)
    arr_dst = np.expand_dims(np.array(ptrs_to), 0)
    matches = [cv2.DMatch(i, i, 0) for i in range(num_disps)]
    tps.estimateTransformation(arr_src, arr_dst, matches)
    grid_warped = tps.applyTransformation(
        grid.reshape(1, -1, 2))[1].reshape(height, width, 2)

    return grid_warped

def save_result_warp(
        path_output: str,
        image: np.ndarray,
        df_forward: np.ndarray,
        df_backward: np.ndarray,
        ptrs_start: List[np.ndarray],
        ptrs_end: List[np.ndarray]):

    postfix_df_forward = '.df.forward.npy'
    postfix_df_backward = '.df.backward.npy'
    postfix_ptrs_start = '.ptrs.start.npy'
    postfix_ptrs_end = '.ptrs.end.npy'

    assert image.shape[:2] == df_forward.shape[:2]
    assert image.shape[:2] == df_backward.shape[:2]
    cv2.imwrite(path_output, image)
    np.save(path_output+postfix_df_forward, df_forward)
    np.save(path_output+postfix_df_backward, df_backward)
    np.save(path_output+postfix_ptrs_start, np.stack(ptrs_start))
    np.save(path_output+postfix_ptrs_end, np.stack(ptrs_end))

class WarpingWindow:
    def __init__(self,
                 image_in: np.ndarray,
                 image_ref: np.ndarray,
                 path_output: str):
        self.image_in = image_in
        self.image_warp = image_in
        self.image_ref = image_ref
        self.path_output = path_output

        self.ptrs_start_control = []
        self.ptr_start_selected = None
        self.ptrs_end_control = []
        self.ptr_end_selected = None

        self.fig_root, self.axes = plt.subplots(1, 3)
        self.fig_root.canvas.mpl_connect('button_press_event', self._callback_press)
        self.fig_root.canvas.mpl_connect('button_release_event', self._callback_release)
        self.fig_root.canvas.mpl_connect('motion_notify_event', self._callback_motion)
        self.fig_root.canvas.mpl_connect('key_press_event', self._callback_key)
        self.axesimage_warp = plot_image(self.axes[1], self.image_in)
        self.scatter_in = self.axes[0].scatter([], [], s=5, c='#0000ff')
        self.scatter_warp = self.axes[1].scatter([], [], s=5, c='#ff0000')
        self.scatter_out = self.axes[2].scatter([], [], s=5, c='#ff0000')

        plot_image(self.axes[0], self.image_in)
        plot_image(self.axes[2], self.image_ref)

    def get_df_forward(self) -> np.ndarray:
        return calc_df(
            self.image_in.shape[1], self.image_in.shape[0],
            self.ptrs_start_control, self.ptrs_end_control)

    def get_df_backward(self) -> np.ndarray:
        return calc_df(
            self.image_in.shape[1], self.image_in.shape[0],
            self.ptrs_end_control, self.ptrs_start_control)

    def _redraw_warp(self) -> None:
        num_disps = min(len(self.ptrs_start_control), len(self.ptrs_end_control))

        if num_disps > 2:
            grid_warped = self.get_df_backward()
            self.image_warp = cv2.remap(
                self.image_in, grid_warped[:, :, 0], grid_warped[:, :, 1], cv2.INTER_LINEAR)

            self.axesimage_warp.set_data(cv2.cvtColor(self.image_warp, cv2.COLOR_BGR2RGB))

    def _update_points(self):
        def update_scatter(scatter: PathCollection, ptrs: List[np.ndarray]):
            scatter.set_offsets(ptrs)

        if self.ptrs_start_control:
            update_scatter(self.scatter_in, self.ptrs_start_control)
        if self.ptrs_end_control:
            update_scatter(self.scatter_out, self.ptrs_end_control)
            update_scatter(self.scatter_warp, self.ptrs_end_control)
        self.fig_root.canvas.draw()

    @staticmethod
    def _event2point(event: Event) -> np.ndarray:
        return np.array([event.xdata, event.ydata], dtype=np.float32)

    @staticmethod
    def _create_point(event: Event, ptrs: List[np.ndarray]) -> None:
        pt = WarpingWindow._event2point(event)
        print(pt)
        ptrs.append(pt)

    @staticmethod
    def _select_point(event: Event, ptrs: List[np.ndarray]) -> np.ndarray:
        assert ptrs
        return sorted(ptrs, key=lambda x: np.linalg.norm(x - WarpingWindow._event2point(event)))[0]

    @staticmethod
    def _move_point(event: Event, ptr: Union[np.ndarray, None]) -> None:
        if ptr is not None:
            ptr[0] = event.xdata
            ptr[1] = event.ydata
    
    def _callback_sub0(self, event: Event) -> None:
        if event.button == 1:  # left mouse -> create
            WarpingWindow._create_point(event, self.ptrs_start_control)
        elif event.button == 3: # right mouse -> select
            self.ptr_start_selected = WarpingWindow._select_point(
                event, self.ptrs_start_control)

    def _callback_sub1(self, event: Event) -> None:
        pass

    def _callback_sub2(self, event: Event) -> None:
        if event.button == 1:  # left mouse -> create
            WarpingWindow._create_point(event, self.ptrs_end_control)
        elif event.button == 3: # right mouse -> select
            self.ptr_start_selected = WarpingWindow._select_point(
                event, self.ptrs_end_control)

    def _callback_press(self, event: Event) -> None:
        if self.axes[0] == event.inaxes:
            self._callback_sub0(event)
        elif self.axes[1] == event.inaxes:
            self._callback_sub1(event)
        elif self.axes[2] == event.inaxes:
            self._callback_sub2(event)

        WarpingWindow._move_point(event, self.ptr_start_selected)
        WarpingWindow._move_point(event, self.ptr_end_selected)
        self._update_points()
        self.fig_root.canvas.draw()

    def _callback_release(self, _: Event) -> None:
        if self.ptr_start_selected is not None:
            self.ptr_start_selected = None
        if self.ptr_end_selected is not None:
            self.ptr_end_selected = None

        self._redraw_warp()
        self._update_points()
        self.fig_root.canvas.draw()

    def _callback_motion(self, event: Event) -> None:
        WarpingWindow._move_point(event, self.ptr_start_selected)
        WarpingWindow._move_point(event, self.ptr_end_selected)
        self._update_points()
        self.fig_root.canvas.draw()

    def _callback_key(self, event: KeyEvent) -> None:
        num_disps = min(len(self.ptrs_start_control), len(self.ptrs_end_control))
        if event.key == 'r':
            if num_disps > 2:
                save_result_warp(
                    self.path_output, self.image_warp,
                    self.get_df_forward(), self.get_df_backward(),
                    self.ptrs_start_control, self.ptrs_end_control)
                print('Saved to {}'.format(self.path_output))
            else:
                print('Put more than two points to save')
    
    def update(self) -> None:
        self._redraw_warp()
        self._update_points()
        self.fig_root.canvas.draw()

    def start_loop(self) -> None:
        plt.show()

def unroll_ndarray(arr: np.ndarray):
    return [arr[i] for i in range(arr.shape[0])]

def main():
    args = get_args()
    path_output = args.output if args.output else PATH_OUTPUT
    image_in = cv2.imread(args.input)
    image_ref = cv2.imread(args.reference)

    win = WarpingWindow(image_in, image_ref, path_output)
    if args.points_start:
        win.ptrs_start_control = unroll_ndarray(np.load(args.points_start))
    if args.points_end:
        win.ptrs_end_control = unroll_ndarray(np.load(args.points_end))
    win.update()
    win.start_loop()


if __name__ == '__main__':
    main()
