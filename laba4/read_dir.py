import struct
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


PATH = '27_08_2024ADC_rawData'


class Frame:
    def __init__(self, data, stop_point):
        self.data = data
        self.stop_point = stop_point
        self.timestamp = None
        self.reserved = None


class Bin:
    def __init__(self, lvl):
        self.lvl = lvl[0]
        self.last = lvl[1]
        self.side = int
        self.mode = int
        self.frame_count = int
        self.reserved = None
        self.frames = []

    def file_header(self, file):
        header = file.read(256)
        side, mode, frame_count = struct.unpack('3B', header[:3])
        return [side, mode, frame_count]

    def file_frames(self, file):
        with open(f'{file}', 'rb') as fid:
            self.side, self.mode, self.frame_count = self.file_header(fid)
            for _ in range(self.frame_count):
                stop_point = struct.unpack('H', fid.read(2))[0]
                timestamp = struct.unpack('I', fid.read(4))[0]  # not working
                reserved = np.fromfile(fid, dtype=np.uint16, count=5)  # for future
                rawdata = np.fromfile(fid, dtype=np.uint16, count=8192).reshape(1024, 8)
                ch18 = np.zeros((1024, 8), dtype=rawdata.dtype)
                for ch in range(8):
                    ch18[:, ch] = np.roll(rawdata[:, ch], stop_point)
                self.frames.append(ch18)
        return self.frames


class rawData:
    def __init__(self, path: str, calibrate=False):
        self.file_dir = Path(path).glob('*.bin')
        self.bins = []
        self.lvls = []
        if "ADC" in path:
            self.date = path.split("ADC")[0]
        else:
            self.date = "None"
        self.calibrate_flag = calibrate

    def file_lvl(self, file):
        filename = file.__str__().split("/")[1]
        if not self.calibrate_flag:
            if "lvl" in filename:
                lvl = [float(filename.split("_lvl")[0])]
            else:
                lvl = [float(filename.split("_side")[0])]
        else:
            if "lvl" in filename:
                lvl = [(filename.split("_lvl")[0])]
            else:
                lvl = [(filename.split("_side")[0])]

        if (filename.split(".bin")[0]).split("data")[1]:
            lvl.append(True)
        else:
            lvl.append(False)

        return lvl

    def get_bin_by_lvl(self, lvl, last=False):
        for k in range(len(self.bins)):
            if self.bins[k].lvl == lvl:
                if last == self.bins[k].last:
                    return self.bins[k]
                else:
                    continue
        print("NO LVL SUCH AS INPUT")
        return None

    def read_directory(self):
        headers_info = []

        for file in self.file_dir:
            bin_file = Bin(self.file_lvl(file))
            bin_file.file_frames(file)
            self.bins.append(bin_file)
            self.lvls.append(bin_file.lvl)

            headers_info.append([[bin_file.lvl, bin_file.last], [bin_file.side, bin_file.mode, bin_file.frame_count]])

        return headers_info

    def hist_bin_by_lvl_frame_channel(self, lvl: float, frame: int, channel: int, last=False):
        bin_file = self.get_bin_by_lvl(lvl, last)
        if bin_file:
            plt.hist(np.array(bin_file.frames[frame])[:, channel],
                     edgecolor="cornflowerblue",
                     bins=1024,
                     density=True)
            plt.title(f"lvl: {lvl}, frame: {frame}, channel: {channel + 1}")
            plt.show()
        else:
            print("input data wrong format")

    def plot_bin_by_lvl_frame_channel(self, lvl, frame: int, channel: int, last=False):
        bin_file = self.get_bin_by_lvl(lvl, last)
        if bin_file:
            plt.plot(np.array(bin_file.frames[frame])[:, channel],
                     color="royalblue")
            plt.title(f"lvl: {lvl}, frame: {frame}, channel: {channel+1}")
            plt.show()
        else:
            print("input data wrong format")

    def hist_bin_by_lvl_frame_all_bins(self, lvl: float, frame: int, last=False):
        bin_file = self.get_bin_by_lvl(lvl, last)
        if bin_file:
            for channel in range(0, 8):
                plt.subplot(2, 4, channel + 1)
                plt.hist(np.array(bin_file.frames[frame])[:, channel],
                         edgecolor="cornflowerblue",
                         bins=1024,
                         density=True)
                plt.title(f"{lvl, frame, channel + 1}")
            plt.show()
        else:
            print("input data wrong format")

    def plot_bin_by_lvl_frame_all_bins(self, lvl, frame: int, flag=True, last=False):
        bin_file = self.get_bin_by_lvl(lvl, last)
        if bin_file:
            if flag:
                for channel in range(0, 8):
                    plt.subplot(2, 4, channel + 1)
                    plt.plot(np.array(bin_file.frames[frame])[:, channel],
                             color="royalblue")
                    plt.title(f"{lvl, frame, channel + 1}")
                plt.show()
            else:
                plt.figure(figsize=(10, 6))
                colors = ["cyan", "deepskyblue", "teal", "darkslateblue",
                          "midnightblue", "indigo", "slategray", "turquoise"]
                colors.reverse()
                # colors = ["#EF476F", "#F78C6B", "#FFD166", "#83D483", "#06D6A0", "#0CB0A9", "#118AB2", "#073B4C"]
                for channel in range(0, 8):
                    plt.plot(np.array(self.get_bin_by_lvl(lvl).frames[frame])[:, channel],
                             color=colors[channel], label=f"channel {channel + 1}", alpha=0.8)
                plt.title(f"{lvl, frame}")
                plt.legend()
                plt.savefig(f"Исходные данные")
        else:
            print("input data wrong format")

rawData_instance = rawData(PATH)
rawData_instance.read_directory()
