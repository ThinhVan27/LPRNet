import shutil
import os
import random


def move_track(dst_dir, track_paths):
    for track in track_paths:
        shutil.move(src=track, dst=dst_dir)

if __name__ == "__main__":
    src_dir = "train"
    dst_dir = "test"
    for s in sorted(os.listdir(src_dir)):
        s_path = os.path.join(src_dir, s)
        for typ in sorted(os.listdir(s_path)):
            src_typ_path = os.path.join(s_path, typ)
            dst_typ_path = os.path.join(dst_dir, s, typ)
            print(f"[INFO] Length of {src_typ_path}: {len(os.listdir(src_typ_path))}")
            print(f"[INFO] Length of {dst_typ_path}: {len(os.listdir(dst_typ_path))}")
            # track_dirs = sorted(os.listdir(src_typ_path))
            # num_tracks = len(track_dirs)
            # indices = random.sample(range(num_tracks), int(0.8 *num_tracks))
            # track_paths = list(map(lambda x : os.path.join(src_typ_path, x), [track_dirs[i] for i in indices]))
            # move_track(dst_dir=dst_typ_path, track_paths=track_paths)
            
            