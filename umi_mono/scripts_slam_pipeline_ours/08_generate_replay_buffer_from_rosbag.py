# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
import h5py
from tqdm import tqdm
from collections import defaultdict
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
    get_mirror_crop_slices
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()


# %%
@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-ir', '--in_res', type=str, default='960,540')
# @click.option('-of', '--out_fov', type=float, default=None)
@click.option('-cl', '--compression_level', type=int, default=99)
# @click.option('-nm', '--no_mirror', is_flag=True, default=False, help="Disable mirror observation by masking them out")
# @click.option('-ms', '--mirror_swap', is_flag=True, default=False)
# @click.option('-n', '--num_workers', type=int, default=None)
def main(input, output, in_res, out_res, compression_level):
        #  no_mirror, mirror_swap, num_workers):
    if os.path.isfile(output):
        if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
            pass
    
    out_res = tuple(int(x) for x in out_res.split(','))
    in_res = tuple(int(x) for x in in_res.split(','))

    # if num_workers is None:
    #     num_workers = multiprocessing.cpu_count()
    # cv2.setNumThreads(1)
            
    # fisheye_converter = None
    # if out_fov is not None:
    #     intr_path = pathlib.Path(os.path.expanduser(ipath)).absolute().joinpath(
    #         'calibration',
    #         'gopro_intrinsics_2_7k.json'
    #     )
    #     opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
    #     fisheye_converter = FisheyeRectConverter(
    #         **opencv_intr_dict,
    #         out_size=out_res,
    #         out_fov=out_fov
    #     )
        
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    # dump lowdim data to replay buffer
    # generate argumnet for videos
    # n_grippers = None
    # n_cameras = None
    # buffer_start = 0
    total_frames = 0
    episode_ends = []
    all_videos = set()
    # vid_args = list()
    resize_tf = get_image_transform(
    in_res=in_res,
    out_res=out_res
    )
    for h5_path in input:
        h5_path = pathlib.Path(h5_path).absolute()
        try:
            with h5py.File(h5_path, 'r') as f:
                obs = f['observation']
                action = obs['action'][()] if 'action' in obs else obs['qpos'][()]
                demo_start_pose = np.empty_like(action[..., :6])
                demo_start_pose[:] = action[0, :6]
                demo_end_pose = np.empty_like(action[..., :6])
                demo_end_pose[:] = action[-1, :6]
                wrist_imgs = obs['images']['wrist'][()]
                side_imgs = obs['images']['side'][()]
                n_frames = action.shape[0]
                wrist_resize_imgs = np.empty((n_frames, *out_res, 3), dtype=np.uint8)
                side_resize_imgs = np.empty((n_frames, *out_res, 3), dtype=np.uint8)
                for i in range(n_frames):
                    wrist_resize_imgs[i] = resize_tf(wrist_imgs[i])
                    side_resize_imgs[i] = resize_tf(side_imgs[i])

                episode_data = {
                    'robot0_eef_pos': action[..., :3].astype(np.float32),
                    'robot0_eef_rot_axis_angle': action[..., 3:6].astype(np.float32),
                    'robot0_gripper_width': action[..., 6].astype(np.float32),
                    'robot0_demo_start_pose': demo_start_pose.astype(np.float32),
                    'robot0_demo_end_pose': demo_end_pose.astype(np.float32),
                    'camera0_rgb': wrist_imgs,
                    'camera1_rgb': side_imgs,
                }
                out_replay_buffer.add_episode(data=episode_data, compressors=None)
                episode_ends.append(total_frames + n_frames)
                total_frames += n_frames
        except OSError as e:
            print(f"❌ 无法打开文件: {h5_path}，原因: {e}")
            continue



    # 再次遍历写入图片
    # frame_start = 0
    # for h5_path in hdf5_paths:
    #     with h5py.File(h5_path, 'r') as f:
    #         obs = f['observation']
    #         wrist_imgs = obs['images']['wrist'][()]
    #         side_imgs = obs['images']['side'][()]
    #         n_frames = wrist_imgs.shape[0]
    #         out_replay_buffer.data['camera0_rgb'][frame_start:frame_start+n_frames] = wrist_imgs
    #         out_replay_buffer.data['camera1_rgb'][frame_start:frame_start+n_frames] = side_imgs
    #         frame_start += n_frames

    # # 保存到 zarr zip
    # with zarr.ZipStore(output, mode='w') as zip_store:
    #     out_replay_buffer.save_to_store(store=zip_store)

    # def video_to_zarr(replay_buffer, mp4_path, tasks):
    #     pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
    #     tag_detection_results = pickle.load(open(pkl_path, 'rb'))

    #     tasks = sorted(tasks, key=lambda x: x['frame_start'])
    #     camera_idx = None
    #     for task in tasks:
    #         if camera_idx is None:
    #             camera_idx = task['camera_idx']
    #         else:
    #             assert camera_idx == task['camera_idx']
    #     name = f'camera{camera_idx}_rgb'
    #     img_array = replay_buffer.data[name]
        
    #     curr_task_idx = 0
        
    #     is_mirror = None
    #     if mirror_swap:
    #         ow, oh = out_res
    #         mirror_mask = np.ones((oh,ow,3),dtype=np.uint8)
    #         mirror_mask = draw_predefined_mask(
    #             mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
    #         is_mirror = (mirror_mask[...,0] == 0)
        
    #     with av.open(mp4_path) as container:
    #         in_stream = container.streams.video[0]
    #         # in_stream.thread_type = "AUTO"
    #         in_stream.thread_count = 1
    #         buffer_idx = 0
    #         for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
    #             if curr_task_idx >= len(tasks):
    #                 # all tasks done
    #                 break
                
    #             if frame_idx < tasks[curr_task_idx]['frame_start']:
    #                 # current task not started
    #                 continue
    #             elif frame_idx < tasks[curr_task_idx]['frame_end']:
    #                 if frame_idx == tasks[curr_task_idx]['frame_start']:
    #                     buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
    #                 # do current task
    #                 img = frame.to_ndarray(format='rgb24')

    #                 # inpaint tags
    #                 this_det = tag_detection_results[frame_idx]
    #                 all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
    #                 for corners in all_corners:
    #                     img = inpaint_tag(img, corners)
                        
    #                 # mask out gripper
    #                 img = draw_predefined_mask(img, color=(0,0,0), 
    #                     mirror=no_mirror, gripper=True, finger=False)
    #                 # resize
    #                 if fisheye_converter is None:
    #                     img = resize_tf(img)
    #                 else:
    #                     img = fisheye_converter.forward(img)
                        
    #                 # handle mirror swap
    #                 if mirror_swap:
    #                     img[is_mirror] = img[:,::-1,:][is_mirror]
                        
    #                 # compress image
    #                 img_array[buffer_idx] = img
    #                 buffer_idx += 1
                    
    #                 if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
    #                     # current task done, advance
    #                     curr_task_idx += 1
    #             else:
    #                 assert False
                    
    # with tqdm(total=len(vid_args)) as pbar:
    #     # one chunk per thread, therefore no synchronization needed
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         futures = set()
    #         for mp4_path, tasks in vid_args:
    #             if len(futures) >= num_workers:
    #                 # limit number of inflight tasks
    #                 completed, futures = concurrent.futures.wait(futures, 
    #                     return_when=concurrent.futures.FIRST_COMPLETED)
    #                 pbar.update(len(completed))

    #             futures.add(executor.submit(video_to_zarr, 
    #                 out_replay_buffer, mp4_path, tasks))

    #         completed, futures = concurrent.futures.wait(futures)
    #         pbar.update(len(completed))

    # print([x.result() for x in completed])

    # # dump to disk
    # print(f"Saving ReplayBuffer to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {len(all_videos)} videos used in total!")

# %%
if __name__ == "__main__":
    main()
