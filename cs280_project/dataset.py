import torch
import numpy as np


class QuickDrawDataset(torch.utils.data.Dataset):
    """
    QuickDraw数据集加载器
    直接返回整个视频序列和类别标签
    """
    def __init__(self, data_path, split='train'):
        """
        Args:
            data_path: npz文件路径
            split: 数据集划分，'train', 'valid', 或 'test'
        """
        # 加载数据
        data = np.load(data_path, allow_pickle=True)
        self.images = data[f'{split}_images']
        self.labels = data[f'{split}_labels']
        
        # 创建视频索引列表
        self.valid_indices = []
        cnt = 0
        zipped = zip(self.images, self.labels)
        for video_idx, video in enumerate(self.images):
        # for video_idx, (video, label) in enumerate(zipped):
            # 只要视频有至少一帧，就加入索引
            # if label[0] != 1 and label[1] != 1:
            #     continue
            cnt += 1
            if len(video) > 0:
                self.valid_indices.append(video_idx)
        # print(cnt)
            # if cnt >= 1:
            #     break
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """返回整个视频序列和类别标签"""
        # 获取视频索引
        video_idx = self.valid_indices[idx]
        
        # 获取整个视频序列
        video = self.images[video_idx]
        
        # 转换为张量
        video_tensor = torch.from_numpy(video).float()  # [frames, H, W]
        
        # 获取类别标签 (确保为int64类型)
        label = torch.tensor(self.labels[video_idx], dtype=torch.long)
        # change to one-hot
        label = torch.nn.functional.one_hot(label, num_classes=10).float()
        
        # 返回视频序列和标签
        return video_tensor, label

# import torch
# import numpy as np


# class QuickDrawDataset(torch.utils.data.Dataset):
#     """
#     QuickDraw数据集加载器
#     支持第一个chunk没有历史帧的情况
#     """
#     def __init__(self, data_path, split='train', time_step=12, predict_frames=12, include_first_chunk=True):
#         """
#         Args:
#             data_path: npz文件路径
#             split: 数据集划分，'train', 'valid', 或 'test'
#             time_step: 上下文帧数量（历史帧数）
#             predict_frames: 预测帧数量
#             include_first_chunk: 是否包含没有历史帧的第一个chunk
#         """
#         # 加载数据
#         data = np.load(data_path, allow_pickle=True)
#         self.images = data[f'{split}_images']
#         self.labels = data[f'{split}_labels']
#         self.time_step = time_step
#         self.predict_frames = predict_frames
#         self.include_first_chunk = include_first_chunk
        
#         # 计算有效索引
#         self.valid_indices = []
        
#         for video_idx, video in enumerate(self.images):
#             # 如果包含第一个chunk（特殊情况，起始帧索引为-1表示）
#             if include_first_chunk and video.shape[0] >= predict_frames:
#                 self.valid_indices.append((video_idx, -1))  # 特殊标记-1表示第一个chunk
            
#             # 常规情况：有足够的历史帧
#             min_frames = time_step + predict_frames
#             if video.shape[0] >= min_frames:
#                 num_samples = video.shape[0] - min_frames + 1
                
#                 for start_frame in range(num_samples):
#                     self.valid_indices.append((video_idx, start_frame))
    
#     def __len__(self):
#         return len(self.valid_indices)
    
#     def __getitem__(self, idx):
#         """返回单个样本：目标帧、历史帧和类别标签"""
#         # 获取视频和起始帧索引
#         video_idx, start_frame = self.valid_indices[idx]
#         video = self.images[video_idx]
        
#         # 特殊情况：第一个chunk没有历史帧
#         if start_frame == -1:
#             # 目标帧是前predict_frames帧
#             target_frames = video[:self.predict_frames]
            
#             # 创建全零的历史帧
#             # 注意：这里假设图像大小为video[0].shape
#             frame_shape = video[0].shape
#             history_frames = torch.zeros((self.time_step, *frame_shape), dtype=torch.float32)
            
#             # 转换目标帧为张量
#             target_frames = torch.from_numpy(target_frames).float()
#         else:
#             # 常规情况：有历史帧
#             history_frames = video[start_frame:start_frame+self.time_step]
#             target_frames = video[start_frame+self.time_step:start_frame+self.time_step+self.predict_frames]
            
#             # 转换为张量
#             history_frames = torch.from_numpy(history_frames).float()
#             target_frames = torch.from_numpy(target_frames).float()
        
#         # 获取类别标签 (确保为int64类型)
#         label = torch.tensor(self.labels[video_idx], dtype=torch.long)
        
#         # 返回目标帧、历史帧和标签
#         return target_frames, history_frames, label

# # import torch
# # import numpy as np


# # class QuickDrawDataset(torch.utils.data.Dataset):
# #     """
# #     QuickDraw数据集加载器
# #     每个视频有60帧，形状为(60, 256, 256)
# #     每次返回连续的24帧，分成两组12帧
# #     """
# #     def __init__(self, data_path, split='train', time_step=12, predict_frames=12):
# #         """
# #         Args:
# #             data_path: npz文件路径
# #             split: 数据集划分，'train', 'valid', 或 'test'
# #             time_step: 上下文帧数量（历史帧数）
# #             predict_frames: 预测帧数量
# #         """
# #         # 加载数据
# #         data = np.load(data_path, allow_pickle=True)
# #         self.images = data[f'{split}_images']
# #         self.labels = data[f'{split}_labels']
# #         self.time_step = time_step
# #         self.predict_frames = predict_frames
        
# #         # 计算有效索引 (每个视频可以提取的帧组)
# #         self.valid_indices = []
# #         for video_idx, video in enumerate(self.images):
# #             # 确保视频长度足够
# #             min_frames = time_step + predict_frames
# #             if video.shape[0] >= min_frames:
# #                 # 计算可以提取的样本数量
# #                 num_samples = video.shape[0] - min_frames + 1
                
# #                 for start_frame in range(num_samples):
# #                     self.valid_indices.append((video_idx, start_frame))
    
# #     def __len__(self):
# #         return len(self.valid_indices)
    
# #     def __getitem__(self, idx):
# #         """返回单个样本：目标帧、历史帧和类别标签"""
# #         # 获取视频和起始帧索引
# #         video_idx, start_frame = self.valid_indices[idx]
# #         video = self.images[video_idx]
        
# #         # 获取历史帧和目标帧
# #         history_frames = video[start_frame:start_frame+self.time_step]
# #         target_frames = video[start_frame+self.time_step:start_frame+self.time_step+self.predict_frames]
        
# #         # 转换为张量
# #         history_frames = torch.from_numpy(history_frames).float()  # [time_step, H, W]
# #         target_frames = torch.from_numpy(target_frames).float()    # [predict_frames, H, W]
        
# #         # 获取类别标签 (确保为int64类型)
# #         label = torch.tensor(self.labels[video_idx], dtype=torch.long)
        
# #         # 返回目标帧、历史帧和标签
# #         return target_frames, history_frames, label


# # # import torch
# # # import numpy as np


# # # class QuickDrawDataset(torch.utils.data.Dataset):
# # #     """
# # #     QuickDraw数据集加载器
# # #     每个视频有60帧，形状为(60, 256, 256, 3)
# # #     每次返回两组12帧
# # #     """
# # #     def __init__(self, data_path, split='train', time_step=5, predict_frames=1, batch_size=1):
# # #         """
# # #         Args:
# # #             data_path: npz文件路径
# # #             split: 数据集划分，'train', 'valid', 或 'test'
# # #             time_step: 上下文帧数量
# # #             predict_frames: 预测帧数量
# # #             convert_to_gray: 是否转换为灰度图
# # #             batch_size: 每次返回的批次大小
# # #         """
# # #         # 加载数据
# # #         data = np.load(data_path, allow_pickle=True)
# # #         self.images = data[f'{split}_images']
# # #         self.labels = data[f'{split}_labels']
# # #         self.time_step = time_step
# # #         self.predict_frames = predict_frames
# # #         self.batch_size = batch_size
        
# # #         # 计算有效索引
# # #         self.valid_indices = []
# # #         for video_idx, video in enumerate(self.images):
# # #             # 确保视频有至少24帧 (2组12帧)
# # #             if video.shape[0] >= 24:
# # #                 # 计算可以提取的帧组数量
# # #                 num_frame_groups = (video.shape[0] - 12) // 12  # 每12帧一组，至少需要2组
                
# # #                 for group_idx in range(num_frame_groups):
# # #                     # 每组的起始帧索引
# # #                     start_frame = group_idx * 12
# # #                     self.valid_indices.append((video_idx, start_frame))
    
# # #     def __len__(self):
# # #         return len(self.valid_indices) // self.batch_size
    
# # #     def __getitem__(self, idx):
# # #         # 创建批次容器
# # #         batch_first_groups = []
# # #         batch_second_groups = []
# # #         batch_labels = []
        
# # #         # 获取一个批次的数据
# # #         batch_start = idx * self.batch_size
# # #         batch_end = min(batch_start + self.batch_size, len(self.valid_indices))
        
# # #         for i in range(batch_start, batch_end):
# # #             video_idx, start_frame = self.valid_indices[i]
# # #             video = self.images[video_idx]
            
# # #             # 获取两组连续的12帧
# # #             first_group = video[start_frame:start_frame+12]
# # #             second_group = video[start_frame+12:start_frame+24]
            
# # #             # 转换为张量
# # #             first_group = torch.from_numpy(first_group).float()
# # #             second_group = torch.from_numpy(second_group).float()
            
# # #             # 添加到批次
# # #             batch_first_groups.append(first_group)
# # #             batch_second_groups.append(second_group)
            
# # #             # 获取类别标签 - 确保为int64类型
# # #             label = torch.tensor(self.labels[video_idx], dtype=torch.long)  # torch.long 是 int64
# # #             batch_labels.append(label)
        
# # #         # 如果批次不满，使用最后一个样本填充
# # #         while len(batch_first_groups) < self.batch_size:
# # #             batch_first_groups.append(batch_first_groups[-1])
# # #             batch_second_groups.append(batch_second_groups[-1])
# # #             batch_labels.append(batch_labels[-1])
        
# # #         # 将列表堆叠为张量
# # #         batch_first = torch.stack(batch_first_groups)    # [batch_size, 12, H, W]
# # #         batch_second = torch.stack(batch_second_groups)  # [batch_size, 12, H, W]
# # #         batch_labels = torch.stack(batch_labels)         # [batch_size]
        
# # #         # 确保batch_labels是(N,)形状的int64张量
# # #         assert batch_labels.shape == (self.batch_size,)
# # #         assert batch_labels.dtype == torch.int64
        
# # #         # 返回两组帧和标签
# # #         # batch_second作为image (目标帧)，batch_first作为x_p (历史帧)
# # #         return batch_second, batch_first, batch_labels
# # #     # def __getitem__(self, idx):
# # #     #     # 创建批次容器
# # #     #     batch_first_groups = []
# # #     #     batch_second_groups = []
# # #     #     batch_labels = []
        
# # #     #     # 获取一个批次的数据
# # #     #     batch_start = idx * self.batch_size
# # #     #     batch_end = min(batch_start + self.batch_size, len(self.valid_indices))
        
# # #     #     for i in range(batch_start, batch_end):
# # #     #         video_idx, start_frame = self.valid_indices[i]
# # #     #         video = self.images[video_idx]
            
# # #     #         # 获取两组连续的12帧
# # #     #         first_group = video[start_frame:start_frame+12]
# # #     #         second_group = video[start_frame+12:start_frame+24]
            
# # #     #         # 转换为张量
# # #     #         first_group = torch.from_numpy(first_group).float()
# # #     #         second_group = torch.from_numpy(second_group).float()
            
# # #     #         # 添加到批次
# # #     #         batch_first_groups.append(first_group)
# # #     #         batch_second_groups.append(second_group)
            
# # #     #         # 获取类别标签
# # #     #         p = torch.tensor(self.labels[video_idx], dtype=torch.long)
# # #     #         batch_labels.append(p)
        
# # #     #     # 如果批次不满，使用最后一个样本填充
# # #     #     while len(batch_first_groups) < self.batch_size:
# # #     #         batch_first_groups.append(batch_first_groups[-1])
# # #     #         batch_second_groups.append(batch_second_groups[-1])
# # #     #         batch_labels.append(batch_labels[-1])
        
# # #     #     # 将列表堆叠为张量
# # #     #     batch_first = torch.stack(batch_first_groups)    # [batch_size, 12, H, W]
# # #     #     batch_second = torch.stack(batch_second_groups)  # [batch_size, 12, H, W]
# # #     #     batch_labels = torch.stack(batch_labels)         # [batch_size]
        
# # #     #     # 返回两组帧和标签
# # #     #     # batch_first作为x_p (历史帧)，batch_second作为image (目标帧)
# # #     #     return batch_second, batch_first, batch_labels