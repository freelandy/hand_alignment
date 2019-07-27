# coding:utf-8
import enhance


n_train_samples = 120 # 166
n_test_samples = 15
n_points = 21
train_data_path = '.\\data\\trainset'
test_data_path = '.\\data\\testset'

n_cascades = 5
canvas_size = (400, 400)
scale_factor = 1.0  # shoudl be larger than or equal to 1.0
desc_scale = [0.16] * n_cascades
desc_cell_size = 16
alpha = [0.005] * n_cascades

# for evaluating alignment error
pts_eval = range(n_points)  # [1:21]
keypoint_1_idx = (8, 9)  # the point between forefinger and middle finger
keypoint_2_idx = (16, 17)  # the point between middle finger and ring finger

# for roi clipping
roi_size = (128, 128)

# for enhancement
f = enhance.generate_filter(15, 7.25, 0.03)

verbose = True
