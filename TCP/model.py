from collections import deque
import numpy as np
import torch 
from torch import nn
from TCP.resnet import *
import math


class PIDController(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative
	

# PositionalEncoding 클래스 정의
class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super(PositionalEncoding, self).__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 차원에 sin 적용
		pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 차원에 cos 적용
		pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
		self.register_buffer('pe', pe)

	def forward(self, x):
		# x: (batch, seq_len, d_model)
		return x + self.pe[:, :x.size(1), :]

	# TransformerModel 수정된 코드
class TransformerModel(nn.Module):
	def __init__(self, config):
		super(TransformerModel, self).__init__()
		self.config = config
		self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
		self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
		self.input_dim = 512 # 512 바꿔야함!!!!!!!!!!!!!!!!!!!!!!!!
		self.output_dim = 4 # 8 바꿔야함!!!!!!!!!!!!!!!!!!!!!
		self.d_model = 512 # 원래 512
		self.nhead = 8
		self.num_layers = 6
		self.dim_ff = 2048

		# CNN feature의 채널 수(input_dim)를 d_model 차원으로 매핑 (encoder용)
		self.input_projection = nn.Linear(self.input_dim, self.d_model)
		# Waypoint 데이터(output_dim)를 d_model 차원으로 매핑 (decoder용)
		self.output_projection = nn.Linear(self.output_dim, self.d_model)
		# Encoder와 Decoder에 각각 적용할 positional encoding
		self.pos_encoder = PositionalEncoding(self.d_model)
		self.pos_decoder = PositionalEncoding(self.d_model)
		# Transformer 모델 (batch_first=True 설정)
		# 인코더
		encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
		# 디코더
		decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
		self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

		# Transformer의 출력을 waypoint 차원(output_dim)으로 매핑
		self.fc_out = nn.Linear(self.d_model, self.output_dim)
		# ResNet 기반 perception 모듈 (resnet34 사용)
		self.perception = resnet34(pretrained=True)

		self.measurement_proj = nn.Linear(128, 512)

		#######################기존 TCP 추가#######################
		self.measurements = nn.Sequential(
							nn.Linear(1+2+6, 128),
							nn.ReLU(inplace=True),
							nn.Linear(128, 128),
							nn.ReLU(inplace=True),
						)
		
		self.speed_branch = nn.Sequential(
					nn.Linear(1000, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)
		
		self.expand_transformer_output = nn.Sequential(
			nn.Linear(512, 768),
			nn.ReLU(inplace=True),
			nn.Linear(768, 1024),
			nn.ReLU(inplace=True),
		)

		self.init_att = nn.Sequential(
				nn.Linear(128, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 29*8),
				nn.Softmax(1)
			)
		
		self.join_traj = nn.Sequential(
							nn.Linear(128+1000, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)

		self.join_ctrl = nn.Sequential(
							nn.Linear(128+512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)
		
		self.value_branch_ctrl = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)
		
		dim_out = 2

		self.policy_head = nn.Sequential(
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 256),
				nn.Dropout2d(p=0.5),
				nn.ReLU(inplace=True),
			)
		self.decoder_ctrl = nn.GRUCell(input_size=256+4, hidden_size=256)
		self.output_ctrl = nn.Sequential(
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
			)
		self.dist_mu = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())
		self.dist_sigma = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())

		self.init_att = nn.Sequential(
				nn.Linear(128, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 29*8),
				nn.Softmax(1)
			)

		self.wp_att = nn.Sequential(
				nn.Linear(256+256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 29*8),
				nn.Softmax(1)
			)

		self.merge = nn.Sequential(
				nn.Linear(512+256, 512),
				nn.ReLU(inplace=True),
				nn.Linear(512, 256),
			)
		
		self.value_branch_traj = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)
		
		self.for_pred_features_traj = nn.Linear(128, 512)
		self.for_transformerdecoder_out = nn.Linear(512, 256)
		self.for_transformerdecoder = nn.Linear(256, 512)
		self.output_traj = nn.Linear(256, 2)



	def forward(self, img, state, target_point):
#############################traj_branch#################################################
		# waypoints 32,4,2
		# 1. 이미지로부터 CNN feature 추출 (resnet의 출력)
		feature_emb, cnn_feature = self.perception(img)
		origin_cnn_feature = cnn_feature
		# cnn_feature의 shape: (batch, channels, H, W)
		outputs = {}
		outputs['pred_speed'] = self.speed_branch(feature_emb)
		measurement_feature = self.measurements(state)
		# 공간 차원을 flatten하여 (batch, H*W, channels)로 변환
		batch_size, channels, H, W = cnn_feature.size()
		cnn_feature = cnn_feature.view(batch_size, channels, H * W).transpose(1, 2)
		

		# measurement_feature 처리
		measurement_feature_proj = self.measurement_proj(measurement_feature)  # (batch, channels)
		measurement_feature_proj = measurement_feature_proj.unsqueeze(1)    # (batch, 1, channels)

		# 두 텐서 이어붙이기 (sequence dimension)
		j_traj = self.join_traj(torch.cat([feature_emb, measurement_feature], 1))
		outputs['pred_value_traj'] = self.value_branch_traj(j_traj)
		outputs['pred_features_traj'] = j_traj


		output_wp = list()
		like_hidden_state = list()


		x = torch.zeros(size=(j_traj.shape[0], 2), dtype=torch.float32).to(target_point.device)

		feature_for_decoder = self.for_transformerdecoder(j_traj)
		feature_for_decoder = feature_for_decoder.unsqueeze(0)  # inference용


		for _ in range(self.config.pred_len):
			x_in = torch.cat([x, target_point], dim=1)
			tgt_emb = self.output_projection(x_in)    # shape: (batch, tgt_len, d_model)
			tgt_emb = tgt_emb.unsqueeze(0)  # inference용
			# 왜 학습됨? python 버전이 달랐음

			pre_out = self.transformer_decoder(tgt_emb, feature_for_decoder)
			out = self.for_transformerdecoder_out(pre_out)
			out = out.squeeze(0)
			like_hidden_state.append(out)
			dx = self.output_traj(out)
			x = dx + x
			output_wp.append(x)

		pred_wp = torch.stack(output_wp, dim=1)
		outputs['pred_wp'] = pred_wp

############################control_branch##############################################
		measurement_feature = self.measurements(state)

		traj_hidden_state = torch.stack(like_hidden_state, dim=1) # Transformer에는 Hidden state가 존재하지 않음
		init_att = self.init_att(measurement_feature).view(-1, 1, 8, 29)
		feature_emb = torch.sum(origin_cnn_feature*init_att, dim=(2, 3))
		j_ctrl = self.join_ctrl(torch.cat([feature_emb, measurement_feature], 1))
		outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl)
		outputs['pred_features_ctrl'] = j_ctrl
		policy = self.policy_head(j_ctrl)
		outputs['mu_branches'] = self.dist_mu(policy)
		outputs['sigma_branches'] = self.dist_sigma(policy)

		x = j_ctrl
		mu = outputs['mu_branches']
		sigma = outputs['sigma_branches']
		future_feature, future_mu, future_sigma = [], [], []

		# initial hidden variable to GRU
		h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

		for _ in range(self.config.pred_len):
			x_in = torch.cat([x, mu, sigma], dim=1)
			h = self.decoder_ctrl(x_in, h) #GRU CELL
			wp_att = self.wp_att(torch.cat([h, traj_hidden_state[:,_, :]], 1)).view(-1, 1, 8, 29)
			new_feature_emb = torch.sum(origin_cnn_feature*wp_att, dim=(2, 3))
			merged_feature = self.merge(torch.cat([h, new_feature_emb], 1))
			dx = self.output_ctrl(merged_feature)
			x = dx + x

			policy = self.policy_head(x)
			mu = self.dist_mu(policy)
			sigma = self.dist_sigma(policy)
			future_feature.append(x)
			future_mu.append(mu)
			future_sigma.append(sigma)


		outputs['future_feature'] = future_feature
		outputs['future_mu'] = future_mu
		outputs['future_sigma'] = future_sigma


		return outputs

	def process_action(self, pred, command, speed, target_point):
		action = self._get_action_beta(pred['mu_branches'].view(1,2), pred['sigma_branches'].view(1,2))
		acc, steer = action.cpu().numpy()[0].astype(np.float64)
		if acc >= 0.0:
			throttle = acc
			brake = 0.0
		else:
			throttle = 0.0
			brake = np.abs(acc)

		throttle = np.clip(throttle, 0, 1)
		steer = np.clip(steer, -1, 1)
		brake = np.clip(brake, 0, 1)

		metadata = {
			'speed': float(speed.cpu().numpy().astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'command': command,
			'target_point': tuple(target_point[0].data.cpu().numpy().astype(np.float64)),
		}
		return steer, throttle, brake, metadata

	def _get_action_beta(self, alpha, beta):
		x = torch.zeros_like(alpha)
		x[:, 1] += 0.5
		mask1 = (alpha > 1) & (beta > 1)
		x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

		mask2 = (alpha <= 1) & (beta > 1)
		x[mask2] = 0.0

		mask3 = (alpha > 1) & (beta <= 1)
		x[mask3] = 1.0

		# mean
		mask4 = (alpha <= 1) & (beta <= 1)
		x[mask4] = alpha[mask4]/torch.clamp((alpha[mask4]+beta[mask4]), min=1e-5)

		x = x * 2 - 1

		return x

	def control_pid(self, waypoints, velocity, target):
		''' Predicts vehicle control with a PID controller.
		Args:
			waypoints (tensor): output of self.plan()
			velocity (tensor): speedometer input
		'''
		assert(waypoints.size(0)==1)
		waypoints = waypoints[0].data.cpu().numpy()
		target = target.squeeze().data.cpu().numpy()

		# flip y (forward is negative in our waypoints)
		waypoints[:,1] *= -1
		target[1] *= -1

		# iterate over vectors between predicted waypoints
		num_pairs = len(waypoints) - 1
		best_norm = 1e5
		desired_speed = 0
		aim = waypoints[0]
		for i in range(num_pairs):
			# magnitude of vectors, used for speed
			desired_speed += np.linalg.norm(
					waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

			# norm of vector midpoints, used for steering
			norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
			if abs(self.config.aim_dist-best_norm) > abs(self.config.aim_dist-norm):
				aim = waypoints[i]
				best_norm = norm

		aim_last = waypoints[-1] - waypoints[-2]

		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
		angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

		# choice of point to aim for steering, removing outlier predictions
		# use target point if it has a smaller angle or if error is large
		# predicted point otherwise
		# (reduces noise in eg. straight roads, helps with sudden turn commands)
		use_target_to_aim = np.abs(angle_target) < np.abs(angle)
		use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.config.angle_thresh and target[1] < self.config.dist_thresh)
		if use_target_to_aim:
			angle_final = angle_target
		else:
			angle_final = angle

		steer = self.turn_controller.step(angle_final)
		steer = np.clip(steer, -1.0, 1.0)

		speed = velocity[0].data.cpu().numpy()
		brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

		delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
		throttle = self.speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, self.config.max_throttle)
		throttle = throttle if not brake else 0.0

		metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_4': tuple(waypoints[3].astype(np.float64)),
			'wp_3': tuple(waypoints[2].astype(np.float64)),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'target': tuple(target.astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'angle_last': float(angle_last.astype(np.float64)),
			'angle_target': float(angle_target.astype(np.float64)),
			'angle_final': float(angle_final.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}

		return steer, throttle, brake, metadata


	def get_action(self, mu, sigma):
		action = self._get_action_beta(mu.view(1,2), sigma.view(1,2))
		acc, steer = action[:, 0], action[:, 1]
		if acc >= 0.0:
			throttle = acc
			brake = torch.zeros_like(acc)
		else:
			throttle = torch.zeros_like(acc)
			brake = torch.abs(acc)

		throttle = torch.clamp(throttle, 0, 1)
		steer = torch.clamp(steer, -1, 1)
		brake = torch.clamp(brake, 0, 1)

		return throttle, steer, brake #n개의 제어신호 예측 #두 개를 융합하여 제어 성능을 높힌다.