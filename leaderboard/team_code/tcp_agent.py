import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict

import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent

from TCP.model import TransformerModel
from TCP.config import GlobalConfig
from team_code.planner import RoutePlanner
import time


SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
	return 'TCPAgent'


class TCPAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		self.alpha = 0.3
		self.status = 0
		self.steer_step = 0
		self.last_moving_status = 0
		self.last_moving_step = -1
		self.last_steers = deque()

		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.config = GlobalConfig()
		self.net = TransformerModel(self.config)

		ckpt = torch.load(path_to_conf_file)
		ckpt = ckpt["state_dict"]
		new_state_dict = OrderedDict()
		for key, value in ckpt.items():
			new_key = key.replace("model.","")
			new_state_dict[new_key] = value

			if "input_projection" in new_key:
				print(f"\nkey: {new_key}")
				print()
				print(value.shape)

			
		self.net.load_state_dict(new_state_dict, strict = False)
		self.net.cuda()
		self.net.eval()

		self.takeover = False
		self.stop_time = 0
		self.takeover_time = 0

		self.save_path = None
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

		self.last_steers = deque()
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

			print (string)

			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			(self.save_path / 'rgb').mkdir()
			(self.save_path / 'meta').mkdir()
			(self.save_path / 'bev').mkdir()
			(self.save_path / 'path').mkdir()
		
		self._vehicle = None  # for waypoint


	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)
		self.path_log = []

		self.initialized = True

	def set_vehicle(self, vehicle): #for way point
		"""외부에서 에고 차량 객체를 할당받는 메소드."""
		self._vehicle = vehicle
		print("TCPAgent: 에고 차량이 할당되었습니다.")

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):
				return [
				{
					'type': 'sensor.camera.rgb',
					'x': -1.5, 'y': 0.0, 'z':2.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 900, 'height': 256, 'fov': 100,
					'id': 'rgb'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 0.0, 'y': 0.0, 'z': 50.0,
					'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
					'width': 512, 'height': 512, 'fov': 5 * 10.0,
					'id': 'bev'
					},	
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
					},
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'speed'
					}
				]

	def tick(self, input_data):
		self.step += 1

		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]

		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
				'rgb': rgb,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				'bev': bev
				}
		
		pos = self._get_position(result)
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos)
		result['next_command'] = next_cmd.value


		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		return result
	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		tick_data = self.tick(input_data)


		# 		# 차량의 현재 속도 (tick_data['speed'])를 확인 (예: m/s)
		# current_speed = tick_data['speed']
		# low_speed_threshold = 0.1  # 임계 속도 (예: 3 m/s 이하)
		# required_duration = 160    # 저속 상태가 유지되어야 할 시간 (예: 5초 이상)

		# # 저속 상태 체크: 차량 속도가 임계값보다 작으면 시작시간 기록
		# if current_speed < low_speed_threshold:
		# 	if self.low_speed_start is None:
		# 		self.low_speed_start = timestamp
		# 		force_throttle = False
		# 	elif timestamp - self.low_speed_start > required_duration:
		# 		force_throttle = True
		# 	else:
		# 		force_throttle = False
		# else:
		# 	self.low_speed_start = None
		# 	force_throttle = False
		# 	##########################################
		if self.step < self.config.seq_len:
			rgb = self._im_transform(tick_data['rgb']).unsqueeze(0)

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			
			return control

		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		command = tick_data['next_command']
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1
		cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
		speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
		speed = speed / 12
		rgb = self._im_transform(tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
										torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)
		state = torch.cat([speed, target_point, cmd_one_hot], 1)

		print('측정 시작')
		start = time.perf_counter()
		pred = self.net(rgb, state, target_point)
		print('현재시간 - 이전시간')
		# GPU 사용 중이라면 연산 종료까지 대기
		if torch.cuda.is_available():
			torch.cuda.synchronize()

		# 3) 측정 종료
		end = time.perf_counter()
		elapsed_ms = (end - start) * 1000
		print(f'Inference time: {elapsed_ms:.2f} ms')

		# --- 추가: 모델이 예측한 웨이포인트 시각화 ---
        # 예측된 웨이포인트를 self.predicted_waypoints로 저장 (배치 크기가 1이라고 가정)
		self.predicted_waypoints = pred['pred_wp'].squeeze(0).cpu().numpy()
		# 에고 차량의 현재 transform 정보를 가져옵니다.
		ego_transform = self._vehicle.get_transform()
		vehicle_location = ego_transform.location
		self.path_log.append({
			"frame": self.step,
			"x": vehicle_location.x,
			"y": vehicle_location.y,
			"z": vehicle_location.z
		})

		yaw_rad = math.radians(ego_transform.rotation.yaw) + (math.pi/2)  # 회전각을 라디안 단위로 변환

		# CARLA 월드 객체
		world = self._vehicle.get_world()

		# 각 웨이포인트를 월드 좌표로 변환하고 점을 그립니다.
		for wp in self.predicted_waypoints:
			# 예측된 좌표가 [dx, dy] 형태의 에고 차량 기준 상대 좌표라고 가정합니다.
			dx, dy = float(wp[0]), float(wp[1])
			
			# 월드 좌표 변환
			world_x = vehicle_location.x + (dx * math.cos(yaw_rad) - dy * math.sin(yaw_rad))
			world_y = vehicle_location.y + (dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad))
			# z 값은 에고 차량 높이에서 약간 올려줍니다 (예: 1미터 위)
			world_z = vehicle_location.z + 1

			# 변환된 좌표로 Location 객체 생성
			waypoint_location = carla.Location(x=world_x, y=world_y, z=world_z)
			
			# 디버그 API를 사용하여 점을 그립니다.
			world.debug.draw_point(
				waypoint_location,
				size=0.05,                     # 점의 크기 (원하는 크기로 조절)
				life_time=0.00001,              # 점이 화면에 유지되는 시간 (초); 매 프레임 갱신 시 적당한 값 선택
				color=carla.Color( r = 0, g = 0, b= 255)    # 점의 색상
			)
			#_____________________________________________________________________________


		steer_ctrl, throttle_ctrl, brake_ctrl, metadata = self.net.process_action(pred, tick_data['next_command'], gt_velocity, target_point)

		steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred['pred_wp'], gt_velocity, target_point)
		if brake_traj < 0.05: brake_traj = 0.0
		if throttle_traj > brake_traj: brake_traj = 0.0

		self.pid_metadata = metadata_traj
		control = carla.VehicleControl()


		# if force_throttle:
		# 	# 저속 상태가 일정 시간 지속된 경우: 스로틀 강제 1.0 적용
		# 	control.throttle = 1.0
		# 	control.steer = 0.0
		# 	control.brake = 0.0
		# 	self.pid_metadata['forced_throttle'] = True

		# else:
		if self.status == 0:
			self.alpha = 0.3
			self.pid_metadata['agent'] = 'traj'
			control.steer = np.clip(self.alpha*steer_ctrl + (1-self.alpha)*steer_traj, -1, 1)
			control.throttle = np.clip(self.alpha*throttle_ctrl + (1-self.alpha)*throttle_traj, 0, 0.75)
			control.brake = np.clip(self.alpha*brake_ctrl + (1-self.alpha)*brake_traj, 0, 1)
		else:
			self.alpha = 0.3
			self.pid_metadata['agent'] = 'ctrl'
			control.steer = np.clip(self.alpha*steer_traj + (1-self.alpha)*steer_ctrl, -1, 1)
			control.throttle = np.clip(self.alpha*throttle_traj + (1-self.alpha)*throttle_ctrl, 0, 0.75)
			control.brake = np.clip(self.alpha*brake_traj + (1-self.alpha)*brake_ctrl, 0, 1)


		self.pid_metadata['steer_ctrl'] = float(steer_ctrl)
		self.pid_metadata['steer_traj'] = float(steer_traj)
		self.pid_metadata['throttle_ctrl'] = float(throttle_ctrl)
		self.pid_metadata['throttle_traj'] = float(throttle_traj)
		self.pid_metadata['brake_ctrl'] = float(brake_ctrl)
		self.pid_metadata['brake_traj'] = float(brake_traj)

		if control.brake > 0.5:
			control.throttle = float(0)

		if len(self.last_steers) >= 20:
			self.last_steers.popleft()
		self.last_steers.append(abs(float(control.steer)))
		#chech whether ego is turning
		# num of steers larger than 0.1
		num = 0
		for s in self.last_steers:
			if s > 0.10:
				num += 1
		if num > 10:
			self.status = 1
			self.steer_step += 1

		else:
			self.status = 0

		self.pid_metadata['status'] = self.status



		if SAVE_PATH is not None and self.step % 10 == 0:
			self.save(tick_data)
		return control

	def save(self, tick_data):
		frame = self.step // 10

		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

		Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))

		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()
		
		with open(self.save_path / 'path.json', 'w', encoding='utf-8') as f:
			json.dump(self.path_log, f, indent=2)




	def destroy(self):
		del self.net
		torch.cuda.empty_cache()