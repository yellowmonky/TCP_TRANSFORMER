import argparse
import os
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

# TCP 모델이 이제 TransformerModel 기반이라고 가정합니다.
# TCP 모델의 forward는 (src, tgt)를 받도록 수정되어 있습니다.
from TCP.model import TransformerModel  
from TCP.data import CARLA_Data
from TCP.config import GlobalConfig

class TCP_planner(pl.LightningModule):
    def __init__(self, config, lr):
        super().__init__()
        self.lr = lr
        self.config = config
        self.model = TransformerModel(config)

        # path_to_conf_file ='log/TCP/best_epoch=14-val_loss=0.482.ckpt'
        # ckpt = torch.load(path_to_conf_file)
        # ckpt = ckpt["state_dict"]
        # new_state_dict = OrderedDict()
        # for key, value in ckpt.items():
        #     new_key = key.replace("model.","")
        #     new_state_dict[new_key] = value

        #     if "input_projection" in new_key:
        #         print(f"\nkey: {new_key}")
        #         print()
        #         print(value.shape)
			
        # self.model.load_state_dict(new_state_dict, strict = False)


        self._load_weight()
        # # TransformerModel의 입력/출력 차원은 상황에 맞게 설정합니다.
        # # 예를 들어, 입력은 state (speed, target_point, command)이고,
        # # 출력은 waypoints (예: 2차원 좌표 시퀀스)라고 가정합니다.
        # input_dim = 9   # speed(1) + target_point(2) + command(6)
        # output_dim = 2  # waypoint의 x, y 좌표
        # self.pred_len = config.pred_len  # 예측 시퀀스 길이
            
            #input_dim=input_dim, output_dim=output_dim, d_model=512, nhead=8, num_layers=6, dim_ff=2048)
        # 기존 weight loading 부분은 필요에 따라 유지 또는 수정합니다.
        # self._load_weight()  # 만약 이전 RL 가중치를 활용해야 한다면

    def _load_weight(self):
        rl_state_dict = torch.load(self.config.rl_ckpt, map_location='cpu')['policy_state_dict']
        # self._load_state_dict(self.model.value_branch_traj, rl_state_dict, 'value_head')
        # self._load_state_dict(self.model.value_branch_ctrl, rl_state_dict, 'value_head')
        self._load_state_dict(self.model.dist_mu, rl_state_dict, 'dist_mu')
        self._load_state_dict(self.model.dist_sigma, rl_state_dict, 'dist_sigma')

    def _load_state_dict(self, il_net, rl_state_dict, key_word):
        rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
        il_keys = il_net.state_dict().keys()
        assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
        new_state_dict = OrderedDict()
        for k_il, k_rl in zip(il_keys, rl_keys):
            new_state_dict[k_il] = rl_state_dict[k_rl]
        il_net.load_state_dict(new_state_dict)


    def forward(self, batch):
        pass

    # def forward(self, batch):
    #     """
    #     Transformer forward:
    #       - src: 인코더 입력. 여기서는 state 벡터(배치, 1, state_dim)
    #       - tgt: 디코더 입력. teacher forcing을 위해, ground truth waypoints를 한 칸 shift하여 사용.
    #     """
    #     front_img = batch['front_img']  # 기존에는 이미지도 사용했으나, 여기서는 state만 사용합니다.
    #     speed = batch['speed'].to(dtype=torch.float32).view(-1, 1) / 12.
    #     target_point = batch['target_point'].to(dtype=torch.float32)
    #     command = batch['target_command']
    #     state = torch.cat([speed, target_point, command], 1)  # (batch, 9)

    #     gt_waypoints = batch['waypoints']  # (batch, pred_len, 2)

    #     batch_size = state.shape[0]
    #     # src: 단일 토큰으로 구성 (state를 그대로 사용)
    #     src = state.unsqueeze(1)  # (batch, 1, state_dim)
    #     # tgt: teacher forcing용 입력. 정답 시퀀스를 오른쪽으로 shift (시작 토큰은 0 벡터로 채움)
    #     start_token = torch.zeros((batch_size, 1, gt_waypoints.shape[2]), 
    #                               device=gt_waypoints.device, dtype=gt_waypoints.dtype)
    #     tgt_input = torch.cat([start_token, gt_waypoints[:, :-1, :]], dim=1)  # (batch, pred_len, 2)
        
    #     # Transformer 모델 forward 호출: (batch, pred_len, output_dim)
    #     pred_waypoints = self.model(src, tgt_input)
    #     outputs = {'pred_wp': pred_waypoints}
    #     return outputs

    def training_step(self, batch, batch_idx):

        front_img = batch['front_img']
        speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
        target_point = batch['target_point'].to(dtype=torch.float32)
        command = batch['target_command']
		
        state = torch.cat([speed, target_point, command], 1)
        value = batch['value'].view(-1,1)
        feature = batch['feature']

        gt_waypoints = batch['waypoints']
        # gt_waypoints_cat = gt_waypoints.view(gt_waypoints.size(0), 1, -1)
        # gt_waypoints_cat = gt_waypoints_cat.float()
        pred = self.model(front_img, state, target_point) # TCP class 의 forward 호출
        # gt_waypoint 를 넣어줘야함

        ## 여기까지는 완벽함

		# 이후 로스 계산
        dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
        dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
        kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
        action_loss = torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
        speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
        value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
        feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) +F.mse_loss(pred['pred_features_ctrl'], feature))* self.config.features_weight

        future_feature_loss = 0
        future_action_loss = 0
        for i in range(self.config.pred_len):
            dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
            dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
            kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
            future_action_loss += torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
            future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
        future_feature_loss /= self.config.pred_len
        future_action_loss /= self.config.pred_len
        wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()
        loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss+ future_feature_loss + future_action_loss
        self.log('train_action_loss', action_loss.item())
        self.log('train_wp_loss_loss', wp_loss.item())
        self.log('train_speed_loss', speed_loss.item())
        self.log('train_value_loss', value_loss.item())
        self.log('train_feature_loss', feature_loss.item())
        self.log('train_future_feature_loss', future_feature_loss.item())
        self.log('train_future_action_loss', future_action_loss.item())
        return loss

        # # forward 호출 시, 입력은 batch 전체를 전달
        # pred = self.forward(batch)
        # gt_waypoints = batch['waypoints']  # (batch, pred_len, 2)
        # # 예시로 L1 loss 사용
        # wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints)
        # self.log('train_wp_loss', wp_loss.item())
        # return wp_loss

#Gru 에서 대회용 코드 트랜스포머 디코더를 사용해서 좋았다
# Trajectory만 잘 설정해봐라
# 제어 네트워크의 성능
# planning 네트워크의 성능이 더 좋아졌다.
# Transformer의 Hidden state가 없음
# Trajectory branch만 transformer
# 0. DECODER 3/23
# 1. 이미지 입력 3/24
# masking
# 레지넷의 출력을 cat해서 입력 
# 2. control branch 추가 3/28
# 3. 완료 3/30
# Q,K,V 에 관한 이해
# self attention, cross attention 차이

    def validation_step(self, batch, batch_idx):
        front_img = batch['front_img']
        speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
        target_point = batch['target_point'].to(dtype=torch.float32)
        command = batch['target_command']
        state = torch.cat([speed, target_point, command], 1)
        value = batch['value'].view(-1,1)
        feature = batch['feature']
        gt_waypoints = batch['waypoints']

        pred = self.model(front_img, state, target_point)

        dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
        dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
        kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
        action_loss = torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
        speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
        value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
        feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) +F.mse_loss(pred['pred_features_ctrl'], feature))* self.config.features_weight

        wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()

        B = batch['action_mu'].shape[0]
        batch_steer_l1 = 0 
        batch_brake_l1 = 0
        batch_throttle_l1 = 0
        for i in range(B):
            throttle, steer, brake = self.model.get_action(pred['mu_branches'][i], pred['sigma_branches'][i])
            batch_throttle_l1 += torch.abs(throttle-batch['action'][i][0])
            batch_steer_l1 += torch.abs(steer-batch['action'][i][1])
            batch_brake_l1 += torch.abs(brake-batch['action'][i][2])

        batch_throttle_l1 /= B
        batch_steer_l1 /= B
        batch_brake_l1 /= B

        future_feature_loss = 0
        future_action_loss = 0
        for i in range(self.config.pred_len-1):
            dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
            dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
            kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
            future_action_loss += torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
            future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
        future_feature_loss /= self.config.pred_len
        future_action_loss /= self.config.pred_len

        val_loss = wp_loss + batch_throttle_l1+5*batch_steer_l1+batch_brake_l1

        self.log("val_action_loss", action_loss.item(), sync_dist=True)
        self.log('val_speed_loss', speed_loss.item(), sync_dist=True)
        self.log('val_value_loss', value_loss.item(), sync_dist=True)
        self.log('val_feature_loss', feature_loss.item(), sync_dist=True)
        self.log('val_wp_loss_loss', wp_loss.item(), sync_dist=True)
        self.log('val_future_feature_loss', future_feature_loss.item(), sync_dist=True)
        self.log('val_future_action_loss', future_action_loss.item(), sync_dist=True)
        self.log('val_loss', val_loss.item(), sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'frequency': 1}]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='TCP', help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=60, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)

    # Config
    config = GlobalConfig()

    # Data
    train_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug=config.img_aug)
    val_set = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data)
    print(len(train_set))
    
    
    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    TCP_model = TCP_planner(config=config, lr=args.lr)



    # TCP_model = TCP_planner(config, args.lr)


    checkpoint_callback = ModelCheckpoint(
        save_weights_only=False, mode="min", monitor="val_loss", save_top_k=2, save_last=True,
        dirpath=args.logdir, filename="best_{epoch:02d}-{val_loss:.3f}"
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

    wandb_logger = WandbLogger(
        project="test",
        name='0406_Transformer',
        save_dir=args.logdir
    )
    
    wandb_logger.log_hyperparams({
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
    })

    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=args.logdir,
        gpus=args.gpus,
        accelerator='ddp',
        sync_batchnorm=True,
        plugins=DDPPlugin(find_unused_parameters=True),
        profiler='simple',
        benchmark=True,
        log_every_n_steps=1,
        flush_logs_every_n_steps=5,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=args.val_every,
        max_epochs=args.epochs,
        logger=wandb_logger,
        resume_from_checkpoint="log/TCP/epoch=34-last.ckpt"
    )

    trainer.fit(TCP_model, dataloader_train, dataloader_val)

