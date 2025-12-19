1:采集数据之前，用run_real_teleop.py跑一下机械臂，找一下end_effector_bounds的值
2:采集数据时，第一轮采集完了最好确认一下采集的视频，尤其时曝光，很可能过曝
3:数据裁剪时，要记录一下裁剪的尺寸，后面check reward classifier要用
4:check reward时要用最新训练的模型，记得修改模型路径
5:rl训练配置文件中的摄像头尺寸，跟模型路径都要修改为最新的