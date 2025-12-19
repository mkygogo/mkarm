v4l2-ctl -d /dev/video0 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=300
v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=3
#v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1
#v4l2-ctl -d /dev/video0 --set-ctrl=exposure_time_absolute=350