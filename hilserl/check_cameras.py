import cv2

def check_and_open_cameras(max_checks=10):
    """
    检查可用摄像头并同时打开所有可用设备的画面
    max_checks: 最大尝试的索引范围 (默认检查 0-9)
    """
    valid_caps = [] # 存储可用的 (索引, cap对象)

    print(f"正在扫描摄像头 (索引范围 0 - {max_checks-1})...")

    # --- 第一步：扫描并打开所有可用设备 ---
    for i in range(max_checks):
        # 尝试打开摄像头
        # 注意：在Windows上，使用 cv2.CAP_DSHOW 可能会让初始化更快，
        # 如果你发现扫描非常慢，可以将下一行改为: cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            # 尝试读取一帧以确保设备真的可用
            ret, frame = cap.read()
            if ret:
                print(f"[成功] 发现摄像头 ID: {i}")
                valid_caps.append((i, cap))
            else:
                print(f"[警告] 摄像头 ID {i} 无法读取画面，已释放。")
                cap.release()
        else:
            # 这里的 print 可以注释掉，以免输出太多干扰信息
            pass 

    # 如果没有找到摄像头
    if not valid_caps:
        print("未检测到任何可用摄像头。")
        return

    print(f"\n共找到 {len(valid_caps)} 个可用设备。按 'q' 键退出程序。\n")

    # --- 第二步：循环显示所有画面 ---
    try:
        while True:
            for index, cap in valid_caps:
                ret, frame = cap.read()
                
                if ret:
                    # 在画面上标记 ID，方便区分
                    text = f"Device ID: {index}"
                    cv2.putText(frame, text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 显示窗口，窗口名为 "Cam <ID>"
                    cv2.imshow(f'Cam {index}', frame)
            
            # 检测按键，按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生错误: {e}")
        
    finally:
        # --- 第三步：清理资源 ---
        print("正在关闭所有设备...")
        for index, cap in valid_caps:
            cap.release()
        cv2.destroyAllWindows()
        print("程序结束。")

if __name__ == "__main__":
    check_and_open_cameras()